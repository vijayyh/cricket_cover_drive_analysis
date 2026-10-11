"""
train_shot_classifier.py
========================
Trains and *honestly* evaluates the cricket shot-type recogniser on the real
image dataset in data/shot_dataset (5 classes x 100 photos, each photo shipped
with 6 augmented copies: main, brightness, +30, -30, shear, salt&pepper).

Two branches are trained and then fused:
  * CNN  — ResNet-18 fine-tuned on the grayscale batsman crops
  * Pose — engineered MediaPipe-pose geometry -> gradient-boosted / SVM model

Evaluation rule: photos are grouped by SOURCE CLIP and a whole clip (with all
6 augmented copies of each of its photos) always lands in the same split.
Splitting copies of a photo, or neighbouring frames of the same video, across
train/val lets the model recognise what it has already seen and inflates
accuracy (the previous train_shot_model.py split individual images at random).

Run from the project root:

    python train_shot_classifier.py              # everything (~1 h on a laptop CPU)
    python train_shot_classifier.py --stage pose # just one stage

Outputs
  models/shot_cnn.pt               fine-tuned CNN (state dict + metadata)
  models/shot_pose.pkl             pose classifier pipeline
  models/shot_recognizer.json      class names + fusion weight
  models/shot_classifier_report.json   honest metrics, confusion matrices
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = "data/shot_dataset"
CACHE_DIR = "data/cache"
POSE_CACHE = os.path.join(CACHE_DIR, "pose_landmarks.npz")
HOLDOUT_CACHE = os.path.join(CACHE_DIR, "cnn_holdout_probs.npz")
POSE_OOF_CACHE = os.path.join(CACHE_DIR, "pose_oof_probs.npz")
SOURCE_CACHE = os.path.join(CACHE_DIR, "source_groups.npz")
CNN_PATH = "models/shot_cnn.pt"
POSE_PATH = "models/shot_pose.pkl"
META_PATH = "models/shot_recognizer.json"
REPORT_PATH = "models/shot_classifier_report.json"

SEED = 42
N_FOLDS = 5
HOLDOUT_FOLD = 0


def clean_class_name(folder: str) -> str:
    name = folder.split(". ", 1)[-1]
    return "Leg Glance" if name == "Leg Glance Shot" else name


# ─────────────────────────────────────────────────────────────────────────────
# Dataset index + grouped folds
# ─────────────────────────────────────────────────────────────────────────────

def build_index():
    folders = sorted(os.listdir(DATA_DIR))
    paths, labels, groups, augs = [], [], [], []
    for ci, folder in enumerate(folders):
        for f in sorted(os.listdir(os.path.join(DATA_DIR, folder))):
            m = re.match(r"(\d+)_[a-z]+_(.+)\.jpg$", f)
            if not m:
                continue
            paths.append(f"{folder}/{f}")
            labels.append(ci)
            groups.append(ci * 1000 + int(m.group(1)))   # one group per original photo
            augs.append(m.group(2))
    return {
        "classes": [clean_class_name(f) for f in folders],
        "paths": np.array(paths), "labels": np.array(labels),
        "groups": np.array(groups), "augs": np.array(augs),
    }


def source_groups(idx):
    """
    Many photos in the dataset are consecutive frames of the same video clip
    (e.g. Leg Glance 11-22). Grouping only by photo would still put
    near-identical frames on both sides of the split, so photos are clustered
    into source clips: neighbouring photo ids that look alike (ImageNet
    ResNet-18 embedding similarity > 0.85), plus any near-duplicate pair (> 0.95).
    """
    if os.path.exists(SOURCE_CACHE):
        d = np.load(SOURCE_CACHE)
        if list(d["paths"]) == list(idx["paths"]):
            return d["source"]
    import torch
    from PIL import Image
    from torchvision import models, transforms

    main_rows = np.where(idx["augs"] == "main")[0]
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    net.fc = torch.nn.Identity()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.inference_mode():
        emb = torch.cat([net(torch.stack([tf(Image.open(os.path.join(DATA_DIR, p)).convert("RGB"))
                                          for p in idx["paths"][main_rows[s:s + 64]]]))
                         for s in range(0, len(main_rows), 64)]).numpy()
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    sim = emb @ emb.T

    parent = list(range(len(main_rows)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    labels, photo = idx["labels"][main_rows], idx["groups"][main_rows]
    for c in np.unique(labels):
        rows = np.where(labels == c)[0]
        rows = rows[np.argsort(photo[rows])]
        for a in range(len(rows)):
            for b in range(len(rows)):
                near = 0 < b - a <= 3 and sim[rows[a], rows[b]] > 0.85
                if a != b and (near or sim[rows[a], rows[b]] > 0.95):
                    parent[find(rows[a])] = find(rows[b])
    by_photo = {photo[i]: find(i) for i in range(len(main_rows))}
    source = np.array([by_photo[g] for g in idx["groups"]])
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(SOURCE_CACHE, paths=idx["paths"], source=source)
    return source


def grouped_folds(idx):
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
    return list(sgkf.split(idx["paths"], idx["labels"], idx["source"]))


# ─────────────────────────────────────────────────────────────────────────────
# Stage: pose landmark cache
# ─────────────────────────────────────────────────────────────────────────────

def stage_pose_cache(idx):
    if os.path.exists(POSE_CACHE):
        print(f"[pose] cache exists -> {POSE_CACHE}")
        return
    import cv2
    from src.shot_recognizer import detect_pose_static
    os.makedirs(CACHE_DIR, exist_ok=True)
    lm, wl, found = [], [], []
    t0 = time.time()
    for i, p in enumerate(idx["paths"]):
        res = detect_pose_static(cv2.imread(os.path.join(DATA_DIR, p)))
        found.append(res is not None)
        lm.append(res[0] if res else np.zeros((33, 4), np.float32))
        wl.append(res[1] if res else np.zeros((33, 4), np.float32))
        if i % 300 == 0:
            print(f"[pose] {i}/{len(idx['paths'])}  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(POSE_CACHE, paths=idx["paths"], img_lm=np.stack(lm),
                        world_lm=np.stack(wl), found=np.array(found))
    print(f"[pose] detected {sum(found)}/{len(found)} -> {POSE_CACHE}")


def load_pose_features(idx):
    from src.shot_pose import pose_features_with_mirror, pose_is_usable
    d = np.load(POSE_CACHE)
    assert list(d["paths"]) == list(idx["paths"]), "pose cache is stale — delete data/cache/"
    usable = np.array([bool(f) and pose_is_usable(l) for f, l in zip(d["found"], d["img_lm"])])
    feats = np.zeros((len(usable), 2, len(pose_features_with_mirror(d["img_lm"][usable.argmax()],
                                                                       d["world_lm"][usable.argmax()])[0])),
                     np.float32)
    for i in np.where(usable)[0]:
        feats[i] = pose_features_with_mirror(d["img_lm"][i], d["world_lm"][i])
    return feats, usable


# ─────────────────────────────────────────────────────────────────────────────
# Stage: pose classifier (grouped 5-fold CV)
# ─────────────────────────────────────────────────────────────────────────────

def make_pose_model(kind: str):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(C=3.0, gamma="scale", probability=True, random_state=SEED))
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=4000))
    if kind == "extratrees":
        return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                                              l2_regularization=1.0, random_state=SEED)
    raise ValueError(kind)


def _fit_pose(kind, X2, y, rows):
    """Fit on rows (each contributes the pose and its mirror)."""
    X = np.concatenate([X2[rows, 0], X2[rows, 1]])
    return make_pose_model(kind).fit(X, np.concatenate([y[rows], y[rows]]))


def _proba_tta(model, X2, rows):
    return (model.predict_proba(X2[rows, 0]) + model.predict_proba(X2[rows, 1])) / 2


def stage_pose_cv(idx, folds):
    X2, usable = load_pose_features(idx)
    y, main = idx["labels"], idx["augs"] == "main"
    k = len(idx["classes"])
    results = {}
    oof_all = {}
    for kind in ["logreg", "svm", "extratrees", "hgb"]:
        oof = np.full((len(y), k), np.nan, np.float32)
        t0 = time.time()
        for tr, te in folds:
            model = _fit_pose(kind, X2, y, tr[usable[tr]])
            te_u = te[usable[te]]
            oof[te_u] = _proba_tta(model, X2, te_u)
        ev = main & usable
        acc = float((oof[ev].argmax(1) == y[ev]).mean())
        results[kind] = acc
        oof_all[kind] = oof
        print(f"[pose-cv] {kind:<11} acc on 'main' photos with a usable pose: {acc:.3f}"
              f"  ({ev.sum()} photos, {time.time() - t0:.0f}s)", flush=True)
    best = max(results, key=results.get)
    print(f"[pose-cv] best pose model: {best}")
    np.savez_compressed(POSE_OOF_CACHE, oof=oof_all[best], usable=usable, best=best,
                        **{f"acc_{k_}": v for k_, v in results.items()})
    return best, results


def stage_pose_final(idx):
    d = np.load(POSE_OOF_CACHE, allow_pickle=True)
    best = str(d["best"])
    X2, usable = load_pose_features(idx)
    import joblib
    model = _fit_pose(best, X2, idx["labels"], np.where(usable)[0])
    joblib.dump({"model": model, "kind": best, "classes": idx["classes"]}, POSE_PATH)
    print(f"[pose-final] {best} fitted on {usable.sum()} poses -> {POSE_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage: CNN
# ─────────────────────────────────────────────────────────────────────────────

def _load_images(idx):
    from PIL import Image
    imgs = []
    for p in idx["paths"]:
        with Image.open(os.path.join(DATA_DIR, p)) as im:
            imgs.append(im.convert("L"))
    return imgs


def train_cnn(images, labels, rows, num_classes, epochs, img_size, eval_fn=None):
    import torch
    import torch.nn as nn
    from src.shot_cnn import build_model, train_transform

    torch.manual_seed(SEED)
    torch.set_num_threads(os.cpu_count() or 4)
    model = build_model(num_classes).to(memory_format=torch.channels_last)
    tf = train_transform(img_size)

    n_groups = len(rows) // 6 if len(rows) >= 6 else len(rows)
    per_epoch = min(len(rows), 3 * n_groups)
    batch = 32
    steps = epochs * ((per_epoch + batch - 1) // batch)

    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": backbone, "lr": 3e-4}, {"params": head, "lr": 1e-3}],
                            weight_decay=0.02)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-4, 1e-3], total_steps=steps,
                                                pct_start=0.12, anneal_strategy="cos")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    rng = np.random.default_rng(SEED)

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        order = rng.permutation(rows)[:per_epoch]
        total, correct, loss_sum = 0, 0, 0.0
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            x = torch.stack([tf(images[i]) for i in b]).to(memory_format=torch.channels_last)
            yb = torch.as_tensor(labels[b])
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(b)
            correct += int((out.argmax(1) == yb).sum())
            total += len(b)
        msg = f"[cnn] epoch {ep:>2}/{epochs}  loss {loss_sum / total:.3f}  train-acc {correct / total:.3f}"
        if eval_fn is not None and (ep == epochs or ep % 3 == 0):
            msg += f"  holdout-acc {eval_fn(model):.3f}"
        print(f"{msg}  ({time.time() - t0:.0f}s)", flush=True)
    return model.eval()


def stage_cnn_holdout(idx, folds, epochs, img_size):
    from src.shot_cnn import predict_proba
    images = _load_images(idx)
    y, main = idx["labels"], idx["augs"] == "main"
    tr, te = folds[HOLDOUT_FOLD]
    te_main = te[main[te]]
    print(f"[cnn-holdout] train views {len(tr)} | holdout photos {len(te_main)}")

    def eval_fn(model):
        p = predict_proba(model, [images[i] for i in te_main], img_size)
        return float((p.argmax(1) == y[te_main]).mean())

    model = train_cnn(images, y, tr, len(idx["classes"]), epochs, img_size, eval_fn)
    probs = predict_proba(model, [images[i] for i in te], img_size)
    np.savez_compressed(HOLDOUT_CACHE, rows=te, probs=probs)
    print(f"[cnn-holdout] saved holdout probabilities -> {HOLDOUT_CACHE}")


def stage_cnn_final(idx, epochs, img_size):
    import torch
    images = _load_images(idx)
    rows = np.arange(len(images))
    model = train_cnn(images, idx["labels"], rows, len(idx["classes"]), epochs, img_size)
    os.makedirs("models", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": idx["classes"],
                "img_size": img_size, "arch": "resnet18"}, CNN_PATH)
    print(f"[cnn-final] saved -> {CNN_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage: fusion + report
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(y_true, y_pred, classes):
    from sklearn.metrics import classification_report, confusion_matrix
    rep = classification_report(y_true, y_pred, labels=range(len(classes)), target_names=classes,
                                output_dict=True, zero_division=0)
    return {
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "n": int(len(y_true)),
        "per_class_f1": {c: round(rep[c]["f1-score"], 3) for c in classes},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(classes))).tolist(),
    }


def fuse(p_cnn, p_pose, has_pose, w):
    """Weighted geometric mean of the branches; CNN only where no pose."""
    log = np.log(np.clip(p_cnn, 1e-6, 1))
    log_pose = np.log(np.clip(np.nan_to_num(p_pose, nan=1.0), 1e-6, 1))
    mix = np.where(has_pose[:, None], (1 - w) * log + w * log_pose, log)
    e = np.exp(mix - mix.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def stage_report(idx, folds):
    classes = idx["classes"]
    y, augs = idx["labels"], idx["augs"]
    h = np.load(HOLDOUT_CACHE)
    rows, p_cnn = h["rows"], h["probs"]
    po = np.load(POSE_OOF_CACHE, allow_pickle=True)
    p_pose, usable = po["oof"][rows], po["usable"][rows]
    main = augs[rows] == "main"
    yt = y[rows]

    # Choose the fusion weight on the non-'main' holdout copies, report on 'main'
    # photos so the weight isn't tuned on the numbers we report.
    tune = ~main
    best_w, best_acc = 0.0, -1.0
    for w in np.round(np.arange(0, 0.75, 0.05), 2):
        acc = float((fuse(p_cnn[tune], p_pose[tune], usable[tune], w).argmax(1) == yt[tune]).mean())
        if acc > best_acc + 1e-9:
            best_w, best_acc = float(w), acc
    p_fused = fuse(p_cnn, p_pose, usable, best_w)

    report = {
        "dataset": {
            "classes": classes,
            "original_photos": int(len(np.unique(idx["groups"]))),
            "source_clips": int(len(np.unique(idx["source"]))),
            "images_including_augmented_copies": int(len(y)),
            "pose_detected_fraction": round(float(np.load(POSE_OOF_CACHE)["usable"].mean()), 3),
        },
        "protocol": ("Source-clip grouped split: near-identical frames from the same clip, and every "
                     "augmented copy, stay on the same side of the split. "
                     f"CNN: fold {HOLDOUT_FOLD} of a {N_FOLDS}-fold StratifiedGroupKFold held out; "
                     "pose model: full grouped 5-fold CV. Accuracy reported on the unaugmented "
                     "('main') holdout photos; fusion weight tuned on the augmented holdout copies."),
        "baseline_previous_landmark_model": "≈30% on real photos (it was trained on synthetic numbers)",
        "pose_model_cv_accuracy": {k[4:]: round(float(po[k]), 4) for k in po.files if k.startswith("acc_")},
        "pose_model_selected": str(po["best"]),
        "holdout": {
            "cnn_only": _metrics(yt[main], p_cnn[main].argmax(1), classes),
            "pose_only_on_detected": _metrics(yt[main & usable], p_pose[main & usable].argmax(1), classes),
            "fused": _metrics(yt[main], p_fused[main].argmax(1), classes),
            "fused_all_augmented_copies": _metrics(yt, p_fused.argmax(1), classes),
        },
        "fusion_pose_weight": best_w,
        "generated": time.strftime("%Y-%m-%d %H:%M"),
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.makedirs("models", exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "pose_weight": best_w,
                   "holdout_accuracy": report["holdout"]["fused"]["accuracy"]}, f, indent=2)

    hd = report["holdout"]
    print("\n" + "=" * 62)
    print("  HONEST HOLDOUT RESULTS (photos never seen in training)")
    print("=" * 62)
    print(f"  CNN only              : {hd['cnn_only']['accuracy']:.1%}  (n={hd['cnn_only']['n']})")
    print(f"  Pose only (detected)  : {hd['pose_only_on_detected']['accuracy']:.1%}  "
          f"(n={hd['pose_only_on_detected']['n']})")
    print(f"  Fused (pose w={best_w:.2f})  : {hd['fused']['accuracy']:.1%}  (n={hd['fused']['n']})")
    print(f"  Chance level          : {1 / len(classes):.0%}")
    print("=" * 62)
    print(f"[report] -> {REPORT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "pose", "pose-cv", "cnn-holdout", "report", "final", "cnn-final"])
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--img-size", type=int, default=192)
    args = ap.parse_args()

    idx = build_index()
    idx["source"] = source_groups(idx)
    folds = grouped_folds(idx)
    print(f"[data] {len(idx['paths'])} images | {len(np.unique(idx['groups']))} original photos | "
          f"{len(np.unique(idx['source']))} source clips | classes {idx['classes']}")

    s = args.stage
    if s in ("all", "pose"):
        stage_pose_cache(idx)
    if s in ("all", "pose-cv"):
        stage_pose_cv(idx, folds)
    if s in ("all", "cnn-holdout"):
        stage_cnn_holdout(idx, folds, args.epochs, args.img_size)
    if s in ("all", "report"):
        stage_report(idx, folds)
    if s in ("all", "final"):
        stage_pose_final(idx)
    if s in ("all", "final", "cnn-final"):
        stage_cnn_final(idx, args.epochs, args.img_size)


if __name__ == "__main__":
    main()
