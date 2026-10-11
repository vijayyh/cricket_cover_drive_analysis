// All backend calls go through relative /api URLs (proxied by Vite in dev).

async function json(res) {
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || detail
    } catch {
      /* not JSON */
    }
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${res.status})`)
  }
  return res.json()
}

export const mediaUrl = (name) => `/api/media/${encodeURIComponent(name)}`

export const getHealth = () => fetch('/api/health').then(json)
export const getModelReport = () => fetch('/api/model').then(json)
export const getResults = (limit = 9) => fetch(`/api/results?limit=${limit}`).then(json)
export const getResult = (id) => fetch(`/api/results/${id}`).then(json)

/**
 * Upload with real progress events (fetch has no upload progress).
 * onProgress(fraction 0..1) fires while bytes are sent; resolves with the analysis JSON.
 */
export function analyzeFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/analyze')
    xhr.responseType = 'json'
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total)
    xhr.upload.onload = () => onProgress?.(1)
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response)
      else reject(new Error(xhr.response?.detail || `Analysis failed (${xhr.status})`))
    }
    xhr.onerror = () => reject(new Error('Cannot reach the analysis server. Is the backend running on port 8000?'))
    const form = new FormData()
    form.append('file', file)
    xhr.send(form)
  })
}
