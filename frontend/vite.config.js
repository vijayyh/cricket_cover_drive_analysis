import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The UI calls the FastAPI backend through relative /api URLs.
// In dev, Vite proxies them to uvicorn; in production FastAPI serves the built UI itself.
export default defineConfig({
  plugins: [react()],
  server: {
    // Bind to IPv4 explicitly: Node resolves "localhost" to ::1 (IPv6 only) while
    // uvicorn listens on 127.0.0.1, so browsers could reach one server but not the other.
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
