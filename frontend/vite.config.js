import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The UI calls the FastAPI backend through relative /api URLs.
// In dev, Vite proxies them to uvicorn; in production FastAPI serves the built UI itself.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
