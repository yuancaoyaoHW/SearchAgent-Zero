import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { resourcesApiPlugin } from './plugins/resources-api'

export default defineConfig({
  plugins: [react(), tailwindcss(), resourcesApiPlugin()],
  server: {
    port: 3000,
    open: true,
  },
})
