import { defineConfig } from 'vite'
import { resolve } from 'path'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { resourcesApiPlugin } from './plugins/resources-api'

export default defineConfig({
  plugins: [react(), tailwindcss(), resourcesApiPlugin()],
  server: {
    port: 3000,
    open: true,
    fs: {
      allow: [resolve(__dirname, '../../../..')],
    },
  },
})
