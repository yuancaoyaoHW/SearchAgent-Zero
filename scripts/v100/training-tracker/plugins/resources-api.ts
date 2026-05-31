import type { Plugin } from 'vite'
import { readFileSync, writeFileSync } from 'fs'
import { resolve } from 'path'

const RESOURCES_PATH = resolve(__dirname, '../src/data/resources.json')

/**
 * Vite dev server plugin: exposes GET/POST /__api/resources
 * - GET: returns current resources.json
 * - POST: writes body to resources.json (persists to repo)
 */
export function resourcesApiPlugin(): Plugin {
  return {
    name: 'resources-api',
    configureServer(server) {
      server.middlewares.use('/__api/resources', (req, res) => {
        // CORS for dev
        res.setHeader('Access-Control-Allow-Origin', '*')
        res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        res.setHeader('Access-Control-Allow-Headers', 'Content-Type')

        if (req.method === 'OPTIONS') {
          res.statusCode = 204
          res.end()
          return
        }

        if (req.method === 'GET') {
          try {
            const data = readFileSync(RESOURCES_PATH, 'utf-8')
            res.setHeader('Content-Type', 'application/json')
            res.end(data)
          } catch (err) {
            res.statusCode = 500
            res.end(JSON.stringify({ error: 'Failed to read resources.json' }))
          }
          return
        }

        if (req.method === 'POST') {
          let body = ''
          req.on('data', (chunk) => { body += chunk })
          req.on('end', () => {
            try {
              // Validate JSON
              const parsed = JSON.parse(body)
              if (!Array.isArray(parsed)) {
                res.statusCode = 400
                res.end(JSON.stringify({ error: 'Expected an array' }))
                return
              }
              writeFileSync(RESOURCES_PATH, JSON.stringify(parsed, null, 2) + '\n')
              res.setHeader('Content-Type', 'application/json')
              res.end(JSON.stringify({ ok: true, count: parsed.length }))
            } catch (err) {
              res.statusCode = 400
              res.end(JSON.stringify({ error: 'Invalid JSON' }))
            }
          })
          return
        }

        res.statusCode = 405
        res.end(JSON.stringify({ error: 'Method not allowed' }))
      })
    },
  }
}
