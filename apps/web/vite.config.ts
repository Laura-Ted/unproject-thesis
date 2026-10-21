import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

/**
 * Everything but the bundle comes from the API, including the websocket the
 * live view runs on. The viewer only ever asks for relative paths, so a
 * deployed build talks to whatever is serving it without a rebuild.
 */
const API = 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': { target: API, ws: true, changeOrigin: true },
      '/artifacts': { target: API, changeOrigin: true },
      '/datasets': { target: API, changeOrigin: true },
      '/uploads': { target: API, changeOrigin: true },
    },
  },
  build: {
    // three.js is most of the bundle and there is nothing to usefully split in
    // a single-view app, so the default 500 kB warning is only noise here.
    chunkSizeWarningLimit: 1500,
  },
  test: {
    // jsdom rather than node: the state races these tests are about live in a
    // component, and a component needs somewhere to render. Test-only - it is
    // a devDependency and never reaches the bundle.
    environment: 'jsdom',
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
  },
});
