/**
 * Polyscriptor PWA — Service Worker
 * Caches static assets for faster startup; API calls always go to network.
 */

const CACHE = 'polyscriptor-pwa-v3';
const STATIC = [
  '/demo',
  '/static/pwa/demo.html',
  '/static/pwa/demo.css',
  '/static/pwa/demo.js',
  '/static/pwa/manifest.json',
  '/static/pwa/icons/icon-192.png',
  '/static/pwa/icons/icon-512.png',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(STATIC)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);

  // API calls: always network-only (no caching)
  if (url.pathname.startsWith('/api/')) {
    e.respondWith(fetch(e.request).catch(() =>
      new Response(JSON.stringify({ detail: 'No server connection' }), {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      })
    ));
    return;
  }

  // Static assets: cache-first
  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request).then(resp => {
      if (resp.ok && STATIC.some(s => url.pathname === s || url.pathname.startsWith(s))) {
        caches.open(CACHE).then(c => c.put(e.request, resp.clone()));
      }
      return resp;
    }))
  );
});
