import adapter from '@sveltejs/adapter-static';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  kit: {
    // SPA served by the FastAPI backend (polyscriptor serve).
    adapter: adapter({ fallback: 'index.html' }),
    paths: { base: '' }
  }
};

export default config;
