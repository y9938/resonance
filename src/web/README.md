# Frontend

Svelte 5 owns rendering. TypeScript domain modules own state and transport:

- `stt/state.svelte.ts`: uploads, batches, selection, durable segments, ephemeral
  previews, SSE replay/reconnect and status recovery. Selection generations reject
  stale responses. Batch rows use `(batch_id, batch_index)`, including before a
  server job ID exists.
- `stt/capture.svelte.ts`, `live-transport.ts`, `pcm.ts`: browser capture,
  System Audio requests, ordered chunk acknowledgements/retries/backpressure,
  and WAV encoding. Components do not own transport.
- `tts/state.svelte.ts`: catalog selection, drafts, synthesis and job recovery.
- `jobs/state.svelte.ts`: history pagination/grouping and row presentation.
- `settings.svelte.ts`, `i18n/`: preferences, configuration and all three locales.
- `components/`: declarative views; `LanguagePicker` alone owns Tom Select's
  generated DOM. It imports the pinned 2.6.2 base package (no plugins), creates
  once after mount, updates options and destroys on unmount.
- `session.ts`: history selection and reload recovery across the two domains.

Local paths are request inputs only. They are not persisted in frontend job state
or localStorage. The backend remains authoritative for authorization and DTOs.

Use `./r dev-deps`, `./r dev`, `./r check` and `./r test` from the repository root.
Development documents/API/SSE are served by FastAPI; Vite only serves modules and
HMR on a free loopback port selected by the runner. No `/api` proxy or extra trusted origin is involved.

Production uses Vite's generated `dist/web/index.html` with hashed assets, served by
FastAPI. HTML/static source files revalidate; hashed `/assets/` files are immutable.
Root-level `dist` and `node_modules` are ignored. `src/web/public` contains source
icons and manifest. Vite collects bundled dependencies' license texts in
`dist/web/third-party/licenses.md` using Vite's built-in `build.license` support.
`npm run build` builds these assets for `./r serve-local`; Node is only needed for setup/builds.
Docker and the macOS launcher use these built assets.

The existing global CSS is retained to preserve the visual design. There is no
animation library; animation and further CSS ownership changes are separate work.

Frontend tooling (`package.json`, lockfile, Vite and TypeScript configuration) lives
at the repository root. Vite uses `src/web` as its root and writes `dist/web`.
The macOS client sources live alongside web sources in `src/macos`.
