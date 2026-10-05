import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

const port = Number(process.env.RESONANCE_PORT || 8000);
const origin = process.env.RESONANCE_VITE_ORIGIN || "http://127.0.0.1:5173";
export default defineConfig({
  root: "src/web",
  build: {
    outDir: "../../dist/web",
    emptyOutDir: true,
    license: { fileName: "third-party/licenses.md" },
  },
  plugins: [svelte()],
  server: {
    host: "127.0.0.1",
    port: Number(new URL(origin).port),
    strictPort: true,
    origin,
    cors: { origin: [`http://localhost:${port}`, `http://127.0.0.1:${port}`] },
  },
});
