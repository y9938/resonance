import type { Job, JobType } from "./api/types";
import * as stt from "./stt/state.svelte";
import * as tts from "./tts/state.svelte";
import { restoreSystem } from "./stt/capture.svelte";
import { history } from "./jobs/state.svelte";
import { switchTab } from "./settings.svelte";
export async function selectJob(id: string, type: JobType) {
  stt.invalidate();
  tts.invalidate();
  history.open = false;
  switchTab(type);
  if (type === "stt") {
    await stt.restore(id);
    restoreSystem();
  } else await tts.restore(id);
}
export function openBatch(id: string, jobs: Job[]) {
  stt.clearBatch();
  stt.setBatch(id, jobs);
  if (jobs.some((j) => ["queued", "running"].includes(j.state))) {
    localStorage.setItem("resonance_stt_active_batch_id", id);
    stt.pollBatch();
  }
  switchTab("stt");
  history.open = false;
}
export async function recover() {
  const a = localStorage.getItem("resonance_stt_active_job_id"),
    b = localStorage.getItem("resonance_tts_active_job_id");
  await Promise.all([
    a ? stt.restore(a, true) : null,
    b ? tts.restore(b) : null,
  ]);
  restoreSystem();
  await stt.restoreBatch();
}
