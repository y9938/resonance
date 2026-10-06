import { finishCapture } from "./capture.svelte";
import type { AudioInput, Job, JobEvent, Segment } from "../api/types";
import { request, status, downloadText } from "../api/client";
import { settings, toast } from "../settings.svelte";
import { t } from "../i18n/state.svelte";
import { refreshHistory, filename, sorted } from "../jobs/state.svelte";
import {
  mergeAdjacentSttTexts,
  buildSttBlocksText,
  formatLocalizedSpeakerTags,
} from "./transcript";

const activeKey = "resonance_stt_active_job_id";
const batchKey = "resonance_stt_active_batch_id";
const selectedKey = "resonance_stt_selected_batch_job_id";
export const stt = $state({
  jobId: null as string | null,
  selectedId: null as string | null,
  filename: "",
  segments: [] as Segment[],
  previews: {} as Record<string, { generation: number; text: string }>,
  closed: {} as Record<string, number>,
  lastSeq: 0,
  current: 0,
  total: 0,
  label: null as string | null,
  complete: false,
  progress: false,
  result: false,
  busy: false,
  error: "",
  liveSource: "" as string,
  startedAt: 0,
  view:
    localStorage.getItem("resonance_stt_view_mode") === "continuous"
      ? "continuous"
      : "blocks",
  batchId: null as string | null,
  jobs: [] as Job[],
});
let epoch = 0;
let stream: EventSource | null = null;
let finishTimer: ReturnType<typeof setTimeout> | undefined;
let pollTimer: ReturnType<typeof setInterval> | undefined;
let batchEpoch = 0;
export function invalidate() {
  epoch++;
  stream?.close();
  stream = null;
  clearTimeout(finishTimer);
}
function active(id: string | null) {
  stt.jobId = id;
  if (id) localStorage.setItem(activeKey, id);
  else localStorage.removeItem(activeKey);
}
export function reset(keepResult = false) {
  stt.progress = false;
  stt.busy = false;
  stt.current = 0;
  stt.total = 0;
  stt.label = null;
  stt.complete = false;
  if (!keepResult) {
    stt.result = false;
    stt.segments = [];
    stt.previews = {};
    stt.closed = {};
    stt.filename = "";
    stt.liveSource = "";
  }
}
export function hide() {
  invalidate();
  active(null);
  reset(true);
  refreshHistory();
}
export function cancel() {
  const id = stt.jobId;
  finishCapture(id);
  invalidate();
  active(null);
  reset();
  if (id)
    void request(`/jobs/${encodeURIComponent(id)}/cancel`, {
      method: "POST",
    }).catch(() => {});
  refreshHistory();
}
function fail(message: string) {
  finishCapture(stt.jobId);
  stt.error = message;
  active(null);
  stream?.close();
  stream = null;
  reset();
  refreshHistory();
}
export function copyText(segments = stt.segments) {
  return formatLocalizedSpeakerTags(
    stt.view === "blocks"
      ? buildSttBlocksText(segments)
      : mergeAdjacentSttTexts(segments),
  );
}
export function displayText() {
  const preview = Object.values(stt.previews)
    .map((p) => p.text)
    .filter(Boolean)
    .join(" ");
  return [copyText(), preview]
    .filter(Boolean)
    .join(stt.view === "blocks" ? "\n\n" : " ");
}
export function setView(view: "blocks" | "continuous") {
  stt.view = view;
  localStorage.setItem("resonance_stt_view_mode", view);
}
export function progressText() {
  return stt.label
    ? t(stt.label)
    : stt.total > 0
      ? Math.round(percent()) + "%"
      : t("progressProcessing");
}
export function percent() {
  return stt.complete
    ? 100
    : stt.label === "progressUploading" || stt.label === "progressStarting"
      ? 5
      : stt.total > 0
        ? Math.max(0, Math.min(100, (stt.current / stt.total) * 100))
        : 0;
}
export function applyStatus(job: Job) {
  stt.segments = (job.result?.segments || []).filter(
    (s) =>
      Number.isFinite(s.start) &&
      Number.isFinite(s.end) &&
      typeof s.text === "string",
  );
  stt.previews = {};
  stt.closed = {};
  stt.filename = job.result?.filename || job.filename || "";
  stt.lastSeq = job.last_event_seq || 0;
  stt.current = Number(job.progress_current) || 0;
  stt.total = Number(job.progress_total) || 0;
  stt.liveSource = job.result?.source || "";
  stt.startedAt = job.started_at || 0;
  stt.result = true;
  stt.progress = !stt.liveSource;
  stt.busy = job.state === "running" || job.state === "queued";
  stt.complete = job.state === "completed";
  stt.label = stt.complete
    ? "progressComplete"
    : job.state === "queued"
      ? "jobsStateQueued"
      : stt.total <= 0
        ? "progressProcessing"
        : null;
}
function finish(token: number) {
  finishCapture(stt.jobId);
  stt.complete = true;
  stt.label = "progressComplete";
  stt.busy = false;
  stream?.close();
  stream = null;
  active(null);
  refreshHistory();
  finishTimer = setTimeout(() => {
    if (token === epoch) reset(true);
  }, 800);
}
export function subscribe(id: string, token = epoch) {
  stream?.close();
  const connection = new EventSource(
    `/api/jobs/${encodeURIComponent(id)}/events?after=${stt.lastSeq}`,
  );
  stream = connection;
  connection.onmessage = (message) => {
    if (token !== epoch || stream !== connection) return;
    let event: JobEvent;
    try {
      event = JSON.parse(message.data);
    } catch {
      return;
    }
    if (event.seq != null) {
      if (event.seq <= stt.lastSeq) return;
      stt.lastSeq = event.seq;
    }
    switch (event.type) {
      case "start":
        stt.current = 0;
        stt.total = event.total || 0;
        stt.label =
          event.stage === "diarization"
            ? "progressDiarizing"
            : "progressProcessing";
        break;
      case "progress": {
        stt.current = event.current || 0;
        stt.total = event.total || 0;
        stt.label =
          event.stage === "diarization"
            ? "progressDiarizing"
            : stt.total <= 0
              ? "progressProcessing"
              : null;
        if (event.segment) {
          const segment = event.segment;
          const source = segment.source || "mic";
          if (Number.isInteger(segment.generation)) {
            stt.closed[source] = segment.generation!;
            if (stt.previews[source]?.generation <= segment.generation!)
              delete stt.previews[source];
          }
          stt.segments.push(segment);
          stt.result = true;
        }
        break;
      }
      case "transcript_preview": {
        const source = event.source || "mic";
        if (
          Number.isInteger(event.generation) &&
          event.generation > (stt.closed[source] || 0)
        )
          stt.previews[source] = {
            generation: event.generation,
            text: event.text,
          };
        break;
      }
      case "complete":
        finish(token);
        break;
      case "error":
        fail(event.message || t("errProcessingFailed"));
        break;
      case "cancelled":
        finishCapture(id);
        active(null);
        connection.close();
        stream = null;
        reset();
        refreshHistory();
        break;
    }
  };
  connection.onerror = async () => {
    if (token !== epoch || stream !== connection) return;
    connection.close();
    stream = null;
    const job = await status(id, "stt", true);
    if (token !== epoch) return;
    if (!job || job.state === "failed") {
      fail(job?.error || t("errNetwork"));
      return;
    }
    if (job.state === "cancelled") {
      finishCapture(id);
      active(null);
      reset();
      return;
    }
    // Status is the durable snapshot. Replay only events after its cursor.
    applyStatus(job);
    if (job.state === "completed") finish(token);
    else subscribe(id, token);
  };
}
export async function restore(id: string, recovering = false) {
  invalidate();
  const token = epoch;
  stt.selectedId = id;
  stt.error = "";
  active(id);
  // Keep the result panel mounted while the next selection loads.
  stt.segments = [];
  stt.previews = {};
  stt.closed = {};
  stt.result = true;
  stt.progress = true;
  stt.complete = false;
  stt.current = 0;
  stt.total = 0;
  stt.label = "progressProcessing";
  const job = await status(id, "stt", recovering);
  if (token !== epoch) return;
  if (!job || job.state === "failed" || job.state === "cancelled") {
    reset();
    active(null);
    if (job?.state === "failed") stt.error = job.error || t("errNetwork");
    return;
  }
  applyStatus(job);
  if (job.result?.batch_id || job.batch_id)
    localStorage.setItem(selectedKey, id);
  if (job.state !== "completed") {
    active(id);
    subscribe(id, token);
  } else active(null);
}
export function attachLive(
  id: string,
  source: string,
  filename: string,
  startedAt = Date.now() / 1000,
) {
  invalidate();
  reset();
  stt.lastSeq = 0;
  stt.liveSource = source;
  stt.filename = filename;
  stt.startedAt = startedAt;
  stt.result = true;
  active(id);
  subscribe(id);
}
function params(live = false) {
  const result = new URLSearchParams();
  if (!live && settings.autoDetect) result.set("detect_language", "true");
  else {
    result.set("language", settings.language);
    if (settings.model) result.set("model", settings.model);
  }
  if (!live && settings.diarization) result.set("diarization", "true");
  return result;
}
export function liveParams() {
  return params(true);
}
async function start(
  file: AudioInput,
  batch?: { id: string; index: number; total: number },
) {
  const query = params();
  if (batch) {
    query.set("batch_id", batch.id);
    query.set("batch_index", String(batch.index));
    query.set("batch_total", String(batch.total));
  }
  const local = "path" in file;
  const form = new FormData();
  if (!local) form.append("file", file);
  const result = await request<{ job_id: string }>(
    `/jobs/stt${local ? "/local" : ""}?${query}`,
    {
      method: "POST",
      body: local ? JSON.stringify({ path: file.path }) : form,
      headers: local ? { "Content-Type": "application/json" } : undefined,
    },
  );
  if (!result.job_id) throw new Error(t("errNoResponseBody"));
  return result.job_id;
}
export function clearBatch() {
  batchEpoch++;
  clearInterval(pollTimer);
  stt.batchId = null;
  stt.jobs = [];
  localStorage.removeItem(batchKey);
  localStorage.removeItem(selectedKey);
}
export function setBatch(id: string, jobs: Job[]) {
  stt.batchId = id;
  stt.jobs = sorted(jobs);
}
export function pollBatch() {
  clearInterval(pollTimer);
  pollTimer = setInterval(() => void refreshBatch(), 1500);
}
export async function refreshBatch() {
  const id = stt.batchId;
  const token = batchEpoch;
  if (!id) return;
  try {
    const payload = await request<{ jobs: Job[] }>("/jobs?limit=200&offset=0");
    if (token !== batchEpoch || id !== stt.batchId) return;
    const jobs = payload.jobs.filter((job) => job.batch_id === id);
    if (!jobs.length) {
      clearBatch();
      return;
    }
    setBatch(id, jobs);
    if (jobs.some((j) => ["running", "queued"].includes(j.state)))
      localStorage.setItem(batchKey, id);
    else {
      localStorage.removeItem(batchKey);
      clearInterval(pollTimer);
    }
    refreshHistory();
  } catch {
    if (token === batchEpoch) toast(t("errNetwork"));
  }
}
export async function restoreBatch() {
  const id = localStorage.getItem(batchKey);
  if (!id) return;
  stt.batchId = id;
  await refreshBatch();
  if (stt.batchId) {
    pollBatch();
    const selected = localStorage.getItem(selectedKey);
    if (selected && stt.jobs.some((j) => j.job_id === selected) && !stt.jobId)
      await restore(selected, true);
  }
}
export async function upload(files: AudioInput[]) {
  if (!files.length) return;
  cancel();
  clearBatch();
  stt.error = "";
  const token = epoch;
  if (files.length === 1) {
    stt.progress = true;
    stt.busy = true;
    stt.label = "path" in files[0] ? "progressStarting" : "progressUploading";
    stt.filename = files[0].name;
    try {
      const id = await start(files[0]);
      if (token !== epoch) {
        void request(`/jobs/${id}/cancel`, { method: "POST" });
        return;
      }
      active(id);
      stt.result = true;
      stt.lastSeq = 0;
      subscribe(id, token);
      refreshHistory();
    } catch (error) {
      if (token === epoch) fail(error.message || t("errNetwork"));
    }
    return;
  }
  const id = "stt_" + crypto.randomUUID();
  const batchToken = batchEpoch;
  stt.batchId = id;
  localStorage.setItem(batchKey, id);
  stt.jobs = files.map((file, index) => ({
    job_id: null,
    job_type: "stt",
    filename: file.name,
    state: "queued",
    batch_id: id,
    batch_index: index + 1,
    batch_total: files.length,
    progress_current: 0,
    progress_total: 0,
  }));
  await Promise.all(
    files.map(async (file, index) => {
      try {
        const jobId = await start(file, {
          id,
          index: index + 1,
          total: files.length,
        });
        if (batchToken !== batchEpoch) {
          void request(`/jobs/${jobId}/cancel`, { method: "POST" });
          return;
        }
        stt.jobs[index].job_id = jobId;
      } catch (error) {
        if (batchToken === batchEpoch) {
          stt.jobs[index].state = "failed";
          stt.jobs[index].error = error.message;
        }
      }
    }),
  );
  if (batchToken !== batchEpoch) return;
  const first = stt.jobs.find((job) => job.job_id);
  if (first?.job_id) await restore(first.job_id);
  await refreshBatch();
  if (stt.batchId) pollBatch();
}
export async function cancelBatch(onlySelected = false) {
  const targets = stt.jobs.filter(
    (j) =>
      j.job_id &&
      ["queued", "running"].includes(j.state) &&
      (!onlySelected || j.job_id === stt.selectedId),
  );
  await Promise.all(
    targets.map((j) =>
      request(`/jobs/${j.job_id}/cancel`, { method: "POST" }).catch(() => null),
    ),
  );
  if (targets.some((j) => j.job_id === stt.jobId)) {
    invalidate();
    active(null);
    reset();
  }
  await refreshBatch();
}
export async function downloadBatch() {
  const parts: string[] = [];
  for (const job of stt.jobs.filter(
    (j) => j.job_id && j.state === "completed",
  )) {
    const snapshot = await status(job.job_id!, "stt");
    const text = copyText(snapshot?.result?.segments || []);
    if (text) parts.push("# " + filename(job) + "\n\n" + text);
  }
  if (parts.length)
    downloadText(parts.join("\n\n---\n\n"), `${stt.batchId}_${stt.view}.txt`);
}
export function dispose() {
  invalidate();
  clearInterval(pollTimer);
}
