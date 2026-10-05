import { request, status } from "../api/client";
import type { Job, JobEvent, Voice } from "../api/types";
import { settings, toast } from "../settings.svelte";
import { t, helpers, ttsLabels } from "../i18n/state.svelte";
import { refreshHistory } from "../jobs/state.svelte";
const key = "resonance_tts_active_job_id";
export const tts = $state({
  text: localStorage.getItem("resonance_ttsText") || "",
  language: "",
  voice: "",
  sourceFilename: "",
  jobId: null as string | null,
  current: 0,
  total: 0,
  label: "",
  progress: false,
  busy: false,
  error: "",
  result: false,
  url: "",
  filename: "",
  chunks: 0,
  duration: 0,
  complete: false,
});
let epoch = 0;
let stream: EventSource | null = null;
let timer: ReturnType<typeof setTimeout>;
export function languages() {
  return settings.config.tts?.languages || [];
}
export function voices() {
  return languages().find((l) => l.id === tts.language)?.voices || [];
}
export function voiceLabel(voice: Voice) {
  const labels = ttsLabels();
  return (
    labels.voiceGroups[voice.backend_id || ""]?.[voice.id] ||
    fallbackVoiceLabel(voice.id)
  );
}
function fallbackVoiceLabel(id: string) {
  if (id.startsWith("ru_")) return id;
  const parts = id.split("_");
  if (parts.length !== 2) return id;
  const regions: Record<string, string> = {
    af: "US",
    am: "US",
    bf: "UK",
    bm: "UK",
  };
  return (
    parts[1].charAt(0).toUpperCase() +
    parts[1].slice(1) +
    " (" +
    (regions[parts[0]] || parts[0].toUpperCase()) +
    ")"
  );
}
export function setVoice(value: string) {
  tts.voice = value;
  let saved: Record<string, string> = {};
  try {
    saved = JSON.parse(
      localStorage.getItem("resonance_ttsVoiceByLanguage") || "{}",
    );
  } catch {
    /* Ignore corrupt preferences. */
  }
  saved[tts.language] = value;
  localStorage.setItem("resonance_ttsVoiceByLanguage", JSON.stringify(saved));
}
export function setLanguage(value: string) {
  tts.language = value;
  localStorage.setItem("resonance_ttsLanguage", value);
  let saved: Record<string, string> = {};
  try {
    saved = JSON.parse(
      localStorage.getItem("resonance_ttsVoiceByLanguage") || "{}",
    );
  } catch {
    /* Use default voice. */
  }
  const language = languages().find((l) => l.id === value);
  setVoice(
    [saved[value], language?.default_voice_id, voices()[0]?.id].find((id) =>
      voices().some((v) => v.id === id),
    ) || "",
  );
}
export function initialize() {
  setLanguage(
    [
      localStorage.getItem("resonance_ttsLanguage"),
      settings.config.tts?.default_language,
      languages()[0]?.id,
    ].find((id) => languages().some((l) => l.id === id)) || "",
  );
}
export function saveDraft() {
  try {
    if (!tts.text || tts.text.length > 100000)
      localStorage.removeItem("resonance_ttsText");
    else localStorage.setItem("resonance_ttsText", tts.text);
  } catch {
    localStorage.removeItem("resonance_ttsText");
  }
}
export function inputLimit() {
  return Math.max(0, Math.floor(settings.config.tts_max_input_chars || 0));
}
export function placeholder() {
  return inputLimit()
    ? t("ttsPlaceholderLimited", { limit: helpers().formatCount(inputLimit()) })
    : t("ttsPlaceholderUnlimited");
}
export function progressText() {
  return tts.label
    ? t(tts.label)
    : t("progressChunk", { current: tts.current, total: tts.total || "~" });
}
export function percent() {
  return tts.complete
    ? 100
    : tts.total > 0
      ? Math.max(0, Math.min(100, (tts.current / tts.total) * 100))
      : tts.label === "progressStarting"
        ? 10
        : 0;
}
export function invalidate() {
  epoch++;
  stream?.close();
  stream = null;
  clearTimeout(timer);
}
function active(id: string | null) {
  tts.jobId = id;
  if (id) localStorage.setItem(key, id);
  else localStorage.removeItem(key);
}
function reset(keep = false) {
  tts.busy = false;
  tts.progress = false;
  tts.complete = false;
  tts.current = 0;
  tts.total = 0;
  tts.label = "";
  if (!keep) {
    tts.result = false;
    tts.url = "";
  }
}
export function cancel() {
  const id = tts.jobId;
  invalidate();
  active(null);
  reset();
  if (id)
    void request(`/jobs/${id}/cancel`, { method: "POST" }).catch(() => {});
  refreshHistory();
}
function fail(message: string) {
  tts.error = message;
  active(null);
  stream?.close();
  stream = null;
  reset();
  refreshHistory();
}
function apply(job: Job) {
  tts.current = job.progress_current || 0;
  tts.total = job.progress_total || 0;
  tts.progress = true;
  tts.busy = job.state !== "completed";
  tts.complete = job.state === "completed";
  tts.label = tts.complete ? "progressComplete" : "";
  if (tts.complete) {
    tts.result = true;
    tts.url = job.result?.download_url || "";
    tts.filename = job.result?.filename || "tts_output.wav";
    tts.chunks = job.result?.chunks || 0;
    tts.duration = job.result?.duration || 0;
  }
}
function finish(token: number) {
  tts.busy = false;
  tts.complete = true;
  tts.label = "progressComplete";
  active(null);
  stream?.close();
  stream = null;
  refreshHistory();
  timer = setTimeout(() => {
    if (token === epoch) reset(true);
  }, 800);
}
function subscribe(id: string, token: number, after = 0) {
  stream?.close();
  const connection = new EventSource(`/api/jobs/${id}/events?after=${after}`);
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
      if (event.seq <= after) return;
      after = event.seq;
    }
    switch (event.type) {
      case "start":
        tts.current = 0;
        tts.total = event.total || 0;
        tts.label = "progressProcessing";
        break;
      case "progress":
        tts.current = event.current;
        tts.total = event.total;
        tts.label = "";
        break;
      case "complete":
        tts.result = true;
        tts.url = event.download_url || "";
        tts.chunks = event.chunks || 0;
        tts.duration = event.duration;
        finish(token);
        break;
      case "error":
        fail(event.message || t("errProcessingFailed"));
        break;
      case "cancelled":
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
    const job = await status(id, "tts");
    if (token !== epoch) return;
    if (!job || job.state === "failed") {
      fail(job?.error || t("errNetwork"));
      return;
    }
    if (job.state === "cancelled") {
      active(null);
      reset();
      return;
    }
    apply(job);
    if (job.state === "completed") finish(token);
    else subscribe(id, token, job.last_event_seq || 0);
  };
}
export async function restore(id: string) {
  invalidate();
  const token = epoch;
  tts.error = "";
  const job = await status(id, "tts");
  if (token !== epoch) return;
  if (!job || job.state === "failed" || job.state === "cancelled") {
    active(null);
    reset();
    if (job?.state === "failed") tts.error = job.error || t("errNetwork");
    return;
  }
  apply(job);
  if (job.state === "completed") active(null);
  else {
    active(id);
    subscribe(id, token, job.last_event_seq || 0);
  }
}
export async function readFile(file: File) {
  try {
    tts.text = await file.text();
    tts.sourceFilename = file.name.replace(/\.[^/.]+$/, "");
    saveDraft();
  } catch {
    toast(t("toastReadFailed"));
  }
}
export async function synthesize() {
  const text = tts.text.trim();
  if (!text) {
    tts.error = t("errPleaseEnterText");
    return;
  }
  if (inputLimit() && text.length > inputLimit()) {
    tts.error = helpers().formatTtsInputTooLongMessage(
      inputLimit(),
      helpers().formatCount,
      t,
    );
    return;
  }
  cancel();
  const token = epoch;
  tts.error = "";
  tts.busy = true;
  tts.progress = true;
  tts.label = "progressStarting";
  const filename =
    tts.sourceFilename ||
    "tts_" +
      new Date()
        .toISOString()
        .replace(/[:.]/g, "-")
        .replace("T", "_")
        .slice(0, -5);
  tts.sourceFilename = "";
  tts.filename = filename + ".wav";
  try {
    const result = await request<{ job_id: string }>(
      "/jobs/tts?" +
        new URLSearchParams({
          text,
          language: tts.language,
          voice_id: tts.voice,
          filename,
        }),
      { method: "POST" },
    );
    if (token !== epoch) {
      void request(`/jobs/${result.job_id}/cancel`, { method: "POST" });
      return;
    }
    active(result.job_id);
    subscribe(result.job_id, token);
    refreshHistory();
  } catch (error) {
    if (token === epoch) fail(error.message || t("errNetwork"));
  }
}
export function dispose() {
  invalidate();
}
