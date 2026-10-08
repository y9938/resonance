import type { CaptureConfig } from "../api/types";
import { settings } from "../settings.svelte";
import { t } from "../i18n/state.svelte";
import { request, status } from "../api/client";
import { stt, attachLive, liveParams, upload } from "./state.svelte";
import { LiveTransport } from "./live-transport";
import { mergeMicPcmChunks, encodeMicWav } from "./pcm";

type CaptureSession = Readonly<
  | { kind: "system"; jobId: string; config: Readonly<CaptureConfig> | null }
  | { kind: "live"; jobId: string; transport: LiveTransport }
  | { kind: "dictation" }
>;
type CaptureOperation =
  | { state: "idle" | "starting" | "failed" }
  | { state: "running" | "stopping" | "unknown"; session: CaptureSession };
// An active session owns its configuration. Unknown means Stop is unconfirmed:
// keep that session and allow another Stop, never infer capture from STT progress.
let operation = $state.raw<CaptureOperation>({ state: "idle" });
function activeSession() {
  return "session" in operation ? operation.session : null;
}
export const capture = $state({
  get state() { return operation.state; },
  get active() { return activeSession() !== null; },
  get config() {
    const session = activeSession();
    return session?.kind === "system" ? session.config : null;
  },
  hint: settings.mode === "live" ? "sttMicHintLive" : "sttMicHintIdle",
  elapsed: 0,
  preview: "",
  duration: 0,
});
let stream: MediaStream | null = null,
  context: AudioContext | null = null,
  source: MediaStreamAudioSourceNode | null = null,
  processor: ScriptProcessorNode | null = null,
  gain: GainNode | null = null;
let chunks: Float32Array[] = [], count = 0, rate = 0,
  file: File | null = null, started = 0;
let timer: ReturnType<typeof setInterval> | undefined;
let epoch = 0;
function startTimer(at = Date.now()) {
  clearInterval(timer);
  started = at;
  const tick = () => (capture.elapsed = Math.floor((Date.now() - started) / 1000));
  tick();
  timer = setInterval(tick, 1000);
}
function stopTracks() {
  source?.disconnect();
  processor?.disconnect();
  gain?.disconnect();
  stream?.getTracks().forEach((track) => track.stop());
  void context?.close().catch(() => {});
  stream = null;
  context = null;
  source = null;
  processor = null;
  gain = null;
}
export function canConfigureCapture() {
  return operation.state === "idle" || operation.state === "failed";
}
export function setIncludeMicrophone(value: boolean) {
  if (!canConfigureCapture()) return;
  settings.includeMic = value;
  localStorage.setItem("resonance_sttSysIncludeMic", String(value));
}
export function setMode(mode: "dictation" | "live") {
  if (!canConfigureCapture()) return;
  settings.mode = mode;
  localStorage.setItem("resonance_sttMicMode", mode);
  capture.hint = mode === "live" ? "sttMicHintLive" : "sttMicHintIdle";
}
export function setSource(value: "mic" | "sys") {
  if (!canConfigureCapture()) return;
  settings.source = value;
  capture.hint = value === "sys" ? "sttSysHintIdle"
    : settings.mode === "live" ? "sttMicHintLive" : "sttMicHintIdle";
}
export function discard() {
  if (capture.preview) URL.revokeObjectURL(capture.preview);
  capture.preview = "";
  capture.duration = 0;
  capture.elapsed = 0;
  file = null;
  chunks = [];
  count = 0;
  capture.hint = settings.mode === "live" ? "sttMicHintLive" : "sttMicHintIdle";
}
function finishSession(session: CaptureSession, state: "idle" | "failed" = "idle") {
  if (activeSession() !== session) return;
  epoch++;
  stopTracks();
  clearInterval(timer);
  if (session.kind === "live") session.transport.dispose();
  operation = { state };
  capture.hint = session.kind === "system" ? "sttSysHintIdle"
    : settings.mode === "live" ? "sttMicHintLive" : "sttMicHintIdle";
}
function failure(session: CaptureSession, message: string) {
  if (activeSession() !== session) return;
  stt.error = message;
  if (session.kind === "live") void stop();
  else finishSession(session, "failed");
}
export function finishCapture(id: string | null) {
  const session = activeSession();
  if (session && session.kind !== "dictation" && session.jobId === id)
    finishSession(session);
}
export function restoreSystem() {
  if (!canConfigureCapture() || stt.liveSource !== "system_audio" || !stt.jobId) return;
  epoch++;
  settings.source = "sys";
  operation = { state: "running", session: {
    kind: "system", jobId: stt.jobId, config: stt.captureConfig,
  }};
  capture.hint = "sttSysHintCapturing";
  startTimer(stt.startedAt ? stt.startedAt * 1000 : Date.now());
}
function reportMicrophone(
  event: "mic-requested" | "mic-started" | "mic-failed",
  stage: "unsupported" | "model" | "permission" | "audio",
  error?: unknown,
) {
  if (!settings.config.local_files_enabled) return;
  const names = ["NotAllowedError", "NotFoundError", "NotReadableError", "AbortError",
    "SecurityError", "InvalidStateError", "TypeError", "UnknownError"];
  const name = error instanceof Error ? error.name : "UnknownError";
  // Best effort: a diagnostic must never delay recording or replace its error.
  void fetch("/api/diagnostics/capture", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ event, stage, mode: settings.mode,
      error: event === "mic-failed" ? (names.includes(name) ? name : "UnknownError") : null }),
    signal: AbortSignal.timeout?.(3000),
  }).catch(() => {});
}
export async function start() {
  if (!canConfigureCapture()) return;
  const token = ++epoch;
  operation = { state: "starting" };
  stt.error = "";
  if (settings.source === "sys") {
    try {
      const params = liveParams();
      if (settings.includeMic) params.set("include_microphone", "true");
      const result = await request<{
        job_id: string; started_at?: number; capture_config?: CaptureConfig;
      }>("/system-audio/start?" + params, { method: "POST" });
      if (token !== epoch) {
        void request("/system-audio/stop?job_id=" + encodeURIComponent(result.job_id),
          { method: "POST" }).catch(() => {});
        return;
      }
      const session: CaptureSession = {
        kind: "system", jobId: result.job_id, config: result.capture_config || null,
      };
      operation = { state: "running", session };
      capture.hint = "sttSysHintCapturing";
      startTimer((result.started_at || Date.now() / 1000) * 1000);
      attachLive(result.job_id, "system_audio", "System Audio Capture.wav",
        result.started_at, session.config);
    } catch (error) {
      if (token !== epoch) return;
      operation = { state: "failed" };
      stt.error = error.message;
      capture.hint = "sttSysHintIdle";
    }
    return;
  }
  const AudioCtor = window.AudioContext ||
    (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!navigator.mediaDevices?.getUserMedia || !AudioCtor) {
    reportMicrophone("mic-failed", "unsupported");
    stt.error = t("errMicUnsupported");
    operation = { state: "failed" };
    return;
  }
  discard();
  const isLive = settings.mode === "live";
  let jobId: string | null = null;
  let stage: "model" | "permission" | "audio" = "model";
  try {
    if (isLive) {
      const result = await request<{ job_id: string }>(
        "/jobs/live/start?" + liveParams(), { method: "POST" });
      jobId = result.job_id;
    }
    stage = "permission";
    reportMicrophone("mic-requested", stage);
    const media = await navigator.mediaDevices.getUserMedia({ audio: true });
    if (token !== epoch) {
      media.getTracks().forEach((track) => track.stop());
      if (jobId) void request(`/jobs/live/${jobId}/stop`, { method: "POST" }).catch(() => {});
      return;
    }
    stage = "audio";
    stream = media;
    context = new AudioCtor();
    await context.resume();
    if (token !== epoch) return;
    rate = context.sampleRate;
    source = context.createMediaStreamSource(stream);
    processor = context.createScriptProcessor(4096, 1, 1);
    gain = context.createGain();
    gain.gain.value = 0;
    let session: CaptureSession = { kind: "dictation" };
    if (jobId) {
      const transport = new LiveTransport(jobId, rate,
        () => failure(session, t("errLiveUndelivered")),
        () => failure(session, t("errLiveBackpressure")));
      session = { kind: "live", jobId, transport };
      attachLive(jobId, "mic_live", "Live Recording.wav");
    }
    processor.onaudioprocess = (event) => {
      if (activeSession() !== session) return;
      const chunk = new Float32Array(event.inputBuffer.getChannelData(0));
      if (session.kind === "live") session.transport.append(chunk);
      else { chunks.push(chunk); count += chunk.length; }
    };
    source.connect(processor);
    processor.connect(gain);
    gain.connect(context.destination);
    operation = { state: "running", session };
    reportMicrophone("mic-started", stage);
    startTimer();
    capture.hint = isLive ? "sttMicHintLive" : "sttMicHintRecording";
  } catch (error) {
    if (token !== epoch) return;
    stopTracks();
    clearInterval(timer);
    operation = { state: "failed" };
    reportMicrophone("mic-failed", stage, error);
    stt.error = error.name === "NotAllowedError" ? t("errMicPermission")
      : error.message || t("errNetwork");
    if (jobId) void request(`/jobs/live/${jobId}/stop`, { method: "POST" }).catch(() => {});
  }
}
export async function stop() {
  if (operation.state !== "running" && operation.state !== "unknown") return;
  const session = operation.session;
  operation = { state: "stopping", session };
  clearInterval(timer);
  try {
    if (session.kind === "system") {
      capture.hint = "sttSysHintProcessing";
      await request("/system-audio/stop?job_id=" + encodeURIComponent(session.jobId),
        { method: "POST", signal: AbortSignal.timeout(30_000) });
    } else {
      stopTracks();
      if (session.kind === "live") {
        capture.hint = "sttMicHintSending";
        await session.transport.drain();
        if (activeSession() !== session) return;
        if (session.transport.failed) {
          await request(`/jobs/${session.jobId}/cancel`,
            { method: "POST", signal: AbortSignal.timeout(10_000) });
          finishSession(session, "failed");
          return;
        }
        await request(`/jobs/live/${session.jobId}/stop`,
          { method: "POST", signal: AbortSignal.timeout(30_000) });
      } else {
        if (!count || !rate) {
          failure(session, t("errMicEmpty"));
          return;
        }
        const blob = encodeMicWav(mergeMicPcmChunks(chunks, count), rate);
        file = new File([blob], "mic_" + new Date().toISOString().replace(/[:.]/g, "-") + ".wav",
          { type: "audio/wav" });
        capture.preview = URL.createObjectURL(blob);
        capture.duration = count / rate;
      }
    }
    finishSession(session);
    if (session.kind === "dictation") capture.hint = "sttMicHintReady";
  } catch (error) {
    if (activeSession() !== session) return;
    stt.error = session.kind === "live" && session.transport.failed
      ? t("errLiveUndelivered") : session.kind !== "dictation"
        ? t("errCaptureStopUnconfirmed") : error.message || t("errNetwork");
    operation = { state: "unknown", session };
    if (session.kind !== "dictation") {
      const job = await status(session.jobId, "stt", true);
      if (job && ["completed", "failed", "cancelled"].includes(job.state))
        finishSession(session, job.state === "failed" ? "failed" : "idle");
    }
  }
}
export async function send() {
  if (!file) return;
  const audio = file;
  discard();
  await upload([audio]);
}
export function dispose() {
  epoch++;
  stopTracks();
  clearInterval(timer);
  const session = activeSession();
  if (session?.kind === "live") session.transport.dispose();
  operation = { state: "idle" };
  if (capture.preview) URL.revokeObjectURL(capture.preview);
}
