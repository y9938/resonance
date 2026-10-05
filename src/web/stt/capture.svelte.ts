import { settings } from "../settings.svelte";
import { t } from "../i18n/state.svelte";
import { request } from "../api/client";
import { stt, attachLive, liveParams, upload } from "./state.svelte";
import { LiveTransport } from "./live-transport";
import { mergeMicPcmChunks, encodeMicWav } from "./pcm";
export const capture = $state({
  recording: false,
  starting: false,
  stopping: false,
  hint: settings.mode === "live" ? "sttMicHintLive" : "sttMicHintIdle",
  elapsed: 0,
  preview: "",
  duration: 0,
  systemId: null as string | null,
});
let stream: MediaStream | null = null,
  context: AudioContext | null = null,
  source: MediaStreamAudioSourceNode | null = null,
  processor: ScriptProcessorNode | null = null,
  gain: GainNode | null = null;
let chunks: Float32Array[] = [],
  count = 0,
  rate = 0,
  file: File | null = null,
  started = 0,
  live: LiveTransport | null = null;
let timer: ReturnType<typeof setInterval> | undefined;
let epoch = 0;
function startTimer(at = Date.now()) {
  clearInterval(timer);
  started = at;
  const tick = () =>
    (capture.elapsed = Math.floor((Date.now() - started) / 1000));
  tick();
  timer = setInterval(tick, 1000);
}
function stopTimer() {
  clearInterval(timer);
}
function stopTracks() {
  source?.disconnect();
  processor?.disconnect();
  gain?.disconnect();
  stream?.getTracks().forEach((t) => t.stop());
  void context?.close().catch(() => {});
  stream = null;
  context = null;
  source = null;
  processor = null;
  gain = null;
}
export function setMode(mode: "dictation" | "live") {
  if (capture.recording || capture.starting || capture.systemId) return;
  settings.mode = mode;
  localStorage.setItem("resonance_sttMicMode", mode);
  capture.hint = mode === "live" ? "sttMicHintLive" : "sttMicHintIdle";
}
export function setSource(value: "mic" | "sys") {
  if (capture.recording || capture.starting || capture.systemId) return;
  settings.source = value;
  capture.hint =
    value === "sys"
      ? "sttSysHintIdle"
      : settings.mode === "live"
        ? "sttMicHintLive"
        : "sttMicHintIdle";
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
function failure(error: Error) {
  stopTracks();
  stopTimer();
  capture.recording = false;
  capture.hint = "sttMicHintLive";
  stt.error = error.message || t("errNetwork");
  live = null;
}
export function restoreSystem() {
  if (stt.liveSource !== "system_audio" || !stt.jobId) return;
  settings.source = "sys";
  capture.systemId = stt.jobId;
  capture.recording = true;
  capture.hint = "sttSysHintCapturing";
  startTimer(stt.startedAt ? stt.startedAt * 1000 : Date.now());
}
export async function start() {
  if (capture.starting || capture.recording) return;
  const token = ++epoch;
  capture.starting = true;
  stt.error = "";
  if (settings.source === "sys") {
    try {
      const params = liveParams();
      if (settings.includeMic) params.set("include_microphone", "true");
      const result = await request<{ job_id: string; started_at?: number }>(
        "/system-audio/start?" + params,
        { method: "POST" },
      );
      if (token !== epoch) {
        void request(
          "/system-audio/stop?job_id=" + encodeURIComponent(result.job_id),
          { method: "POST" },
        );
        return;
      }
      capture.systemId = result.job_id;
      capture.recording = true;
      capture.hint = "sttSysHintCapturing";
      startTimer((result.started_at || Date.now() / 1000) * 1000);
      attachLive(
        result.job_id,
        "system_audio",
        "System Audio Capture.wav",
        result.started_at,
      );
    } catch (error) {
      stt.error = error.message;
      capture.hint = "sttSysHintIdle";
    } finally {
      capture.starting = false;
    }
    return;
  }
  const AudioCtor =
    window.AudioContext ||
    (window as Window & { webkitAudioContext?: typeof AudioContext })
      .webkitAudioContext;
  if (!navigator.mediaDevices?.getUserMedia || !AudioCtor) {
    stt.error = t("errMicUnsupported");
    capture.starting = false;
    return;
  }
  discard();
  const isLive = settings.mode === "live";
  let jobId: string | null = null;
  try {
    if (isLive) {
      const result = await request<{ job_id: string }>(
        "/jobs/live/start?" + liveParams(),
        { method: "POST" },
      );
      jobId = result.job_id;
    }
    const media = await navigator.mediaDevices.getUserMedia({ audio: true });
    if (token !== epoch) {
      media.getTracks().forEach((t) => t.stop());
      if (jobId) void request(`/jobs/live/${jobId}/stop`, { method: "POST" });
      return;
    }
    stream = media;
    context = new AudioCtor();
    await context.resume();
    rate = context.sampleRate;
    source = context.createMediaStreamSource(stream);
    processor = context.createScriptProcessor(4096, 1, 1);
    gain = context.createGain();
    gain.gain.value = 0;
    if (jobId) {
      attachLive(jobId, "mic_live", "Live Recording.wav");
      live = new LiveTransport(
        jobId,
        rate,
        () => failure(new Error(t("errNetwork"))),
        () => {
          stopTracks();
          stopTimer();
          capture.recording = false;
          capture.hint = "sttMicHintLive";
          stt.error = t("errLiveBackpressure");
        },
      );
    }
    processor.onaudioprocess = (event) => {
      const chunk = new Float32Array(event.inputBuffer.getChannelData(0));
      if (isLive) live?.append(chunk);
      else {
        chunks.push(chunk);
        count += chunk.length;
      }
    };
    source.connect(processor);
    processor.connect(gain);
    gain.connect(context.destination);
    capture.recording = true;
    startTimer();
    capture.hint = isLive ? "sttMicHintLive" : "sttMicHintRecording";
  } catch {
    stopTracks();
    stopTimer();
    capture.recording = false;
    stt.error = t("errMicPermission");
    if (jobId)
      void request(`/jobs/live/${jobId}/stop`, { method: "POST" }).catch(
        () => {},
      );
  } finally {
    capture.starting = false;
  }
}
export async function stop() {
  if (capture.stopping) return;
  capture.stopping = true;
  stopTimer();
  try {
    if (capture.systemId) {
      const id = capture.systemId;
      capture.hint = "sttSysHintProcessing";
      await request("/system-audio/stop?job_id=" + encodeURIComponent(id), {
        method: "POST",
      });
      capture.systemId = null;
      capture.recording = false;
      capture.hint = "sttSysHintIdle";
      return;
    }
    stopTracks();
    capture.recording = false;
    if (live) {
      const transport = live;
      await transport.flush();
      if (!transport.failed && !transport.backpressure) {
        await request(`/jobs/live/${transport.jobId}/stop`, { method: "POST" });
        live = null;
      }
      capture.hint = "sttMicHintLive";
      return;
    }
    if (!count || !rate) {
      stt.error = t("errMicEmpty");
      return;
    }
    const blob = encodeMicWav(mergeMicPcmChunks(chunks, count), rate);
    file = new File(
      [blob],
      "mic_" + new Date().toISOString().replace(/[:.]/g, "-") + ".wav",
      { type: "audio/wav" },
    );
    capture.preview = URL.createObjectURL(blob);
    capture.duration = count / rate;
    capture.hint = "sttMicHintReady";
  } catch (error) {
    stt.error = error.message || t("errNetwork");
  } finally {
    capture.stopping = false;
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
  stopTimer();
  live?.dispose();
  live = null;
  if (capture.preview) URL.revokeObjectURL(capture.preview);
}
