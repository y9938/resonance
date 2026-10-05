import type { LocaleHelpers, TtsLabels } from "./types";
export const messages: Record<string, string> = {
  pageTitle: "Resonance",
  metaDescription: "Resonance — Speech-to-Text and Text-to-Speech",
  ...{
    ariaLangGroup: "Interface language",
    speakerMic: "🎤 Microphone",
    speakerSys: "🔊 System",
    hintSttMedia: "Any files — {limit}",
    hintLimitMb: "up to {mb} MB",
    hintAnySize: "any size",
    progressUploading: "Uploading…",
    progressProcessing: "Processing…",
    progressDiarizing: "Diarizing speakers…",
    progressComplete: "Complete",
    progressChunk: "Chunk {current} / {total}",
    progressStarting: "Starting…",
    errNetwork: "Network error",
    errLiveBackpressure: "Live transcription cannot keep up",
    errPleaseEnterText: "Please enter text",
    ttsPlaceholderLimited:
      "Enter text to synthesize... (up to {limit} characters)",
    ttsPlaceholderUnlimited: "Enter text to synthesize...",
    errTextTooLong: "Text too long",
    errUploadFailed: "Upload failed",
    errNoResponseBody: "No response body",
    errRequestFailed: "Request failed",
    errProcessingFailed: "Processing failed",
    toastCopied: "Copied to clipboard",
    toastCopyFailed: "Copy failed",
    toastReadFailed: "Failed to read file",
    localeSearchEmpty: "No matching languages",
    languageSearchPlaceholder: "Search language…",
    charUnit: "chars",
    defaultTranscriptionFile: "transcription",
    sttMicHintIdle: "Record speech from your microphone",
    sttMicHintLive: "Speak into microphone — real-time transcription active",
    sttMicHintRecording: "Recording… press Stop when finished",
    sttMicHintReady: "Recording is ready to send",
    sttSysHintIdle: "Capture internal system audio (meetings, videos, etc.)",
    sttSysHintCapturing: "Capturing system audio… press Stop when finished",
    sttSysHintProcessing: "Processing system audio…",
    sttSysStart: "Start capture",
    sttSysIncludeMic: "Include microphone",
    errMicUnsupported: "Microphone recording is not supported in this browser",
    errMicPermission: "Microphone access was denied",
    errMicEmpty: "Recorded audio is empty",
    jobsLoading: "Loading jobs…",
    jobsLoadError: "Failed to load jobs list.",
    jobsTypeStt: "STT",
    jobsTypeTts: "TTS",
    jobsStateQueued: "Queued",
    jobsStateRunning: "Running",
    jobsStateCompleted: "Completed",
    jobsStateFailed: "Failed",
    jobsStateCancelled: "Cancelled",
    sttBatchTitle: "Batch queue",
    sttBatchNextReady: "Next ready",
    sttBatchDownloadAll: "Download all",
    sttBatchCancelCurrent: "Cancel current",
    sttBatchCancel: "Cancel active",
    sttBatchSummary: "{done} / {total} complete",
    sttBatchEmpty: "No files in this batch.",
    jobsBatchTitle: "STT batch",
    jobsBatchSummary: "{done} / {total}",
    jobsBatchOpen: "Open",
    sttLanguageLabel: "STT Language",
    sttModelLabel: "STT Model",
    sttModelRecommended: "Recommended",
    sttAutoDetectLabel: "Detect language automatically (Whisper Turbo)",
    sttAutoModelEffective: "Files & dictation: Whisper Turbo",
    sttDiarizationLabelText: "Diarization",
  },
  ...{
    jobsOpenAria: "Open jobs list",
    jobsDrawerTitle: "Jobs",
    jobsCloseAria: "Close jobs list",
    jobsEmpty: "No jobs yet.",
    jobsLoadingMore: "Loading more…",
    jobsListEnd: "All jobs shown.",
    tabStt: "Speech to Text",
    tabTts: "Text to Speech",
    sttDropzoneText: "Drop audio files or click to browse",
    configLoading: "Loading configuration...",
    btnHide: "Hide",
    btnCancel: "Cancel",
    sttLocalTitle: "Local files — without uploading",
    sttLocalPathsLabel: "Absolute paths, one file per line",
    sttLocalHint:
      "Files must stay available on this computer until transcription finishes.",
    sttLocalStart: "Transcribe local files",
    sttBatchTitle: "Batch queue",
    sttBatchNextReady: "Next ready",
    sttBatchDownloadAll: "Download all",
    sttBatchCancelCurrent: "Cancel current",
    sttBatchCancel: "Cancel active",
    sttMicTitle: "Microphone",
    sttSysTitle: "System Audio",
    sttModeDictation: "Dictation",
    sttModeLive: "Live",
    sttSysIncludeMic: "Include microphone",
    sttMicHintIdle: "Record speech from your microphone",
    sttMicStart: "Start recording",
    sttMicStop: "Stop",
    sttMicSend: "Send recording",
    sttMicDiscard: "Discard",
    sttLanguageLabel: "STT Language",
    sttModelLabel: "STT Model",
    sttAutoDetectLabel: "Detect language automatically (Whisper Turbo)",
    sttAutoModelEffective: "Files & dictation: Whisper Turbo",
    sttDiarizationLabelText: "Diarization",
    resultTitleTranscription: "Transcription",
    sttViewBlocks: "Blocks",
    sttViewContinuous: "Continuous",
    btnCopy: "Copy",
    btnDownload: "Download",
    ttsDropzoneText: "Drop text files or click to browse",
    ttsDropzoneHint: "Any text — any size",
    ttsPlaceholder: "Enter text to synthesize...",
    btnSynthesize: "Synthesize",
    resultTitleSynth: "Synthesized Audio",
    btnDownloadWav: "Download WAV",
  },
};
export const helpers: LocaleHelpers = {
  formatCount(value) {
    return Number(value || 0).toLocaleString("en-US");
  },
  formatSttMeta(n) {
    return n + " " + (n === 1 ? "segment" : "segments");
  },
  formatSttProcessedTextDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds <= 0) return "0.0s";
    if (seconds < 60) {
      return seconds.toFixed(1) + "s";
    }
    const total = Math.max(1, Math.round(seconds));
    const hours = Math.floor(total / 3600);
    const mins = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    if (hours > 0) {
      return String(hours) + "h " + String(mins) + "m";
    }
    return String(mins) + "m " + String(secs) + "s";
  },
  formatSttProcessedTextLabel(durationText) {
    return "Processed text: " + durationText;
  },
  formatTtsCharLine(len, maxChars, t) {
    const chunkN = len === 0 ? 0 : Math.ceil(len / maxChars);
    return (
      len +
      " " +
      t("charUnit") +
      " · " +
      chunkN +
      " " +
      (chunkN === 1 ? "chunk" : "chunks")
    );
  },
  formatTtsMeta(chunks, durationSec) {
    return (
      chunks +
      " " +
      (chunks === 1 ? "chunk" : "chunks") +
      " · " +
      Number(durationSec).toFixed(1) +
      "s"
    );
  },
  formatTtsInputTooLongMessage(inputLimit, formatCount) {
    return "Text too long (max " + formatCount(inputLimit) + " characters)";
  },
  formatMicDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds <= 0) return null;
    const total = Math.max(1, Math.round(seconds));
    const mins = Math.floor(total / 60);
    const secs = total % 60;
    if (mins > 0) {
      return String(mins) + ":" + String(secs).padStart(2, "0");
    }
    return String(secs) + "s";
  },
  formatDateTime(date) {
    return date.toLocaleString("en-US");
  },
};
export const tts: TtsLabels = {
  languages: {
    ru: "Russian",
    en: "English",
  },
  voiceGroups: {
    silero_ru: {
      ru_alexandr: "Alexandr (Male)",
      ru_alfia: "Alfia (Female)",
      ru_alfia2: "Alfia 2 (Female)",
      ru_bogdan: "Bogdan (Male)",
      ru_dmitriy: "Dmitriy (Male)",
      ru_ekaterina: "Ekaterina (Female)",
      ru_vika: "Vika (Female)",
      ru_gamat: "Gamat (Male)",
      ru_igor: "Igor (Male)",
      ru_karina: "Karina (Female)",
      ru_kejilgan: "Kejilgan (Male)",
      ru_kermen: "Kermen (Female)",
      ru_marat: "Marat (Male)",
      ru_miyau: "Miyau (Female)",
      ru_nurgul: "Nurgul (Female)",
      ru_oksana: "Oksana (Female)",
      ru_onaoy: "Onaoy (Male)",
      ru_ramilia: "Ramilia (Female)",
      ru_roman: "Roman (Male) ★",
      ru_safarhuja: "Safarhuja (Male)",
      ru_saida: "Saia (Female)",
      ru_sibday: "Sibday (Male)",
      ru_zara: "Zara (Female)",
      ru_zhadyra: "Zhadyra (Female) ★",
      ru_zhazira: "Zhazira (Female)",
      ru_zinaida: "Zinaida (Female)",
      ru_eduard: "Eduard (Male)",
    },
    kokoro_en: {
      af_heart: "Heart (Female)",
      af_alloy: "Alloy (Female)",
      af_aoede: "Aoede (Female)",
      af_bella: "Bella (Female)",
      af_jessica: "Jessica (Female)",
      af_kore: "Kore (Female)",
      af_nicole: "Nicole (Female)",
      af_nova: "Nova (Female)",
      af_river: "River (Female)",
      af_sarah: "Sarah (Female)",
      af_sky: "Sky (Female)",
      am_adam: "Adam (Male)",
      am_echo: "Echo (Male)",
      am_eric: "Eric (Male)",
      am_fenrir: "Fenrir (Male)",
      am_liam: "Liam (Male)",
      am_michael: "Michael (Male)",
      am_onyx: "Onyx (Male)",
      am_puck: "Puck (Male)",
      am_santa: "Santa (Male)",
      bf_alice: "Alice (Female)",
      bf_emma: "Emma (Female)",
      bf_isabella: "Isabella (Female)",
      bf_lily: "Lily (Female)",
      bm_daniel: "Daniel (Male)",
      bm_fable: "Fable (Male)",
      bm_george: "George (Male)",
      bm_lewis: "Lewis (Male)",
    },
  },
};
