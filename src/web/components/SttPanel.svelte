<script lang="ts">
  import { onDestroy } from "svelte";
  import {
    stt,
    upload,
    cancel,
    hide,
    progressText,
    percent,
    displayText,
    copyText,
    setView,
  } from "../stt/state.svelte";
  import {
    settings,
    compatible,
    languageOptions,
    setLanguage,
    setModel,
    toast,
  } from "../settings.svelte";
  import { t, helpers } from "../i18n/state.svelte";
  import { buildSttTimeRangesText } from "../stt/transcript";
  import { downloadText } from "../api/client";
  import LanguagePicker from "./LanguagePicker.svelte";
  import BatchPanel from "./BatchPanel.svelte";
  import CapturePanel from "./CapturePanel.svelte";
  let fileInput: HTMLInputElement;
  let dragging = $state(false);
  let localPaths = $state("");
  let localStarting = $state(false);
  async function localStart() {
    if (localStarting) return;
    const files = localPaths
      .split(/\r?\n/)
      .map((p) => p.trim())
      .map((p) => (p.startsWith('"') && p.endsWith('"') ? p.slice(1, -1) : p))
      .filter(Boolean)
      .map((path) => ({ path, name: path.split(/[\\/]/).pop()! }));
    localPaths = "";
    localStarting = true;
    try {
      await upload(files);
    } finally {
      localStarting = false;
    }
  }
  async function copy() {
    const text = copyText();
    try {
      await navigator.clipboard.writeText(text);
      toast(t("toastCopied"));
    } catch {
      const node = document.createElement("textarea");
      node.value = text;
      node.style.position = "fixed";
      node.style.opacity = "0";
      document.body.append(node);
      node.select();
      try {
        toast(
          t(document.execCommand("copy") ? "toastCopied" : "toastCopyFailed"),
        );
      } finally {
        node.remove();
      }
    }
  }
  let height = $state(0),
    resizing = $state(false);
  let startY = 0,
    startHeight = 0,
    lastY = 0,
    frame = 0;
  function applyHeight() {
    height = Math.max(220, startHeight + window.scrollY + lastY - startY);
  }
  function resizeTick() {
    if (!resizing) return;
    const below = lastY - (window.innerHeight - 56),
      above = 56 - lastY;
    const delta =
      below > 0
        ? Math.min(24, Math.max(6, below * 0.35))
        : above > 0
          ? -Math.min(24, Math.max(6, above * 0.35))
          : 0;
    if (delta) {
      window.scrollBy(0, delta);
      applyHeight();
    }
    frame = requestAnimationFrame(resizeTick);
  }
  function resizeStart(e: PointerEvent) {
    if (e.button !== 0) return;
    e.preventDefault();
    const node = document.getElementById("sttResultText")!;
    startHeight = node.getBoundingClientRect().height;
    startY = window.scrollY + e.clientY;
    lastY = e.clientY;
    resizing = true;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    document.body.classList.add("stt-resizing");
    frame = requestAnimationFrame(resizeTick);
  }
  function resizeMove(e: PointerEvent) {
    if (resizing) {
      lastY = e.clientY;
      applyHeight();
    }
  }
  function resizeStop() {
    resizing = false;
    cancelAnimationFrame(frame);
    document.body.classList.remove("stt-resizing");
  }
  onDestroy(resizeStop);
</script>

<section
  class="panel"
  id="stt-panel"
  role="tabpanel"
  class:active={settings.tab === "stt"}
>
  <div
    class="dropzone"
    id="sttDropzone"
    class:processing={stt.busy}
    class:dragover={dragging}
    onclick={() => fileInput.click()}
    ondragover={(e) => {
      e.preventDefault();
      dragging = true;
    }}
    ondragleave={() => (dragging = false)}
    ondrop={(e) => {
      e.preventDefault();
      dragging = false;
      void upload(Array.from(e.dataTransfer?.files || []));
    }}
    role="button"
    tabindex="0"
    onkeydown={(e) => {
      if (e.key === "Enter" || e.key === " ") fileInput.click();
    }}
  >
    <svg
      class="dropzone-icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="1.5"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" x2="12" y1="3" y2="15" />
    </svg>
    <div class="dropzone-text">{t("sttDropzoneText")}</div>
    <div class="dropzone-hint" id="sttDropzoneHint">
      {settings.loaded
        ? t("hintSttMedia", {
            limit: settings.config.upload_limit_mb
              ? t("hintLimitMb", { mb: settings.config.upload_limit_mb })
              : t("hintAnySize"),
          })
        : t("configLoading")}
    </div>

    <div class="progress" id="sttProgress" class:active={stt.progress}>
      <div class="progress-bar">
        <div
          class="progress-fill"
          id="sttProgressFill"
          style:width={percent() + "%"}
        ></div>
      </div>
      <div class="progress-text" id="sttProgressText">{progressText()}</div>
    </div>

    <div
      style="display: flex; gap: 0.5rem; justify-content: center; margin-top: 0.75rem;"
    >
      <button
        class="cancel-btn"
        id="sttHide"
        class:active={stt.progress && !stt.liveSource}
        onclick={(e) => {
          e.stopPropagation();
          hide();
        }}
      >
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
        </svg>
        <span>{t("btnHide")}</span>
      </button>
      <button
        class="cancel-btn"
        id="sttCancel"
        class:active={stt.progress && !stt.liveSource}
        onclick={(e) => {
          e.stopPropagation();
          cancel();
        }}
      >
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
        <span>{t("btnCancel")}</span>
      </button>
    </div>
  </div>
  <input
    type="file"
    id="sttFileInput"
    accept="audio/*,video/*"
    multiple
    bind:this={fileInput}
    onchange={(e) => {
      void upload(Array.from(e.currentTarget.files || []));
      e.currentTarget.value = "";
    }}
  />

  <details id="sttLocalFiles" hidden={!settings.config.local_files_enabled}>
    <summary>{t("sttLocalTitle")}</summary>
    <label for="sttLocalPaths">{t("sttLocalPathsLabel")}</label>
    <textarea
      class="text-input"
      id="sttLocalPaths"
      rows="3"
      spellcheck="false"
      aria-describedby="sttLocalHint"
      bind:value={localPaths}></textarea>
    <p class="dropzone-hint" id="sttLocalHint">{t("sttLocalHint")}</p>
    <button
      class="btn btn-primary"
      id="sttLocalStart"
      disabled={localStarting}
      type="button"
      onclick={localStart}>{t("sttLocalStart")}</button
    >
  </details>

  <BatchPanel />
  <CapturePanel />
  <div class="input-footer">
    <div
      class="stt-controls-group"
      style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;"
    >
      <span id="sttLanguageLabel">{t("sttLanguageLabel")}</span>
      <LanguagePicker
        id="sttLanguage"
        options={languageOptions()}
        value={settings.language}
        onchange={setLanguage}
        disabled={!settings.loaded}
        labelId="sttLanguageLabel"
      />

      <div
        id="sttModelContainer"
        style="display: flex; align-items: center; gap: 0.5rem; transition: opacity 0.2s;"
        style:display={compatible().length > 1 &&
        !(
          settings.autoDetect &&
          settings.source === "mic" &&
          settings.mode === "dictation"
        )
          ? "flex"
          : "none"}
      >
        <span>{t("sttModelLabel")}</span>
        <select
          class="speaker-select"
          id="sttModel"
          disabled={!settings.loaded}
          value={settings.model}
          onchange={(e) => setModel(e.currentTarget.value)}
          ><option value="">{t("sttModelRecommended")}</option
          >{#each compatible() as model (model.id)}<option value={model.id}
              >{model.name}</option
            >{/each}</select
        >
      </div>

      <label
        for="sttAutoDetect"
        style="display: flex; align-items: center; gap: 0.5rem; color: var(--text-secondary); font-size: 0.875rem; cursor: pointer;"
      >
        <input
          type="checkbox"
          id="sttAutoDetect"
          style="width: 16px; height: 16px; accent-color: var(--accent);"
          bind:checked={settings.autoDetect}
        />
        <span>{t("sttAutoDetectLabel")}</span>
      </label>
      <span
        id="sttAutoModelEffective"
        style:display={settings.autoDetect ? "inline" : "none"}
        >{t("sttAutoModelEffective")}</span
      >

      <div
        id="sttDiarizationContainer"
        style="display: flex; align-items: center; gap: 0.35rem; transition: opacity 0.2s;"
        style:display={settings.source === "sys" || settings.mode === "live"
          ? "none"
          : "flex"}
      >
        <input
          type="checkbox"
          id="sttDiarization"
          style="cursor: pointer; width: 16px; height: 16px; accent-color: var(--accent);"
          bind:checked={settings.diarization}
          onchange={() =>
            localStorage.setItem(
              "resonance_sttDiarization",
              String(settings.diarization),
            )}
        />
        <label
          for="sttDiarization"
          style="cursor: pointer; font-size: 0.875rem; color: var(--text-primary); user-select: none;"
          >{t("sttDiarizationLabelText")}</label
        >
      </div>
    </div>
  </div>

  <div class="error" id="sttError" class:active={!!stt.error}>{stt.error}</div>

  <div class="result" id="sttResult" class:active={stt.result}>
    <div class="result-header">
      <span class="result-title">{t("resultTitleTranscription")}</span>
      <div class="result-meta-stack">
        <span class="result-meta" id="sttMeta"
          >{helpers().formatSttMeta(stt.segments.length)}</span
        >
        <span class="result-meta-sub" id="sttTimeRanges"
          >{buildSttTimeRangesText(stt.segments)}</span
        >
      </div>
    </div>
    <div class="stt-view-toggle" role="group" aria-label="STT transcript view">
      <button
        class="stt-view-btn active"
        id="sttViewBlocks"
        type="button"
        class:active={stt.view === "blocks"}
        aria-pressed={stt.view === "blocks"}
        onclick={() => setView("blocks")}>{t("sttViewBlocks")}</button
      >
      <button
        class="stt-view-btn"
        id="sttViewContinuous"
        type="button"
        class:active={stt.view === "continuous"}
        aria-pressed={stt.view === "continuous"}
        onclick={() => setView("continuous")}>{t("sttViewContinuous")}</button
      >
    </div>
    <div class="result-actions">
      <button class="btn btn-secondary" id="sttCopy" onclick={copy}>
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </svg>
        <span>{t("btnCopy")}</span>
      </button>
      <button
        class="btn btn-secondary"
        id="sttDownload"
        onclick={() =>
          downloadText(
            copyText(),
            (stt.filename || t("defaultTranscriptionFile")).replace(
              /\.[^/.]+$/,
              "",
            ) + ".txt",
          )}
      >
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" x2="12" y1="15" y2="3" />
        </svg>
        <span>{t("btnDownload")}</span>
      </button>
    </div>
    <textarea
      class="result-content"
      id="sttResultText"
      readonly
      spellcheck="false"
      value={displayText()}
      data-filename={stt.filename}
      style:height={height ? height + "px" : undefined}></textarea>
    <div
      class="stt-resize-handle"
      id="sttResizeHandle"
      aria-label="Resize transcription panel"
      role="separator"
      onpointerdown={resizeStart}
      onpointermove={resizeMove}
      onpointerup={resizeStop}
      onpointercancel={resizeStop}
      onlostpointercapture={resizeStop}
      class:active={resizing}
      aria-valuenow={height || 300}
    ></div>
  </div>
</section>
