<script lang="ts">
  import {
    tts,
    languages,
    voices,
    voiceLabel,
    setLanguage,
    setVoice,
    placeholder,
    inputLimit,
    saveDraft,
    progressText,
    percent,
    cancel,
    synthesize,
    readFile,
  } from "../tts/state.svelte";
  import { settings } from "../settings.svelte";
  import { t, helpers, ttsLabels } from "../i18n/state.svelte";
  import { download } from "../api/client";
  let fileInput: HTMLInputElement;
  let dragging = $state(false);
</script>

<section
  class="panel"
  id="tts-panel"
  role="tabpanel"
  class:active={settings.tab === "tts"}
>
  <div
    class="dropzone"
    id="ttsDropzone"
    style="margin-bottom: 1rem;"
    onclick={() => fileInput.click()}
    ondragover={(e) => {
      e.preventDefault();
      dragging = true;
    }}
    ondragleave={() => (dragging = false)}
    ondrop={(e) => {
      e.preventDefault();
      dragging = false;
      const file = e.dataTransfer?.files[0];
      if (file) void readFile(file);
    }}
    class:dragover={dragging}
    role="button"
    tabindex="0"
    onkeydown={(e) => {
      if (e.key === "Enter") fileInput.click();
    }}
  >
    <input
      type="file"
      id="ttsFileInput"
      accept=".txt,.md,.text"
      hidden
      bind:this={fileInput}
      onclick={(e) => e.stopPropagation()}
      onchange={(e) => {
        const file = e.currentTarget.files?.[0];
        if (file) void readFile(file);
      }}
    />
    <svg
      width="32"
      height="32"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="1.5"
    >
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="12" y1="18" x2="12" y2="12" />
      <line x1="9" y1="15" x2="15" y2="15" />
    </svg>
    <span class="dropzone-text">{t("ttsDropzoneText")}</span>
    <span class="dropzone-hint">{t("ttsDropzoneHint")}</span>
  </div>

  <div class="text-input-wrapper">
    <textarea
      class="text-input"
      id="ttsInput"
      data-i18n="ttsPlaceholder"
      data-i18n-attr="placeholder"
      bind:value={tts.text}
      placeholder={placeholder()}
      oninput={saveDraft}></textarea>
  </div>

  <div class="input-footer">
    <select
      class="speaker-select"
      id="ttsLanguage"
      value={tts.language}
      onchange={(e) => setLanguage(e.currentTarget.value)}
      >{#each languages() as language (language.id)}<option value={language.id}
          >{ttsLabels().languages[language.id] || language.id}</option
        >{/each}</select
    >
    <select
      class="speaker-select"
      id="ttsVoice"
      value={tts.voice}
      onchange={(e) => setVoice(e.currentTarget.value)}
      >{#each voices() as voice (voice.id)}<option value={voice.id}
          >{voiceLabel(voice)}</option
        >{/each}</select
    >
    <span
      class="char-count"
      class:warning={inputLimit() > 0 && tts.text.length > inputLimit()}
      id="ttsCharCount"
      >{helpers().formatTtsCharLine(
        tts.text.length,
        settings.config.tts_max_chars || 1000,
        t,
      )}</span
    >
  </div>

  <button
    class="btn btn-primary"
    id="ttsSubmit"
    style="margin-top: 1rem; width: 100%;"
    onclick={synthesize}
    disabled={tts.busy}
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
      <polygon points="5 3 19 12 5 21 5 3" />
    </svg>
    <span>{t("btnSynthesize")}</span>
  </button>

  <div
    class="progress"
    id="ttsProgress"
    style="margin-top: 1rem;"
    class:active={tts.progress}
  >
    <div class="progress-bar">
      <div
        class="progress-fill"
        id="ttsProgressFill"
        style:width={percent() + "%"}
      ></div>
    </div>
    <div class="progress-text" id="ttsProgressText">{progressText()}</div>
    <button
      class="cancel-btn"
      id="ttsCancel"
      class:active={tts.busy}
      onclick={cancel}
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

  <div class="error" id="ttsError" class:active={!!tts.error}>{tts.error}</div>

  <div class="result" id="ttsResult" class:active={tts.result}>
    <div class="result-header">
      <span class="result-title">{t("resultTitleSynth")}</span>
      <span class="result-meta" id="ttsMeta"
        >{helpers().formatTtsMeta(tts.chunks, tts.duration)}</span
      >
    </div>
    <div class="audio-player">
      <audio id="ttsAudio" controls src={tts.url || undefined}></audio>
    </div>
    <button
      class="btn btn-secondary"
      id="ttsDownload"
      style="margin-top: 0.75rem;"
      onclick={() => download(tts.url, tts.filename)}
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
      <span>{t("btnDownloadWav")}</span>
    </button>
  </div>
</section>
