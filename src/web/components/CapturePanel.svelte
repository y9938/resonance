<script lang="ts">
  import {
    capture,
    start,
    stop,
    send,
    discard,
    setMode,
    setSource,
    setIncludeMicrophone,
    canConfigureCapture,
  } from "../stt/capture.svelte";
  import { settings } from "../settings.svelte";
  import { t, helpers } from "../i18n/state.svelte";
</script>

<div class="mic-box" id="sttMicBox" class:mic-recording={capture.active}>
  <div class="mic-header">
    <div
      class="capture-source-tabs"
      style="display: flex; gap: 1rem; font-size: 0.875rem; font-weight: 500; align-items: center;"
    >
      <span
        id="tabMic"
        class="capture-tab active"
        style="color: var(--accent); cursor: pointer;"
        onclick={() => setSource("mic")}
        class:active={settings.source === "mic"}
        style:color={settings.source === "mic"
          ? "var(--accent)"
          : "var(--text-muted)"}
        role="button"
        tabindex="0"
        onkeydown={(e) => {
          if (e.key === "Enter") setSource("mic");
        }}><span>{t("sttMicTitle")}</span></span
      >
      <span
        id="tabSys"
        class="capture-tab"
        style="color: var(--text-muted); cursor: pointer;"
        hidden={settings.config.system_audio_enabled === false}
        onclick={() => setSource("sys")}
        class:active={settings.source === "sys"}
        style:color={settings.source === "sys"
          ? "var(--accent)"
          : "var(--text-muted)"}
        role="button"
        tabindex="0"
        onkeydown={(e) => {
          if (e.key === "Enter") setSource("sys");
        }}><span>{t("sttSysTitle")}</span></span
      >
      <div
        id="sttMicModeToggle"
        class="stt-view-toggle"
        style="margin-bottom: 0; margin-left: 0.25rem;"
        style:display={settings.source === "mic" ? "inline-flex" : "none"}
      >
        <button
          class="stt-view-btn active"
          id="sttMicModeDictation"
          type="button"
          class:active={settings.mode === "dictation"}
          disabled={!canConfigureCapture()}
          onclick={() => setMode("dictation")}>{t("sttModeDictation")}</button
        >
        <button
          class="stt-view-btn"
          id="sttMicModeLive"
          type="button"
          class:active={settings.mode === "live"}
          disabled={!canConfigureCapture()}
          onclick={() => setMode("live")}>{t("sttModeLive")}</button
        >
      </div>
      <div
        id="sttSysIncludeMicContainer"
        style="display: none; align-items: center; gap: 0.35rem; margin-left: 0.5rem;"
        style:display={settings.source === "sys" ? "flex" : "none"}
      >
        <input
          type="checkbox"
          id="sttSysIncludeMic"
          style="cursor: pointer; width: 14px; height: 14px; accent-color: var(--accent);"
          checked={capture.config?.include_microphone ?? settings.includeMic}
          disabled={!canConfigureCapture()}
          onchange={(event) => {
            setIncludeMicrophone(event.currentTarget.checked);
          }}
        />
        <label
          for="sttSysIncludeMic"
          style="cursor: pointer; font-size: 0.8125rem; color: var(--text-secondary); user-select: none;"
          >{t("sttSysIncludeMic")}</label
        >
      </div>
    </div>
    <span class="mic-timer">
      <span class="mic-timer-dot"></span>
      <span id="sttMicTimer"
        >{Math.floor(capture.elapsed / 60) +
          ":" +
          String(capture.elapsed % 60).padStart(2, "0")}</span
      >
    </span>
  </div>
  <div class="mic-hint" id="sttMicHint" data-mic-hint-key="sttMicHintIdle">
    {t(capture.hint)}{capture.duration
      ? " · " + helpers().formatMicDuration(capture.duration)
      : ""}
  </div>
  <div class="mic-actions">
    <button
      class="btn btn-secondary"
      id="sttMicStart"
      type="button"
      onclick={start}
      style:display={capture.active ? "none" : ""}
      disabled={capture.state === "starting" || capture.state === "stopping"}
      >{t(settings.source === "sys" ? "sttSysStart" : "sttMicStart")}</button
    >
    <button
      class="btn btn-secondary"
      id="sttMicStop"
      type="button"
      style="display: none;"
      onclick={stop}
      style:display={capture.active ? "" : "none"}
      disabled={capture.state === "stopping"}>{t("sttMicStop")}</button
    >
    <button
      class="btn btn-primary"
      id="sttMicSend"
      type="button"
      style="display: none;"
      onclick={send}
      style:display={capture.preview ? "" : "none"}>{t("sttMicSend")}</button
    >
    <button
      class="btn btn-secondary"
      id="sttMicDiscard"
      type="button"
      style="display: none;"
      onclick={discard}
      style:display={capture.preview ? "" : "none"}>{t("sttMicDiscard")}</button
    >
  </div>
  <div class="mic-preview" id="sttMicPreview" class:active={!!capture.preview}>
    <audio
      id="sttMicAudio"
      controls
      preload="metadata"
      src={capture.preview || undefined}
    ></audio>
  </div>
</div>
