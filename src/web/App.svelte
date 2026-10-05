<script lang="ts">
  import { onMount } from "svelte";
  import { settings, loadSettings, switchTab } from "./settings.svelte";
  import { t, locale, locales, setLocale } from "./i18n/state.svelte";
  import { history, loadHistory } from "./jobs/state.svelte";
  import { recover } from "./session";
  import * as stt from "./stt/state.svelte";
  import * as tts from "./tts/state.svelte";
  import * as capture from "./stt/capture.svelte";
  import LanguagePicker from "./components/LanguagePicker.svelte";
  import JobsDrawer from "./components/JobsDrawer.svelte";
  import SttPanel from "./components/SttPanel.svelte";
  import TtsPanel from "./components/TtsPanel.svelte";
  import StarField from "./components/StarField.svelte";
  let main: HTMLElement;
  onMount(() => {
    document.documentElement.lang = locale.code;
    void loadSettings()
      .then(() => tts.initialize())
      .catch(() => (stt.stt.error = t("errNetwork")));
    void recover();
    const close = () => {
      stt.dispose();
      tts.dispose();
      capture.dispose();
    };
    window.addEventListener("pagehide", close);
    return () => {
      window.removeEventListener("pagehide", close);
      close();
    };
  });
</script>

<svelte:head
  ><title>{t("pageTitle")}</title><meta
    name="description"
    content={t("metaDescription")}
  /></svelte:head
>
<svelte:window
  onkeydown={(e) => {
    if (e.key === "Escape") history.open = false;
  }}
/>
<header class="header">
  <div class="header-left">
    <button
      class="menu-btn"
      id="jobsMenuBtn"
      type="button"
      data-i18n="jobsOpenAria"
      data-i18n-attr="aria-label"
      onclick={() => {
        history.open = !history.open;
        if (history.open) void loadHistory();
      }}
      aria-expanded={history.open}
      aria-label={t("jobsOpenAria")}
    >
      <svg
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
      >
        <line x1="3" y1="6" x2="21" y2="6"></line>
        <line x1="3" y1="12" x2="21" y2="12"></line>
        <line x1="3" y1="18" x2="21" y2="18"></line>
      </svg>
    </button>
  </div>
  <div class="header-center">
    <div class="brand">
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
        <line x1="12" x2="12" y1="19" y2="22" />
      </svg>
      Resonance
    </div>
  </div>
  <div class="header-right">
    <div class="locale-picker" id="localePicker">
      <LanguagePicker
        id="localeInput"
        options={locales}
        value={locale.code}
        onchange={setLocale}
        localePicker={true}
      />
    </div>
  </div>
</header>

<JobsDrawer />
<main class="main" bind:this={main}>
  <div class="tabs" role="tablist">
    <button
      class="tab"
      class:active={settings.tab === "stt"}
      role="tab"
      data-tab="stt"
      aria-selected={settings.tab === "stt"}
      onclick={() => switchTab("stt")}>{t("tabStt")}</button
    >
    <button
      class="tab"
      class:active={settings.tab === "tts"}
      role="tab"
      data-tab="tts"
      aria-selected={settings.tab === "tts"}
      onclick={() => switchTab("tts")}>{t("tabTts")}</button
    >
  </div>
  <SttPanel /><TtsPanel />
</main>
<StarField content={main} />
{#if settings.toast}<div class="toast">{settings.toast}</div>{/if}
