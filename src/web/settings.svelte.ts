import { request } from "./api/client";
import type { Config, Model, JobType } from "./api/types";
import { locale } from "./i18n/state.svelte";
export const settings = $state({
  config: {} as Config,
  models: [] as Model[],
  names: {} as Record<string, string>,
  loaded: false,
  language: localStorage.getItem("resonance_sttLanguage") || "",
  model: localStorage.getItem("resonance_sttModelOverride") || "",
  autoDetect: false,
  diarization: localStorage.getItem("resonance_sttDiarization") === "true",
  source: "mic" as "mic" | "sys",
  mode:
    localStorage.getItem("resonance_sttMicMode") === "live"
      ? "live"
      : "dictation",
  includeMic: localStorage.getItem("resonance_sttSysIncludeMic") === "true",
  tab: (localStorage.getItem("resonance_activeTab") || "stt") as JobType,
  toast: "",
});
let toastTimer: ReturnType<typeof setTimeout>;
export function toast(message: string) {
  settings.toast = message;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (settings.toast = ""), 2500);
}
export function switchTab(tab: JobType) {
  settings.tab = tab;
  localStorage.setItem("resonance_activeTab", tab);
}
export function compatible() {
  return settings.models.filter((model) =>
    model.languages.includes(settings.language),
  );
}
export function setLanguage(value: string) {
  settings.language = value;
  localStorage.setItem("resonance_sttLanguage", value);
  if (
    compatible().length < 2 ||
    !compatible().some((model) => model.id === settings.model)
  )
    setModel("");
}
export function setModel(value: string) {
  settings.model = value;
  localStorage.setItem("resonance_sttModelOverride", value);
}
export function languageOptions() {
  const local = new Intl.DisplayNames([locale.code], { type: "language" });
  const en = new Intl.DisplayNames(["en"], { type: "language" });
  return [...new Set(settings.models.flatMap((m) => m.languages))]
    .map((value) => {
      const upstream = settings.names[value] || value;
      const fallback = upstream.charAt(0).toUpperCase() + upstream.slice(1);
      const english = en.of(value),
        localized = local.of(value),
        native = new Intl.DisplayNames([value], { type: "language" }).of(value);
      return {
        value,
        label:
          localized && localized !== value
            ? localized
            : english && english !== value
              ? english
              : fallback,
        english: english && english !== value ? english : fallback,
        native: native && native !== value ? native : fallback,
      };
    })
    .sort((a, b) => a.label.localeCompare(b.label, locale.code));
}
export async function loadSettings() {
  const [config, catalog] = await Promise.all([
    request<Config>("/config"),
    request<{
      stt: { models: Model[]; language_names?: Record<string, string> };
    }>("/models"),
  ]);
  settings.config = config;
  settings.models = catalog.stt.models;
  settings.names = catalog.stt.language_names || {};
  const codes = settings.models.flatMap((m) => m.languages);
  setLanguage(
    [settings.language, locale.code.split("-")[0], "en", codes[0]].find((v) =>
      codes.includes(v),
    ) || "",
  );
  settings.loaded = true;
}
