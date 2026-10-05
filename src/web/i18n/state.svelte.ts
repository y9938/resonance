import type { LocaleDefinition } from "./types";
import * as en from "./en";
import ru from "./ru";
import zh from "./zh-CN";

export const locales = [
  { value: "en", label: "English", native: "English" },
  { value: "ru", label: "Russian", native: "Русский" },
  { value: "zh-CN", label: "Chinese", native: "简体中文" },
];
function canonical(value: string) {
  try {
    return Intl.getCanonicalLocales(value.replaceAll("_", "-"))[0];
  } catch {
    return "";
  }
}
function supported(value: string) {
  const code = canonical(value);
  return (
    locales.find((l) => l.value.toLowerCase() === code.toLowerCase())?.value ||
    locales.find((l) => l.value === code.split("-")[0])?.value
  );
}
const initial =
  supported(localStorage.getItem("resonance_locale") || "") ||
  navigator.languages.map(supported).find(Boolean) ||
  "en";
export const locale = $state({ code: initial });
const definitions: Record<string, LocaleDefinition> = { en, ru, "zh-CN": zh };
export function setLocale(code: string) {
  if (!(code in definitions)) return false;
  locale.code = code;
  localStorage.setItem("resonance_locale", code);
  document.documentElement.lang = code;
  return true;
}
export function t(key: string, params?: Record<string, unknown>): string {
  const messages = definitions[locale.code].messages;
  let text = messages[key] ?? en.messages[key] ?? key;
  for (const [name, value] of Object.entries(params || {}))
    text = text.split("{" + name + "}").join(String(value));
  return text;
}
export function helpers() {
  return { ...en.helpers, ...definitions[locale.code].helpers };
}
export function ttsLabels() {
  const translated = definitions[locale.code].tts || {};
  return {
    languages: { ...en.tts.languages, ...translated.languages },
    voiceGroups: { ...en.tts.voiceGroups, ...translated.voiceGroups },
  };
}
