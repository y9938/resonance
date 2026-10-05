# Locales

`en.ts` contains the English fallback messages, formatters and TTS voice labels.
`ru.ts` and `zh-CN.ts` preserve the existing translations and locale-specific formatters.

`state.svelte.ts` owns the selected locale. Components call `t()` and `helpers()`
inside reactive expressions, so switching language updates the UI without DOM scans.
Locale selection is saved under `resonance_locale`; unsupported messages fall back to English.
Add a locale definition and a selector entry together when introducing another language.
