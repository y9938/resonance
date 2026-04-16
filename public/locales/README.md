# Locales

Rules for UI translations:

- English DOM text lives in `public/index.html` and is the SEO/static fallback.
- English is never loaded from a locale file.
- `public/locales/manifest.js` is the single source of truth for available locales.
- Locale codes in the manifest must be canonical BCP 47 tags such as `en`, `ru`, `zh-CN`.
- `public/locales/*.js` stores non-English locale dictionaries.
- A locale file may export `messages`, optional `helpers`, and optional `speakerNames`.
- Non-English locales are loaded lazily from `src` entries in `window.__resonanceLocaleManifest`.
- Do not duplicate English `data-i18n` values in locale files.
- JS-only English strings stay in `public/index.html` as the runtime fallback.
- Missing keys in a locale must fall back to English.
- Locale matching uses `Intl.getCanonicalLocales()` with `exact -> base language -> default locale`.

To add a new language:

1. Create `public/locales/<lang>.js`
2. Register it via `window.__registerResonanceLocale('<lang>', { messages, helpers, speakerNames })`
3. Add one entry to `public/locales/manifest.js`
   Example: `{ code: 'de', label: 'German', nativeLabel: 'Deutsch', src: '/locales/de.js' }`
