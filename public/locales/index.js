(function () {
    var locales = window.__resonanceLocales || {};

    window.__resonanceLocales = locales;
    window.__registerResonanceLocale = function (locale, definition) {
        if (!locale || typeof locale !== 'string') return;
        if (definition && typeof definition === 'object' && ('messages' in definition || 'helpers' in definition || 'speakerNames' in definition)) {
            locales[locale] = {
                messages: definition.messages || {},
                helpers: definition.helpers || {},
                speakerNames: definition.speakerNames || {}
            };
            return;
        }
        locales[locale] = {
            messages: definition || {},
            helpers: {},
            speakerNames: {}
        };
    };
})();
