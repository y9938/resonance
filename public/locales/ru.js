(function () {
    if (typeof window.__registerResonanceLocale !== 'function') return;

    function pluralRu(n, forms) {
        var abs = Math.abs(n) % 100;
        var rem = abs % 10;
        if (abs > 10 && abs < 20) return forms[2];
        if (rem === 1) return forms[0];
        if (rem >= 2 && rem <= 4) return forms[1];
        return forms[2];
    }

    window.__registerResonanceLocale('ru', {
        messages: {
            pageTitle: 'Resonance',
            metaDescription: 'Resonance — распознавание речи и синтез речи',
            ariaLangGroup: 'Язык интерфейса',
            tabStt: 'Речь → текст',
            tabTts: 'Текст → речь',
            sttDropzoneText: 'Перетащите аудио или нажмите для выбора файлов',
            configLoading: 'Загрузка настроек…',
            btnHide: 'Скрыть',
            btnCancel: 'Отмена',
            resultTitleTranscription: 'Транскрипт',
            btnCopy: 'Копировать',
            btnDownload: 'Скачать',
            ttsDropzoneText: 'Перетащите текстовые файлы или нажмите для выбора',
            ttsDropzoneHint: 'Любой текст — любой размер',
            ttsPlaceholder: 'Введите текст для синтеза…',
            ttsPlaceholderLimited: 'Введите текст для синтеза… (до {limit} символов)',
            ttsPlaceholderUnlimited: 'Введите текст для синтеза…',
            btnSynthesize: 'Синтезировать',
            resultTitleSynth: 'Синтезированное аудио',
            btnDownloadWav: 'Скачать WAV',
            hintSttMedia: 'Любые файлы — {limit}',
            hintLimitMb: 'до {mb} МБ',
            hintAnySize: 'любого размера',
            progressUploading: 'Загрузка…',
            progressProcessing: 'Обработка…',
            progressComplete: 'Готово',
            progressChunk: 'Фрагмент {current} / {total}',
            progressStarting: 'Запуск…',
            errNetwork: 'Ошибка сети',
            errPleaseEnterText: 'Введите текст',
            errTextTooLong: 'Текст слишком длинный',
            errUploadFailed: 'Не удалось загрузить файл',
            errNoResponseBody: 'Пустой ответ сервера',
            errRequestFailed: 'Запрос не выполнен',
            errProcessingFailed: 'Ошибка обработки',
            errMicUnsupported: 'Запись с микрофоном не поддерживается в этом браузере',
            errMicPermission: 'Доступ к микрофону запрещён',
            errMicEmpty: 'Запись пустая',
            toastCopied: 'Скопировано в буфер обмена',
            toastCopyFailed: 'Не удалось скопировать',
            toastReadFailed: 'Не удалось прочитать файл',
            localeSearchEmpty: 'Подходящие языки не найдены',
            charUnit: 'симв.',
            defaultTranscriptionFile: 'транскрипт',
            sttViewBlocks: 'Блоки',
            sttViewContinuous: 'Сплошной текст',
            sttMicTitle: 'Микрофон',
            sttMicHintIdle: 'Запишите речь с микрофона',
            sttMicHintRecording: 'Идёт запись… Нажмите «Стоп», когда закончите',
            sttMicHintReady: 'Запись готова к отправке',
            sttMicStart: 'Начать запись',
            sttMicStop: 'Стоп',
            sttMicSend: 'Отправить запись',
            sttMicDiscard: 'Сбросить',
            jobsOpenAria: 'Открыть список задач',
            jobsCloseAria: 'Закрыть список задач',
            jobsDrawerTitle: 'Задачи',
            jobsEmpty: 'Список задач пока пуст.',
            jobsLoading: 'Загрузка задач…',
            jobsLoadError: 'Не удалось загрузить список задач.',
            jobsTypeStt: 'STT',
            jobsTypeTts: 'TTS',
            jobsStateQueued: 'В очереди',
            jobsStateRunning: 'Выполняется',
            jobsStateCompleted: 'Завершена',
            jobsStateFailed: 'Ошибка',
            jobsStateCancelled: 'Отменена',
            jobsLoadingMore: 'Подгрузка…',
            jobsListEnd: 'Все задачи показаны.',
            sttBatchTitle: 'Пакетная очередь',
            sttBatchNextReady: 'Следующий готовый',
            sttBatchDownloadAll: 'Скачать все',
            sttBatchCancelCurrent: 'Отменить текущий',
            sttBatchCancel: 'Отменить активные',
            sttBatchSummary: 'Готово {done} / {total}',
            sttBatchEmpty: 'В этом пакете нет файлов.',
            jobsBatchTitle: 'Пакет STT',
            jobsBatchSummary: '{done} / {total}',
            jobsBatchOpen: 'Открыть'
        },
        helpers: {
            formatCount: function (value) {
                return Number(value || 0).toLocaleString('ru-RU');
            },
            formatSttMeta: function (n) {
                return n + ' ' + pluralRu(n, ['сегмент', 'сегмента', 'сегментов']);
            },
            formatSttProcessedTextDuration: function (seconds) {
                if (!Number.isFinite(seconds) || seconds <= 0) return '0.0с';
                if (seconds < 60) {
                    return seconds.toFixed(1) + 'с';
                }
                var total = Math.max(1, Math.round(seconds));
                var hours = Math.floor(total / 3600);
                var mins = Math.floor((total % 3600) / 60);
                var secs = total % 60;
                if (hours > 0) {
                    return String(hours) + 'ч ' + String(mins) + 'м';
                }
                return String(mins) + 'м ' + String(secs) + 'с';
            },
            formatSttProcessedTextLabel: function (durationText) {
                return 'Обработано текста: ' + durationText;
            },
            formatTtsCharLine: function (len, maxChars, t) {
                var chunkN = len === 0 ? 0 : Math.ceil(len / maxChars);
                return len + ' ' + t('charUnit') + ' · ' + chunkN + ' ' + pluralRu(chunkN, ['часть', 'части', 'частей']);
            },
            formatTtsMeta: function (chunks, durationSec) {
                return chunks + ' ' + pluralRu(chunks, ['часть', 'части', 'частей']) + ' · ' + Number(durationSec).toFixed(1) + ' с';
            },
            formatTtsInputTooLongMessage: function (inputLimit, formatCount) {
                return 'Текст слишком длинный (макс. ' + formatCount(inputLimit) + ' символов)';
            },
            formatMicDuration: function (seconds) {
                if (!Number.isFinite(seconds) || seconds <= 0) return null;
                var total = Math.max(1, Math.round(seconds));
                var mins = Math.floor(total / 60);
                var secs = total % 60;
                if (mins > 0) {
                    return String(mins) + ':' + String(secs).padStart(2, '0');
                }
                return String(secs) + 'с';
            },
            formatDateTime: function (date) {
                return date.toLocaleString('ru-RU');
            }
        },
        tts: {
            languages: {
                ru: 'Русский',
                en: 'Английский'
            },
            voiceGroups: {
                silero_ru: {
                    ru_alexandr: 'Александр (муж.)',
                    ru_alfia: 'Альфия (жен.)',
                    ru_alfia2: 'Альфия 2 (жен.)',
                    ru_bogdan: 'Богдан (муж.)',
                    ru_dmitriy: 'Дмитрий (муж.)',
                    ru_ekaterina: 'Екатерина (жен.)',
                    ru_vika: 'Вика (жен.)',
                    ru_gamat: 'Гамат (муж.)',
                    ru_igor: 'Игорь (муж.)',
                    ru_karina: 'Карина (жен.)',
                    ru_kejilgan: 'Кейжиган (муж.)',
                    ru_kermen: 'Кермен (жен.)',
                    ru_marat: 'Марат (муж.)',
                    ru_miyau: 'Мияу (жен.)',
                    ru_nurgul: 'Нургуль (жен.)',
                    ru_oksana: 'Оксана (жен.)',
                    ru_onaoy: 'Онаой (муж.)',
                    ru_ramilia: 'Рамиля (жен.)',
                    ru_roman: 'Роман (муж.) ★',
                    ru_safarhuja: 'Сафархуджа (муж.)',
                    ru_saida: 'Саида (жен.)',
                    ru_sibday: 'Сибдай (муж.)',
                    ru_zara: 'Зара (жен.)',
                    ru_zhadyra: 'Жадыра (жен.) ★',
                    ru_zhazira: 'Жазира (жен.)',
                    ru_zinaida: 'Зинаида (жен.)',
                    ru_eduard: 'Эдуард (муж.)'
                },
                kokoro_en: {
                    af_heart: 'Харт (жен.)',
                    af_alloy: 'Аллой (жен.)',
                    af_aoede: 'Аоэде (жен.)',
                    af_bella: 'Белла (жен.)',
                    af_jessica: 'Джессика (жен.)',
                    af_kore: 'Коре (жен.)',
                    af_nicole: 'Николь (жен.)',
                    af_nova: 'Нова (жен.)',
                    af_river: 'Ривер (жен.)',
                    af_sarah: 'Сара (жен.)',
                    af_sky: 'Скай (жен.)',
                    am_adam: 'Адам (муж.)',
                    am_echo: 'Эхо (муж.)',
                    am_eric: 'Эрик (муж.)',
                    am_fenrir: 'Фенрир (муж.)',
                    am_liam: 'Лиам (муж.)',
                    am_michael: 'Майкл (муж.)',
                    am_onyx: 'Оникс (муж.)',
                    am_puck: 'Пак (муж.)',
                    am_santa: 'Санта (муж.)',
                    bf_alice: 'Алиса (жен.)',
                    bf_emma: 'Эмма (жен.)',
                    bf_isabella: 'Изабелла (жен.)',
                    bf_lily: 'Лили (жен.)',
                    bm_daniel: 'Дэниел (муж.)',
                    bm_fable: 'Фейбл (муж.)',
                    bm_george: 'Джордж (муж.)',
                    bm_lewis: 'Льюис (муж.)'
                }
            }
        }
    });
})();
