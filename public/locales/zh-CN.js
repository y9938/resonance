(function () {
    if (typeof window.__registerResonanceLocale !== 'function') return;

    window.__registerResonanceLocale('zh-CN', {
        messages: {
            speakerMic: '🎤 麦克风',
            speakerSys: '🔊 系统声音',
            pageTitle: 'Resonance',
            metaDescription: 'Resonance — 语音识别与语音合成',
            ariaLangGroup: '界面语言',
            tabStt: '语音 → 文本',
            tabTts: '文本 → 语音',
            sttDropzoneText: '拖拽音频文件或点击选择',
            configLoading: '正在加载设置…',
            btnHide: '隐藏',
            btnCancel: '取消',
            resultTitleTranscription: '转录结果',
            btnCopy: '复制',
            btnDownload: '下载',
            ttsDropzoneText: '拖拽文本文件或点击选择',
            ttsDropzoneHint: '任意文本 — 不限大小',
            ttsPlaceholder: '请输入要合成的文本…',
            ttsPlaceholderLimited: '请输入要合成的文本…（最多 {limit} 字符）',
            ttsPlaceholderUnlimited: '请输入要合成的文本…',
            btnSynthesize: '合成语音',
            resultTitleSynth: '合成音频',
            btnDownloadWav: '下载 WAV',
            hintSttMedia: '支持多个文件 — {limit}',
            hintLimitMb: '最大 {mb} MB',
            hintAnySize: '不限大小',
            progressUploading: '上传中…',
            progressProcessing: '处理中…',
            progressDiarizing: '说话人分离中…',
            progressComplete: '已完成',
            progressChunk: '片段 {current} / {total}',
            progressStarting: '启动中…',
            errNetwork: '网络错误',
            errPleaseEnterText: '请输入文本',
            errTextTooLong: '文本过长',
            errUploadFailed: '文件上传失败',
            errNoResponseBody: '服务器响应为空',
            errRequestFailed: '请求失败',
            errProcessingFailed: '处理失败',
            errMicUnsupported: '当前浏览器不支持麦克风录音',
            errMicPermission: '麦克风权限被拒绝',
            errMicEmpty: '录音内容为空',
            toastCopied: '已复制到剪贴板',
            toastCopyFailed: '复制失败',
            toastReadFailed: '文件读取失败',
            localeSearchEmpty: '未找到匹配的语言',
            charUnit: '字符',
            defaultTranscriptionFile: '转录文本',
            sttViewBlocks: '分块',
            sttViewContinuous: '连续文本',
            sttMicTitle: '麦克风',
            sttMicHintIdle: '使用麦克风录制语音',
            sttMicHintRecording: '正在录音… 完成后请按「停止」',
            sttMicHintReady: '录音已就绪，可发送',
            sttMicStart: '开始录音',
            sttMicStop: '停止',
            sttMicSend: '发送录音',
            sttMicDiscard: '丢弃',
            sttSysTitle: '系统声音',
            sttSysHintIdle: '捕获系统内部音频（会议、视频等）',
            sttSysHintCapturing: '正在捕获系统音频… 完成后请按「停止」',
            sttSysHintProcessing: '正在处理系统音频…',
            sttSysStart: '开始捕获',
            sttSysIncludeMic: '包含麦克风',
            jobsOpenAria: '打开任务列表',
            jobsCloseAria: '关闭任务列表',
            jobsDrawerTitle: '任务',
            jobsEmpty: '暂无任务。',
            jobsLoading: '正在加载任务…',
            jobsLoadError: '加载任务列表失败。',
            jobsTypeStt: 'STT',
            jobsTypeTts: 'TTS',
            jobsStateQueued: '排队中',
            jobsStateRunning: '处理中',
            jobsStateCompleted: '已完成',
            jobsStateFailed: '失败',
            jobsStateCancelled: '已取消',
            jobsLoadingMore: '加载更多…',
            jobsListEnd: '已显示全部任务。',
            sttBatchTitle: '批量队列',
            sttBatchNextReady: '下一个已完成',
            sttBatchDownloadAll: '全部下载',
            sttBatchCancelCurrent: '取消当前',
            sttBatchCancel: '取消活动任务',
            sttBatchSummary: '已完成 {done} / {total}',
            sttBatchEmpty: '此批次中没有文件。',
            jobsBatchTitle: 'STT 批次',
            jobsBatchSummary: '{done} / {total}',
            jobsBatchOpen: '打开',
            sttLanguageLabel: 'STT 语言',
            sttLangRu: '俄语',
            sttLangEn: '英语',
            sttModelLabel: 'STT 模型',
            sttModelWhisper: 'Whisper (默认)',
            sttModelGranite: 'IBM Granite',
            sttDiarizationLabelText: '角色分离'
        },
        helpers: {
            formatCount: function (value) {
                return Number(value || 0).toLocaleString('zh-CN');
            },
            formatSttMeta: function (n) {
                return n + ' 段';
            },
            formatSttProcessedTextDuration: function (seconds) {
                if (!Number.isFinite(seconds) || seconds <= 0) return '0.0秒';
                if (seconds < 60) {
                    return seconds.toFixed(1) + '秒';
                }
                var total = Math.max(1, Math.round(seconds));
                var hours = Math.floor(total / 3600);
                var mins = Math.floor((total % 3600) / 60);
                var secs = total % 60;
                if (hours > 0) {
                    return String(hours) + '小时 ' + String(mins) + '分';
                }
                return String(mins) + '分 ' + String(secs) + '秒';
            },
            formatSttProcessedTextLabel: function (durationText) {
                return '已处理文本：' + durationText;
            },
            formatTtsCharLine: function (len, maxChars, t) {
                var chunkN = len === 0 ? 0 : Math.ceil(len / maxChars);
                return len + ' ' + t('charUnit') + ' · ' + chunkN + ' 段';
            },
            formatTtsMeta: function (chunks, durationSec) {
                return chunks + ' 段 · ' + Number(durationSec).toFixed(1) + ' 秒';
            },
            formatTtsInputTooLongMessage: function (inputLimit, formatCount) {
                return '文本过长（最多 ' + formatCount(inputLimit) + ' 字符）';
            },
            formatMicDuration: function (seconds) {
                if (!Number.isFinite(seconds) || seconds <= 0) return null;
                var total = Math.max(1, Math.round(seconds));
                var mins = Math.floor(total / 60);
                var secs = total % 60;
                if (mins > 0) {
                    return String(mins) + ' 分 ' + String(secs) + ' 秒';
                }
                return String(secs) + ' 秒';
            },
            formatDateTime: function (date) {
                return date.toLocaleString('zh-CN');
            }
        },
        tts: {
            languages: {
                ru: '俄语',
                en: '英语'
            },
            voiceGroups: {
                silero_ru: {
                    ru_alexandr: '亚历山大 (男)',
                    ru_alfia: '阿尔菲亚 (女)',
                    ru_alfia2: '阿尔菲亚 2 (女)',
                    ru_bogdan: '博格丹 (男)',
                    ru_dmitriy: '德米特里 (男)',
                    ru_ekaterina: '叶卡捷琳娜 (女)',
                    ru_vika: '维卡 (女)',
                    ru_gamat: '加马特 (男)',
                    ru_igor: '伊戈尔 (男)',
                    ru_karina: '卡琳娜 (女)',
                    ru_kejilgan: '凯吉尔干 (男)',
                    ru_kermen: '克尔曼 (女)',
                    ru_marat: '马拉特 (男)',
                    ru_miyau: '米娅乌 (女)',
                    ru_nurgul: '努尔古丽 (女)',
                    ru_oksana: '奥克萨娜 (女)',
                    ru_onaoy: '奥纳奥伊 (男)',
                    ru_ramilia: '拉米莉亚 (女)',
                    ru_roman: '罗曼 (男) ★',
                    ru_safarhuja: '萨法尔胡贾 (男)',
                    ru_saida: '赛伊达 (女)',
                    ru_sibday: '西布戴 (男)',
                    ru_zara: '扎拉 (女)',
                    ru_zhadyra: '扎德拉 (女) ★',
                    ru_zhazira: '扎兹拉 (女)',
                    ru_zinaida: '季娜伊达 (女)',
                    ru_eduard: '爱德华 (男)'
                },
                kokoro_en: {
                    af_heart: '哈特 (女)',
                    af_alloy: '阿洛伊 (女)',
                    af_aoede: '阿俄厄得 (女)',
                    af_bella: '贝拉 (女)',
                    af_jessica: '杰西卡 (女)',
                    af_kore: '科蕾 (女)',
                    af_nicole: '妮可 (女)',
                    af_nova: '诺娃 (女)',
                    af_river: '里弗 (女)',
                    af_sarah: '萨拉 (女)',
                    af_sky: '斯凯 (女)',
                    am_adam: '亚当 (男)',
                    am_echo: '艾科 (男)',
                    am_eric: '埃里克 (男)',
                    am_fenrir: '芬里尔 (男)',
                    am_liam: '利亚姆 (男)',
                    am_michael: '迈克尔 (男)',
                    am_onyx: '奥尼克斯 (男)',
                    am_puck: '帕克 (男)',
                    am_santa: '桑塔 (男)',
                    bf_alice: '爱丽丝 (女)',
                    bf_emma: '艾玛 (女)',
                    bf_isabella: '伊莎贝拉 (女)',
                    bf_lily: '莉莉 (女)',
                    bm_daniel: '丹尼尔 (男)',
                    bm_fable: '费布尔 (男)',
                    bm_george: '乔治 (男)',
                    bm_lewis: '刘易斯 (男)'
                }
            }
        }
    });
})();
