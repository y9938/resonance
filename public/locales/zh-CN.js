(function () {
    if (typeof window.__registerResonanceLocale !== 'function') return;

    window.__registerResonanceLocale('zh-CN', {
        messages: {
            pageTitle: 'Resonance',
            metaDescription: 'Resonance — 语音识别与语音合成',
            ariaLangGroup: '界面语言',
            tabStt: '语音 → 文本',
            tabTts: '文本 → 语音',
            sttDropzoneText: '拖拽音频文件或点击选择',
            configLoading: '正在加载设置…',
            btnStop: '停止',
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
            hintSttMedia: '任意文件 — {limit}',
            hintLimitMb: '最大 {mb} MB',
            hintAnySize: '不限大小',
            progressUploading: '上传中…',
            progressProcessing: '处理中…',
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
            jobsOpenAria: '打开任务列表',
            jobsCloseAria: '关闭任务列表',
            jobsDrawerTitle: '任务',
            jobsEmpty: '暂无任务。',
            jobsLoading: '正在加载任务…',
            jobsLoadError: '加载任务列表失败。',
            jobsTypeStt: '语音 → 文本',
            jobsTypeTts: '文本 → 语音',
            jobsStateQueued: '排队中',
            jobsStateRunning: '处理中',
            jobsStateCompleted: '已完成',
            jobsStateFailed: '失败',
            jobsStateCancelled: '已取消',
            jobsLoadingMore: '加载更多…',
            jobsListEnd: '已显示全部任务。'
        },
        helpers: {
            formatCount: function (value) {
                return Number(value || 0).toLocaleString('zh-CN');
            },
            formatSttMeta: function (n) {
                return n + ' 段';
            },
            formatSttProcessedTextDuration: function (seconds) {
                return seconds.toFixed(1) + '秒';
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
        speakerNames: {
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
        }
    });
})();
