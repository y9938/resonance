export type Translate = (
  key: string,
  params?: Record<string, unknown>,
) => string;
export interface LocaleHelpers {
  formatCount(value: number): string;
  formatSttMeta(count: number): string;
  formatSttProcessedTextDuration(seconds: number): string;
  formatSttProcessedTextLabel(duration: string): string;
  formatTtsCharLine(length: number, maxChars: number, t: Translate): string;
  formatTtsMeta(chunks: number, duration: number): string;
  formatTtsInputTooLongMessage(
    limit: number,
    formatCount: (value: number) => string,
    t: Translate,
  ): string;
  formatMicDuration(seconds: number): string | null;
  formatDateTime(date: Date): string;
}
export interface TtsLabels {
  languages: Record<string, string>;
  voiceGroups: Record<string, Record<string, string>>;
}
export interface LocaleDefinition {
  messages: Record<string, string>;
  helpers: Partial<LocaleHelpers>;
  tts: TtsLabels;
}
