export type JobState =
  "queued" | "running" | "completed" | "failed" | "cancelled";
export type JobType = "stt" | "tts";
export interface Segment {
  start: number;
  end: number;
  text: string;
  source?: string;
  generation?: number;
}
export interface Result {
  filename?: string;
  source?: "mic_live" | "system_audio";
  segments?: Segment[];
  duration?: number;
  download_url?: string;
  chunks?: number;
  batch_id?: string;
  batch_index?: number;
  batch_total?: number;
}
export interface Job extends Result {
  job_id: string | null;
  job_type: JobType;
  state: JobState;
  progress_current: number;
  progress_total: number;
  result?: Result;
  error?: string;
  created_at?: number;
  updated_at?: number;
  started_at?: number;
  last_event_seq?: number;
}
export type JobEvent = (
  | { type: "start"; total: number; duration?: number; stage?: string }
  | {
      type: "progress";
      current: number;
      total: number;
      stage?: string;
      segment?: Segment;
    }
  | {
      type: "transcript_preview";
      source?: string;
      generation: number;
      text: string;
    }
  | {
      type: "complete";
      duration: number;
      chunks?: number;
      download_url?: string;
    }
  | { type: "error"; message: string }
  | { type: "cancelled" }
) & { seq?: number };
export interface Model {
  id: string;
  name: string;
  languages: string[];
}
export interface Voice {
  id: string;
  name?: string;
  backend_id?: string;
}
export interface Language {
  id: string;
  voices: Voice[];
  default_voice_id?: string;
}
export interface Config {
  local_files_enabled?: boolean;
  system_audio_enabled?: boolean;
  upload_limit_mb?: number;
  tts_max_input_chars?: number;
  tts_max_chars?: number;
  tts?: { default_language?: string; languages: Language[] };
}
export interface LocalFile {
  name: string;
  path: string;
}
export type AudioInput = File | LocalFile;
