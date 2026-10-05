import type { Job } from "../api/types";
import { request } from "../api/client";
import { t, helpers } from "../i18n/state.svelte";
import { toast } from "../settings.svelte";
export const history = $state({
  open: false,
  jobs: [] as Job[],
  loading: false,
  loadingMore: false,
  hasMore: false,
  offset: 0,
  error: false,
});
let generation = 0;
export function filename(job: Job) {
  return job.filename || job.result?.filename || t("defaultTranscriptionFile");
}
export function duration(job: Job) {
  const seconds = job.duration || job.result?.duration;
  if (seconds && seconds > 0) return seconds;
  return job.job_type === "stt"
    ? Math.max(0, ...(job.result?.segments || []).map((s) => s.end))
    : 0;
}
export function stateText(job: Job) {
  return t("jobsState" + job.state[0].toUpperCase() + job.state.slice(1));
}
export function detail(job: Job) {
  if (job.state === "queued") return t("jobsStateQueued");
  if (job.state === "running")
    return job.progress_total > 0
      ? Math.round(
          Math.max(
            0,
            Math.min(100, (job.progress_current / job.progress_total) * 100),
          ),
        ) + "%"
      : t("progressProcessing");
  const seconds = duration(job);
  return (
    stateText(job) +
    (seconds > 0
      ? " · " + helpers().formatSttProcessedTextDuration(seconds)
      : "")
  );
}
export function rowKey(job: Job) {
  return job.batch_id && job.batch_index != null
    ? `batch:${job.batch_id}:${job.batch_index}`
    : `job:${job.job_id}`;
}
export function sorted(jobs: Job[]) {
  return [...jobs].sort(
    (a, b) =>
      (a.batch_index || 0) - (b.batch_index || 0) ||
      (a.created_at || 0) - (b.created_at || 0),
  );
}
export async function loadHistory(more = false) {
  if (more && (history.loadingMore || !history.hasMore)) return;
  const token = more ? generation : ++generation;
  if (more) history.loadingMore = true;
  else history.loading = true;
  history.error = false;
  try {
    const payload = await request<{
      jobs: Job[];
      has_more: boolean;
      next_offset: number;
    }>(`/jobs?limit=60&offset=${more ? history.offset : 0}`);
    if (token !== generation) return;
    history.jobs = more
      ? [
          ...history.jobs,
          ...payload.jobs.filter(
            (j) => !history.jobs.some((old) => old.job_id === j.job_id),
          ),
        ]
      : payload.jobs;
    history.hasMore = payload.has_more;
    history.offset = payload.next_offset;
  } catch {
    if (token === generation) {
      history.error = true;
      toast(t("errNetwork"));
    }
  } finally {
    if (token === generation) {
      history.loading = false;
      history.loadingMore = false;
    }
  }
}
export function refreshHistory() {
  if (history.open) void loadHistory();
}
export function groups() {
  const result: {
    key: string;
    jobs: Job[];
    batch: boolean;
    updated: number;
  }[] = [];
  for (const job of history.jobs) {
    const key =
      job.job_type === "stt" && job.batch_id
        ? `batch:${job.batch_id}`
        : `job:${job.job_id}`;
    const group = result.find((g) => g.key === key);
    if (group) {
      group.jobs.push(job);
      group.updated = Math.max(group.updated, job.updated_at || 0);
    } else
      result.push({
        key,
        jobs: [job],
        batch: key.startsWith("batch:"),
        updated: job.updated_at || 0,
      });
  }
  return result
    .sort((a, b) => b.updated - a.updated)
    .map((g) => ({ ...g, jobs: sorted(g.jobs) }));
}
