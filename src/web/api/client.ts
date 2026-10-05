import type { Job, JobType } from "./types";

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch("/api" + path, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      typeof body?.detail === "string"
        ? body.detail
        : `HTTP ${response.status}`,
    );
  }
  return response.json();
}
export async function status(
  id: string,
  type: JobType,
  retries = false,
): Promise<Job | null> {
  for (const delay of retries ? [0, 250, 750] : [0]) {
    if (delay) await new Promise((resolve) => setTimeout(resolve, delay));
    try {
      const job = await request<Job>("/jobs/" + encodeURIComponent(id));
      if (job.job_type === type) return job;
    } catch {
      /* A bounded retry restores live jobs after transient reload failures. */
    }
  }
  return null;
}
export function download(url: string, filename: string) {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
}
export function downloadText(text: string, filename: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
  download(url, filename);
  URL.revokeObjectURL(url);
}
