import { mergeMicPcmChunks, encodeMicWav } from "./pcm";
interface Pending {
  sequence: number;
  blob: Blob;
  durationSec: number;
  attempts: number;
}
/** One ordered upload queue. Retried requests reuse their original sequence. */
export class LiveTransport {
  private chunks: Float32Array[] = [];
  private samples = 0;
  private pending: Pending[] = [];
  private pendingDuration = 0;
  private sequence = 1;
  private queue: Promise<void> | null = null;
  private controller = new AbortController();
  failed = false;
  backpressure = false;
  constructor(
    readonly jobId: string,
    readonly sampleRate: number,
    private onFailure: (error: Error) => void,
    private onBackpressure: () => void,
  ) {}
  append(chunk: Float32Array) {
    if (this.failed || this.backpressure) return;
    this.chunks.push(chunk);
    this.samples += chunk.length;
    if (this.samples >= this.sampleRate) void this.flush();
  }
  flush(): Promise<void> {
    if (this.samples && !this.failed) {
      const samples = mergeMicPcmChunks(this.chunks, this.samples);
      this.chunks = [];
      this.samples = 0;
      const durationSec = samples.length / this.sampleRate;
      this.pending.push({
        sequence: this.sequence++,
        blob: encodeMicWav(samples, this.sampleRate),
        durationSec,
        attempts: 0,
      });
      this.pendingDuration += durationSec;
      if (this.pendingDuration > 10 && !this.backpressure) {
        this.backpressure = true;
        this.onBackpressure();
      }
    }
    if (this.queue) return this.queue;
    this.queue = this.pump().finally(() => {
      this.queue = null;
    });
    return this.queue;
  }
  async drain(): Promise<void> {
    const timeout = setTimeout(() => {
      this.failed = true;
      this.controller.abort();
      this.onFailure(new Error("Live audio drain timed out"));
    }, 30_000);
    try {
      await this.flush();
    } finally {
      clearTimeout(timeout);
    }
  }
  private async send(
    item: Pending,
  ): Promise<{ retry?: boolean; error?: Error }> {
    const body = new FormData();
    body.append("file", item.blob, "chunk.wav");
    let response: Response | undefined;
    const signal = AbortSignal.any([
      this.controller.signal, AbortSignal.timeout(10_000),
    ]);
    try {
      response = await fetch(
        `/api/jobs/live/${encodeURIComponent(this.jobId)}/chunk?sequence=${item.sequence}`,
        { method: "POST", body, signal },
      );
      if (!response.ok)
        return {
          retry: response.status >= 500,
          error: new Error(await response.text()),
        };
      const payload = await response.json();
      if (payload.ack_sequence !== item.sequence)
        throw new Error("Live audio sequence acknowledgement mismatch");
      return {};
    } catch (error) {
      return { retry: !response || signal.aborted, error: error as Error };
    }
  }
  private async pump() {
    while (this.pending.length && !this.failed) {
      const item = this.pending[0];
      let result: { retry?: boolean; error?: Error };
      do {
        const delay = [0, 250, 750, 1500][Math.min(item.attempts, 3)];
        if (delay) await new Promise((r) => setTimeout(r, delay));
        item.attempts++;
        if (this.failed) return;
        result = await this.send(item);
      } while (result.retry && item.attempts < 4 && !this.failed);
      if (this.failed) return;
      if (result.error) {
        this.failed = true;
        this.onFailure(result.error);
        return;
      }
      this.pending.shift();
      this.pendingDuration = Math.max(
        0,
        this.pendingDuration - item.durationSec,
      );
    }
  }
  dispose() {
    this.failed = true;
    this.controller.abort();
  }
}
