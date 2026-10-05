<script lang="ts">
  import {
    stt,
    restore,
    cancelBatch,
    downloadBatch,
  } from "../stt/state.svelte";
  import { rowKey, filename, detail } from "../jobs/state.svelte";
  import { t } from "../i18n/state.svelte";
  const ready = $derived(
    stt.jobs.filter((j) => j.state === "completed" && j.job_id),
  );
  const selected = $derived(stt.jobs.find((j) => j.job_id === stt.selectedId));
  const active = $derived(
    stt.jobs.some((j) => j.state === "queued" || j.state === "running"),
  );
  function next() {
    const index = ready.findIndex((j) => j.job_id === stt.selectedId);
    if (ready.length) void restore(ready[(index + 1) % ready.length].job_id!);
  }
</script>

<div
  class="stt-batch-panel"
  class:active={stt.jobs.length > 0}
  id="sttBatchPanel"
  aria-live="polite"
>
  <div class="stt-batch-header">
    <div>
      <div class="stt-batch-title">{t("sttBatchTitle")}</div>
      <div class="stt-batch-summary" id="sttBatchSummary">
        {t("sttBatchSummary", { done: ready.length, total: stt.jobs.length })}
      </div>
    </div>
    <div class="stt-batch-actions">
      <button
        class="btn btn-secondary"
        id="sttBatchNext"
        hidden={!ready.length}
        onclick={next}>{t("sttBatchNextReady")}</button
      >
      <button
        class="btn btn-secondary"
        id="sttBatchDownloadAll"
        hidden={!ready.length}
        onclick={downloadBatch}>{t("sttBatchDownloadAll")}</button
      >
      <button
        class="btn btn-secondary"
        id="sttBatchCancelCurrent"
        hidden={!selected || !["queued", "running"].includes(selected.state)}
        onclick={() => cancelBatch(true)}>{t("sttBatchCancelCurrent")}</button
      >
      <button
        class="btn btn-secondary"
        id="sttBatchCancel"
        hidden={!active}
        onclick={() => cancelBatch()}>{t("sttBatchCancel")}</button
      >
    </div>
  </div>
  <div class="stt-batch-list" id="sttBatchList">
    {#each stt.jobs as job (rowKey(job))}
      <button
        type="button"
        class="stt-batch-row"
        class:selected={!!job.job_id && job.job_id === stt.selectedId}
        disabled={!job.job_id}
        aria-pressed={!!job.job_id && job.job_id === stt.selectedId}
        onclick={() => job.job_id && restore(job.job_id)}
      >
        <div>
          <div class="stt-batch-file">{filename(job)}</div>
          <div class="stt-batch-detail">{detail(job)}</div>
        </div>
      </button>
    {/each}
  </div>
</div>
