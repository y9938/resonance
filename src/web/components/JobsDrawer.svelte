<script lang="ts">
  import {
    history,
    loadHistory,
    groups,
    rowKey,
    filename,
    stateText,
    duration,
  } from "../jobs/state.svelte";
  import { selectJob, openBatch } from "../session";
  import { t, helpers } from "../i18n/state.svelte";
  function infinite(node: HTMLElement) {
    const observer = new IntersectionObserver((entries) => {
      if (
        entries.some((e) => e.isIntersecting) &&
        history.open &&
        history.hasMore &&
        !history.loading
      )
        void loadHistory(true);
    });
    observer.observe(node);
    return { destroy: () => observer.disconnect() };
  }
</script>

<aside
  class="jobs-drawer"
  class:open={history.open}
  id="jobsDrawer"
  aria-hidden={!history.open}
  inert={!history.open}
>
  <div class="jobs-drawer-header">
    <span class="jobs-drawer-title">{t("jobsDrawerTitle")}</span><button
      class="menu-btn"
      id="jobsDrawerClose"
      onclick={() => (history.open = false)}
      aria-label={t("jobsCloseAria")}
      ><svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        ><line x1="18" y1="6" x2="6" y2="18"></line><line
          x1="6"
          y1="6"
          x2="18"
          y2="18"
        ></line></svg
      ></button
    >
  </div>
  <div class="jobs-drawer-scroll" id="jobsDrawerScroll">
    {#if (history.loading && !history.jobs.length) || history.error || !history.jobs.length}<div
        id="jobsListState"
      >
        {t(
          history.loading
            ? "jobsLoading"
            : history.error
              ? "jobsLoadError"
              : "jobsEmpty",
        )}
      </div>{/if}
    <div id="jobsList">
      {#each groups() as group (group.key)}
        {#if group.batch}
          <details style="margin-bottom:0.5rem">
            <summary
              class="btn btn-secondary"
              style="display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;width:100%;text-align:left;padding:0.625rem 0.75rem;cursor:pointer"
            >
              <div>
                <div style="font-weight:600">
                  {t("jobsBatchTitle")} · {t("jobsBatchSummary", {
                    done: group.jobs.filter((j) => j.state === "completed")
                      .length,
                    total: group.jobs.length,
                  })}
                </div>
                <div style="opacity:0.75;font-size:0.8rem">
                  {helpers().formatDateTime(new Date(group.updated * 1000))}
                </div>
              </div>
              <button
                class="jobs-batch-open"
                onclick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  openBatch(group.jobs[0].batch_id!, group.jobs);
                }}>{t("jobsBatchOpen")}</button
              >
            </summary>
            <div style="display:grid;gap:0.35rem;padding:0.45rem 0 0 0.5rem">
              {#each group.jobs as job (rowKey(job))}<button
                  class="jobs-child-row"
                  style="padding:0.52rem 0.6rem"
                  onclick={() =>
                    job.job_id && selectJob(job.job_id, job.job_type)}
                >
                  <div>
                    <div class="jobs-child-file">{filename(job)}</div>
                    <div class="jobs-child-detail">
                      {stateText(job)}{duration(job) > 0
                        ? " · " +
                          helpers().formatSttProcessedTextDuration(
                            duration(job),
                          )
                        : ""}
                    </div>
                  </div>
                  <span class={"jobs-batch-state " + job.state}
                    >{stateText(job)}</span
                  >
                </button>{/each}
            </div>
          </details>
        {:else}
          {@const job = group.jobs[0]}
          <button
            class="btn btn-secondary"
            style="display:block;width:100%;text-align:left;margin-bottom:0.5rem;padding:0.625rem 0.75rem"
            onclick={() => job.job_id && selectJob(job.job_id, job.job_type)}
          >
            <div style="font-weight:600">
              {job.job_type.toUpperCase()} · {stateText(job)}{duration(job) > 0
                ? " · " +
                  helpers().formatSttProcessedTextDuration(duration(job))
                : ""}
            </div>
            <div style="opacity:0.75;font-size:0.8rem">
              {helpers().formatDateTime(new Date(group.updated * 1000))}
            </div>
          </button>
        {/if}
      {/each}
    </div>
    <div id="jobsFooter" class="jobs-footer">
      {#if history.loadingMore}<div id="jobsLoadMoreState">
          {t("jobsLoadingMore")}
        </div>{:else if history.jobs.length && !history.hasMore}<div
          id="jobsListEnd"
        >
          {t("jobsListEnd")}
        </div>{/if}
    </div>
    {#if history.open && history.hasMore && !history.loadingMore}<div
        id="jobsListSentinel"
        class="jobs-list-sentinel"
        aria-hidden="true"
        use:infinite
      ></div>{/if}
  </div>
</aside>
<button
  class="jobs-drawer-backdrop"
  class:open={history.open}
  id="jobsDrawerBackdrop"
  onclick={() => (history.open = false)}
  aria-label={t("jobsCloseAria")}
  tabindex="-1"
></button>
