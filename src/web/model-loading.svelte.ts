import { request } from "./api/client";

interface ModelStatus {
  id: string;
  name: string;
  loaded: boolean;
  loading: boolean;
}
interface ModelCatalog {
  stt: { models: ModelStatus[] };
  tts_backends: ModelStatus[];
}

export const modelLoading = $state({ models: [] as ModelStatus[], observing: false });
let watchers = 0;
let inFlight = false;
let timer: ReturnType<typeof setInterval> | undefined;

async function refresh() {
  if (!watchers || inFlight) return;
  inFlight = true;
  try {
    const catalog = await request<ModelCatalog>("/models");
    modelLoading.models = [
      ...(catalog.stt?.models || []),
      ...(catalog.tts_backends || []),
    ];
  } catch {
    // Status is informational; operation errors remain the operation's response.
  } finally {
    inFlight = false;
  }
}

export function watchModelLoading() {
  watchers++;
  if (watchers === 1) {
    modelLoading.observing = true;
    void refresh();
    timer = setInterval(() => void refresh(), 300);
  }
  let stopped = false;
  return () => {
    if (stopped) return;
    stopped = true;
    watchers = Math.max(0, watchers - 1);
    if (!watchers) {
      modelLoading.observing = false;
      clearInterval(timer);
      timer = undefined;
    }
  };
}

export function modelIsLoading(id: string | null | undefined) {
  return (
    modelLoading.observing &&
    !!id &&
    modelLoading.models.find((model) => model.id === id)?.loading === true
  );
}

export function loadingModelName(id: string | null | undefined) {
  return id ? modelLoading.models.find((model) => model.id === id)?.name || id : "";
}
