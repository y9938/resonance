import type { Segment } from "../api/types";
import { t, helpers } from "../i18n/state.svelte";
export function formatSttBlockTime(startSec: number, endSec: number) {
  const formatPart = (totalSec: number) => {
    const safe = Math.max(0, Math.floor(totalSec));
    const hours = Math.floor(safe / 3600);
    const minutes = Math.floor((safe % 3600) / 60);
    const seconds = safe % 60;
    if (hours > 0) {
      return (
        String(hours) +
        ":" +
        String(minutes).padStart(2, "0") +
        ":" +
        String(seconds).padStart(2, "0")
      );
    }
    return (
      String(minutes).padStart(2, "0") + ":" + String(seconds).padStart(2, "0")
    );
  };
  return formatPart(startSec) + "-" + formatPart(endSec);
}

export function formatSttProcessedTextDuration(seconds: number) {
  return helpers().formatSttProcessedTextDuration(seconds);
}

export function formatSttProcessedTextLabel(durationText: string) {
  return helpers().formatSttProcessedTextLabel(durationText);
}

/** Merge overlapping chunk transcripts for display; keep in sync with tests/test_stt_merge.py */
export function mergeTwoSttStrings(a: string, b: string) {
  const aTrim = (a || "").trim();
  const bTrim = (b || "").trim();
  if (!bTrim) return aTrim;
  if (!aTrim) return bTrim;

  const tagPattern = /^\[(Speaker \d+|SOURCE:[A-Z0-9_-]+)\]:\s*/i;
  const bTagMatch = bTrim.match(tagPattern);
  const bTag = bTagMatch ? bTagMatch[1].toUpperCase() : null;
  const bClean = bTagMatch ? bTrim.slice(bTagMatch[0].length).trim() : bTrim;

  const globalTagPattern = /\[(Speaker \d+|SOURCE:[A-Z0-9_-]+)\]:\s*/gi;
  const aMatches = Array.from(aTrim.matchAll(globalTagPattern));
  const aLastTag = aMatches.length
    ? aMatches[aMatches.length - 1][1].toUpperCase()
    : null;

  if (bTag === aLastTag) {
    const maxK = Math.min(aTrim.length, bClean.length, 1200);
    for (let k = maxK; k >= 1; k--) {
      if (aTrim.slice(-k) === bClean.slice(0, k)) {
        return aTrim + bClean.slice(k);
      }
    }
    const wa = aTrim.split(/\s+/).filter(Boolean);
    const wbClean = bClean.split(/\s+/).filter(Boolean);
    const maxW = Math.min(wa.length, wbClean.length, 48);
    for (let kw = maxW; kw >= 1; kw--) {
      let match = true;
      for (let j = 0; j < kw; j++) {
        if (wa[wa.length - kw + j] !== wbClean[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        return wa.concat(wbClean.slice(kw)).join(" ");
      }
    }
    return aTrim + " " + bClean;
  }

  return aTrim + "\n" + bTrim;
}

export function mergeAdjacentSttTexts(segments: Segment[]) {
  if (!segments.length) return "";
  let out = String(segments[0].text != null ? segments[0].text : "").trim();
  for (let i = 1; i < segments.length; i++) {
    const next = String(
      segments[i].text != null ? segments[i].text : "",
    ).trim();
    out = mergeTwoSttStrings(out, next);
  }
  return out;
}

/** Merge overlapping/adjacent segment coverage for display; keep in sync with tests/test_stt_merge.py */
export function mergeSttTimeRanges(segments: Segment[]) {
  const ranges = segments
    .filter(
      (segment) =>
        segment &&
        Number.isFinite(segment.start) &&
        Number.isFinite(segment.end),
    )
    .map((segment) => ({
      start: Math.min(segment.start, segment.end),
      end: Math.max(segment.start, segment.end),
    }))
    .sort((a, b) => a.start - b.start || a.end - b.end);
  if (!ranges.length) return [];
  const out = [ranges[0]];
  for (let i = 1; i < ranges.length; i++) {
    const next = ranges[i];
    const last = out[out.length - 1];
    if (next.start <= last.end) {
      last.end = Math.max(last.end, next.end);
      continue;
    }
    out.push(next);
  }
  return out;
}

export function buildSttTimeRangesText(segments: Segment[]) {
  const ranges = mergeSttTimeRanges(segments);
  if (!ranges.length) return "";
  const durationSec = ranges[ranges.length - 1].end;
  return formatSttProcessedTextLabel(
    formatSttProcessedTextDuration(durationSec),
  );
}

export function buildSttBlocks(segments: Segment[], blockSec = 30) {
  if (!(blockSec > 0)) {
    throw new Error("blockSec must be positive");
  }
  const totalDuration = processedSttTextDuration(segments);
  if (!(totalDuration > 0)) return [];

  const blocks = new Map();
  let transcript = "";
  for (const segment of segments) {
    const segmentText = String(
      segment && segment.text != null ? segment.text : "",
    ).trim();
    if (!segmentText) continue;

    const globalTagPattern = /\[(Speaker \d+|SOURCE:[A-Z0-9_-]+)\]:\s*/gi;
    const aMatches = Array.from(transcript.matchAll(globalTagPattern));
    const activeTagHeader = aMatches.length
      ? aMatches[aMatches.length - 1][0]
      : "";

    const merged = mergeTwoSttStrings(transcript, segmentText);
    const delta = merged.startsWith(transcript)
      ? merged.slice(transcript.length).trim()
      : "";
    transcript = merged;
    if (!delta) continue;
    const midpoint = (Number(segment.start) + Number(segment.end)) / 2;
    const index = Math.floor(midpoint / blockSec);
    if (!blocks.has(index)) {
      const start = index * blockSec;
      blocks.set(index, {
        start,
        end: Math.min((index + 1) * blockSec, totalDuration),
        text: "",
      });
    }
    const block = blocks.get(index);
    if (block.text) {
      if (delta.startsWith("[")) {
        block.text = block.text + "\n" + delta;
      } else {
        block.text = (block.text + " " + delta).trim();
      }
    } else {
      if (!delta.startsWith("[") && activeTagHeader) {
        block.text = activeTagHeader + delta;
      } else {
        block.text = delta;
      }
    }
  }

  return Array.from(blocks.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([, block]) => block)
    .filter((block) => block.text);
}

export function buildSttBlocksText(segments: Segment[]) {
  return buildSttBlocks(segments)
    .map(
      (block) =>
        "[" + formatSttBlockTime(block.start, block.end) + "] " + block.text,
    )
    .join("\n\n");
}

export function formatLocalizedSpeakerTags(text: string) {
  if (!text) return "";
  return text
    .replace(/\[SOURCE:MIC\]:\s*/gi, "[" + t("speakerMic") + "]: ")
    .replace(
      /\[SOURCE:SYS\]:\s*/gi,
      "[" + (t("speakerSys") || "🔊 System") + "]: ",
    );
}

export function processedSttTextDuration(segments: Segment[]) {
  const ranges = mergeSttTimeRanges(segments);
  return ranges.length ? ranges[ranges.length - 1].end : 0;
}
