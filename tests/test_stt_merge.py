"""Keep STT display helpers in sync with public/index.html."""


def merge_two_stt_strings(a: str, b: str) -> str:
    a_trim = (a or "").strip()
    b_trim = (b or "").strip()
    if not b_trim:
        return a_trim
    if not a_trim:
        return b_trim
    max_k = min(len(a_trim), len(b_trim), 1200)
    for k in range(max_k, 0, -1):
        if a_trim[-k:] == b_trim[:k]:
            return a_trim + b_trim[k:]
    wa = a_trim.split()
    wb = b_trim.split()
    max_w = min(len(wa), len(wb), 48)
    for kw in range(max_w, 0, -1):
        if wa[-kw:] == wb[:kw]:
            return " ".join(wa + wb[kw:])
    return a_trim + " " + b_trim


def merge_adjacent_stt_texts(segments: list[dict]) -> str:
    if not segments:
        return ""
    out = str(segments[0].get("text") or "").strip()
    for i in range(1, len(segments)):
        nxt = str(segments[i].get("text") or "").strip()
        out = merge_two_stt_strings(out, nxt)
    return out


def merge_stt_time_ranges(segments: list[dict]) -> list[dict]:
    ranges = sorted(
        (
            {
                "start": min(float(segment["start"]), float(segment["end"])),
                "end": max(float(segment["start"]), float(segment["end"])),
            }
            for segment in segments
            if segment is not None
            and isinstance(segment.get("start"), (int, float))
            and isinstance(segment.get("end"), (int, float))
        ),
        key=lambda item: (item["start"], item["end"]),
    )
    if not ranges:
        return []
    out = [ranges[0]]
    for nxt in ranges[1:]:
        last = out[-1]
        if nxt["start"] <= last["end"]:
            last["end"] = max(last["end"], nxt["end"])
        else:
            out.append(nxt)
    return out


def processed_stt_text_duration(segments: list[dict]) -> float:
    ranges = merge_stt_time_ranges(segments)
    return ranges[-1]["end"] if ranges else 0.0


def build_stt_blocks(segments: list[dict], block_sec: float = 30.0) -> list[dict]:
    if block_sec <= 0:
        raise ValueError("block_sec must be positive")
    total_duration = processed_stt_text_duration(segments)
    if total_duration <= 0:
        return []

    blocks: dict[int, dict] = {}
    transcript = ""
    for segment in segments:
        segment_text = str(segment.get("text") or "").strip()
        if not segment_text:
            continue
        merged = merge_two_stt_strings(transcript, segment_text)
        delta = merged[len(transcript) :].strip() if merged.startswith(transcript) else ""
        transcript = merged
        if not delta:
            continue
        midpoint = (float(segment["start"]) + float(segment["end"])) / 2.0
        index = int(midpoint // block_sec)
        block = blocks.setdefault(
            index,
            {
                "start": index * block_sec,
                "end": min((index + 1) * block_sec, total_duration),
                "text": "",
            },
        )
        block["text"] = (block["text"] + " " + delta).strip()

    return [block for _, block in sorted(blocks.items()) if block["text"]]


def format_stt_block_time(start_sec: float, end_sec: float) -> str:
    def format_part(total_sec: float) -> str:
        safe = max(0, int(total_sec))
        hours = safe // 3600
        minutes = (safe % 3600) // 60
        seconds = safe % 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    return f"{format_part(start_sec)}-{format_part(end_sec)}"


def build_stt_blocks_text(segments: list[dict], block_sec: float = 30.0) -> str:
    blocks = build_stt_blocks(segments, block_sec=block_sec)
    return "\n\n".join(
        f"[{format_stt_block_time(block['start'], block['end'])}] {block['text']}"
        for block in blocks
    )


def get_stt_copy_text(segments: list[dict], mode: str) -> str:
    if mode == "blocks":
        return build_stt_blocks_text(segments)
    return merge_adjacent_stt_texts(segments)


def test_merge_overlap_suffix_prefix() -> None:
    a = "hello world"
    b = "world foo"
    assert merge_two_stt_strings(a, b) == "hello world foo"


def test_merge_overlap_words() -> None:
    a = "one two three"
    b = "two three four"
    assert merge_two_stt_strings(a, b) == "one two three four"


def test_merge_no_overlap_joins_with_space() -> None:
    assert merge_two_stt_strings("aaa", "bbb") == "aaa bbb"


def test_merge_three_segments() -> None:
    segs = [
        {"text": "start middle"},
        {"text": "middle end"},
        {"text": "end tail"},
    ]
    assert merge_adjacent_stt_texts(segs) == "start middle end tail"


def test_merge_stt_time_ranges_joins_overlaps() -> None:
    segs = [
        {"start": 0.0, "end": 20.0, "text": "a"},
        {"start": 18.0, "end": 26.0, "text": "b"},
    ]
    assert merge_stt_time_ranges(segs) == [{"start": 0.0, "end": 26.0}]


def test_merge_stt_time_ranges_keeps_gaps() -> None:
    segs = [
        {"start": 5.0, "end": 8.0, "text": "b"},
        {"start": 0.0, "end": 2.0, "text": "a"},
        {"start": 9.0, "end": 11.0, "text": "c"},
    ]
    assert merge_stt_time_ranges(segs) == [
        {"start": 0.0, "end": 2.0},
        {"start": 5.0, "end": 8.0},
        {"start": 9.0, "end": 11.0},
    ]


def test_processed_stt_text_duration_returns_last_end() -> None:
    segs = [
        {"start": 0.0, "end": 20.0, "text": "a"},
        {"start": 18.0, "end": 26.0, "text": "b"},
    ]
    assert processed_stt_text_duration(segs) == 26.0


def test_build_stt_blocks_groups_new_text_into_fixed_intervals() -> None:
    segs = [
        {"start": 0.0, "end": 20.0, "text": "start middle"},
        {"start": 18.0, "end": 26.0, "text": "middle end"},
        {"start": 44.0, "end": 56.0, "text": "end tail final"},
    ]
    assert build_stt_blocks(segs) == [
        {"start": 0.0, "end": 30.0, "text": "start middle end"},
        {"start": 30.0, "end": 56.0, "text": "tail final"},
    ]


def test_format_stt_block_time_uses_mm_ss() -> None:
    assert format_stt_block_time(30.0, 56.0) == "00:30-00:56"


def test_get_stt_copy_text_uses_visible_blocks_in_blocks_mode() -> None:
    segs = [
        {"start": 0.0, "end": 20.0, "text": "start middle"},
        {"start": 18.0, "end": 26.0, "text": "middle end"},
    ]
    assert get_stt_copy_text(segs, "blocks") == "[00:00-00:26] start middle end"


def test_get_stt_copy_text_uses_continuous_text_in_continuous_mode() -> None:
    segs = [
        {"start": 0.0, "end": 20.0, "text": "start middle"},
        {"start": 18.0, "end": 26.0, "text": "middle end"},
    ]
    assert get_stt_copy_text(segs, "continuous") == "start middle end"
