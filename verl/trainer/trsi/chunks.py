"""Semantic chunking of generated responses (method section 10).

A response is partitioned into contiguous, non-overlapping chunks that cover
every valid generated token:

* complete ``<obj>..</obj><box>..</box>at<t>..</t>s`` structures, ``<t>..</t>s
  to <t>..</t>s`` intervals and single ``<t>..</t>s`` times are atomic and are
  never cut by a chunk boundary;
* inside the reasoning block, chunks follow sentence (and line) boundaries;
  overly long sentences are split only at interpretable clause boundaries,
  and short or formatting-only fragments are attached to adjacent content;
* the final ``<answer>..</answer>`` block is a separate chunk;
* without tags, sentences/clauses are used; when the tag structure is
  inconsistent or the text cannot be aligned to the generated tokens, a
  deterministic contiguous-length fallback is used and marked ungrounded.

Boundaries are derived from the generated *text only*; no divergence,
entropy or probability gap is used, so segmentation and optimization
feedback have different sources.  Character spans are mapped back to the
*original generated token IDs* through exact byte offsets of the byte-level
BPE vocabulary (no re-tokenization).  A token belongs to the chunk that
contains its first byte, which yields a contiguous token partition.

Chunk ids and spans are local to one response; they are never reused across
responses, rounds or teacher versions.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping, Optional, Sequence


__all__ = [
    "AlignedText",
    "AlignmentError",
    "ChunkedResponse",
    "ChunkingConfig",
    "FieldClaim",
    "ROLE_ANSWER",
    "ROLE_FALLBACK",
    "ROLE_OBJECT",
    "ROLE_OBJECT_BOX",
    "ROLE_OBJECT_BOX_TIME",
    "ROLE_STATEMENT",
    "ROLE_TEMPORAL_INTERVAL",
    "ROLE_TEMPORAL_POINT",
    "SemanticChunk",
    "TokenTextAligner",
    "bytes_to_unicode",
    "chunk_response",
    "fallback_length_spans",
    "parse_field_claims",
]


ROLE_ANSWER = "answer"
ROLE_OBJECT_BOX_TIME = "object_box_time"
ROLE_TEMPORAL_INTERVAL = "temporal_interval"
ROLE_OBJECT_BOX = "object_box"
ROLE_TEMPORAL_POINT = "temporal_point"
ROLE_OBJECT = "object_mention"
ROLE_STATEMENT = "statement"
ROLE_FALLBACK = "fallback_span"

# Higher value wins when several claim kinds share one chunk.
_ROLE_PRIORITY = {
    ROLE_STATEMENT: 0,
    ROLE_OBJECT: 1,
    ROLE_TEMPORAL_POINT: 2,
    ROLE_OBJECT_BOX: 3,
    ROLE_TEMPORAL_INTERVAL: 4,
    ROLE_OBJECT_BOX_TIME: 5,
}

_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"
_ANSWER_OPEN, _ANSWER_CLOSE = "<answer>", "</answer>"

# --------------------------------------------------------------------------
# Structured field claims (atomic units)
# --------------------------------------------------------------------------

_CLAIM_RE = re.compile(
    r"(?P<obt><obj>(?P<obt_name>[^<>]{0,200})</obj>\s*"
    r"(?P<obt_boxes>(?:<box>[^<>]{0,400}</box>\s*)+)"
    r"(?:at\s*<t>(?P<obt_time>[^<>]{0,40})</t>\s*s?)?)"
    r"|(?P<iv>(?:(?:from|between)\s+)?<t>(?P<iv_start>[^<>]{0,40})</t>\s*s?\s*"
    r"(?:to|until|till|and|-|–|—|~)\s*<t>(?P<iv_end>[^<>]{0,40})</t>\s*s?)"
    r"|(?P<pt><t>(?P<pt_time>[^<>]{0,40})</t>\s*s?)"
    r"|(?P<bx><box>(?P<bx_boxes>[^<>]{0,400})</box>)"
    r"|(?P<ob><obj>(?P<ob_name>[^<>]{0,200})</obj>)",
    re.IGNORECASE,
)
_BOX_LIST_RE = re.compile(r"\[([^\[\]]*)\]")
_NUMBER_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*s?\s*$", re.IGNORECASE)
# Negation / contrast cues.  Field verification treats a chunk containing any of them as unverifiable
# (a negated or contrasted statement can mention an annotated interval or box without asserting it).
# CJK: \u4e0d \u6ca1 \u65e0 \u7121 \u672a \u975e \u522b \u52ff (bu, mei, wu, wu, wei, fei, bie, wu).
_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|none|nothing|nobody|nowhere|neither|nor|without|cannot|absent|absence|"
    r"lacks?|lacking|lacked|missing|fails?\s+to|failed\s+to|instead\s+of|rather\s+than|hardly|barely)\b"
    r"|n't|[\u4e0d\u6ca1\u65e0\u7121\u672a\u975e\u522b\u52ff]",
    re.IGNORECASE,
)
_RELATION_RE = re.compile(
    r"\b(?:before|after|then|while|during|until|first|finally|next|later|earlier|meanwhile|followed)\b",
    re.IGNORECASE,
)
_FORMAT_TOKEN_RE = re.compile(r"</?(?:think|answer)>|<\|[^|<>]{1,40}\|>", re.IGNORECASE)
# Ideographic full stop and fullwidth exclamation / question marks, comma, semicolon, colon.
_SENTENCE_PUNCT = ".!?\u3002\uff01\uff1f"
_CJK_PUNCT = "\u3002\uff01\uff1f"
_ABBREVIATIONS = frozenset({"e.g", "i.e", "etc", "vs", "approx", "mr", "mrs", "dr", "fig", "cf"})
_CLAUSE_PUNCT = ",;:\uff0c\uff1b\uff1a"
_CONJUNCTION_RE = re.compile(
    r"\s+(?=(?:and|but|then|while|because|so|whereas|after|before|when|which)\b)", re.IGNORECASE
)


def _parse_number(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    match = _NUMBER_RE.match(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if value == value and value not in (float("inf"), float("-inf")) else None


def _parse_boxes(text: Optional[str]) -> list[list[float]]:
    boxes: list[list[float]] = []
    for inner in _BOX_LIST_RE.findall(text or ""):
        parts = [part.strip() for part in inner.split(",")]
        if len(parts) != 4:
            continue
        values = [_parse_number(part) for part in parts]
        if all(value is not None for value in values):
            boxes.append([float(value) for value in values])
    return boxes


@dataclass(frozen=True)
class FieldClaim:
    """One structured, potentially verifiable field expressed in the response text."""

    kind: str
    char_start: int
    char_end: int
    text: str
    object_name: Optional[str] = None
    boxes: tuple[tuple[float, float, float, float], ...] = ()
    time: Optional[float] = None
    interval: Optional[tuple[float, float]] = None

    @property
    def role(self) -> str:
        return {
            "object_box_time": ROLE_OBJECT_BOX_TIME,
            "object_box": ROLE_OBJECT_BOX,
            "interval": ROLE_TEMPORAL_INTERVAL,
            "point": ROLE_TEMPORAL_POINT,
            "box": ROLE_OBJECT_BOX,
            "object": ROLE_OBJECT,
        }[self.kind]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "char_span": [self.char_start, self.char_end],
            "object_name": self.object_name,
            "boxes": [list(box) for box in self.boxes],
            "time": self.time,
            "interval": list(self.interval) if self.interval is not None else None,
        }


def parse_field_claims(text: str, start: int = 0, end: Optional[int] = None) -> list[FieldClaim]:
    """Find atomic structured fields in ``text[start:end]`` (absolute offsets)."""

    end = len(text) if end is None else end
    claims: list[FieldClaim] = []
    for match in _CLAIM_RE.finditer(text, start, end):
        span_start, span_end = match.span()
        raw = match.group(0)
        if match.group("obt") is not None:
            boxes = tuple(tuple(box) for box in _parse_boxes(match.group("obt_boxes")))
            time = _parse_number(match.group("obt_time"))
            kind = "object_box_time" if match.group("obt_time") is not None else "object_box"
            claims.append(
                FieldClaim(
                    kind=kind,
                    char_start=span_start,
                    char_end=span_end,
                    text=raw,
                    object_name=re.sub(r"\s+", " ", match.group("obt_name")).strip() or None,
                    boxes=boxes,
                    time=time,
                )
            )
        elif match.group("iv") is not None:
            first, second = _parse_number(match.group("iv_start")), _parse_number(match.group("iv_end"))
            interval = (first, second) if first is not None and second is not None else None
            claims.append(FieldClaim("interval", span_start, span_end, raw, interval=interval))
        elif match.group("pt") is not None:
            claims.append(FieldClaim("point", span_start, span_end, raw, time=_parse_number(match.group("pt_time"))))
        elif match.group("bx") is not None:
            boxes = tuple(tuple(box) for box in _parse_boxes(match.group("bx_boxes")))
            claims.append(FieldClaim("box", span_start, span_end, raw, boxes=boxes))
        else:
            name = re.sub(r"\s+", " ", match.group("ob_name")).strip() or None
            claims.append(FieldClaim("object", span_start, span_end, raw, object_name=name))
    return claims


# --------------------------------------------------------------------------
# Exact token/text alignment for byte-level BPE vocabularies (Qwen2/2.5/3)
# --------------------------------------------------------------------------


class AlignmentError(ValueError):
    """The generated token IDs cannot be mapped to text offsets exactly."""


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 / Qwen byte-level alphabet: byte value -> printable unicode char."""

    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    code_points = byte_values[:]
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            code_points.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, (chr(point) for point in code_points)))


@dataclass(frozen=True)
class AlignedText:
    text: str
    """Decoded text including special tokens; invalid UTF-8 bytes become U+FFFD."""
    token_char_starts: tuple[int, ...]
    """Character offset of each token's first byte (non-decreasing)."""


def _collapse_replacements(text: str) -> str:
    return re.sub("�+", "�", text)


class TokenTextAligner:
    """Map generated token IDs to exact character offsets of the decoded response.

    Only byte-level BPE vocabularies are supported (the Qwen family used by
    Open-o3-Video).  Added/special tokens are decoded as their literal
    content.  When ``verify_with_decode`` is true, the reconstruction is
    compared with ``tokenizer.decode`` and any difference raises
    :class:`AlignmentError`, which makes the caller fall back to ungrounded
    length chunks instead of silently mis-aligning spans.
    """

    def __init__(self, tokenizer: Any, *, verify_with_decode: bool = True):
        self.tokenizer = tokenizer
        self.verify_with_decode = bool(verify_with_decode)
        self._byte_decoder = {char: byte for byte, char in bytes_to_unicode().items()}
        self._added = self._collect_added_tokens(tokenizer)
        self._cache: dict[int, bytes] = {}

    @staticmethod
    def _collect_added_tokens(tokenizer: Any) -> dict[int, str]:
        result: dict[int, str] = {}
        decoder = getattr(tokenizer, "added_tokens_decoder", None)
        if isinstance(decoder, Mapping):
            for token_id, token in decoder.items():
                result[int(token_id)] = str(getattr(token, "content", token))
        for token_id in list(getattr(tokenizer, "all_special_ids", None) or []):
            token_id = int(token_id)
            if token_id not in result:
                result[token_id] = str(tokenizer.convert_ids_to_tokens(token_id))
        return result

    def token_bytes(self, token_id: int) -> bytes:
        token_id = int(token_id)
        cached = self._cache.get(token_id)
        if cached is not None:
            return cached
        if token_id in self._added:
            value = self._added[token_id].encode("utf-8")
        else:
            piece = self.tokenizer.convert_ids_to_tokens(token_id)
            if not isinstance(piece, str):
                raise AlignmentError(f"Token id {token_id} has no vocabulary string.")
            try:
                value = bytes(self._byte_decoder[char] for char in piece)
            except KeyError as exc:
                raise AlignmentError("The tokenizer vocabulary is not byte-level BPE.") from exc
        self._cache[token_id] = value
        return value

    def align(self, token_ids: Sequence[int]) -> AlignedText:
        pieces = [self.token_bytes(token_id) for token_id in token_ids]
        data = b"".join(pieces)
        escaped = data.decode("utf-8", errors="surrogateescape")
        char_of_byte = [0] * (len(data) + 1)
        position = 0
        for char_index, char in enumerate(escaped):
            code = ord(char)
            width = 1 if 0xDC80 <= code <= 0xDCFF else len(char.encode("utf-8"))
            for offset in range(width):
                char_of_byte[position + offset] = char_index
            position += width
        if position != len(data):
            raise AlignmentError("Byte/character offset reconstruction is inconsistent.")
        char_of_byte[len(data)] = len(escaped)
        text = "".join("�" if 0xDC80 <= ord(char) <= 0xDCFF else char for char in escaped)
        starts = []
        byte_position = 0
        for piece in pieces:
            starts.append(char_of_byte[byte_position])
            byte_position += len(piece)
        if self.verify_with_decode and hasattr(self.tokenizer, "decode"):
            reference = self.tokenizer.decode(
                list(int(token_id) for token_id in token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if _collapse_replacements(reference) != _collapse_replacements(text):
                raise AlignmentError("Byte-level reconstruction differs from tokenizer.decode.")
        return AlignedText(text=text, token_char_starts=tuple(starts))


# --------------------------------------------------------------------------
# Chunk records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkingConfig:
    min_tokens: int = 4
    """Fragments shorter than this attach to adjacent content of the same block."""
    max_tokens: int = 128
    """Longer reasoning chunks are split at interpretable clause boundaries only."""
    fallback_tokens: int = 32
    """Chunk length of the deterministic ungrounded fallback."""

    def __post_init__(self) -> None:
        if int(self.min_tokens) < 1:
            raise ValueError("min_tokens must be >= 1.")
        if int(self.max_tokens) < int(self.min_tokens):
            raise ValueError("max_tokens must be >= min_tokens.")
        if int(self.fallback_tokens) < 1:
            raise ValueError("fallback_tokens must be >= 1.")


@dataclass
class SemanticChunk:
    index: int
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    role: str
    block: str
    grounded: bool
    claims: list[FieldClaim] = field(default_factory=list)
    has_negation: bool = False
    has_relation: bool = False
    status: str = "unknown"
    evidence_refs: list[str] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return self.token_end - self.token_start

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_chunk_id": self.index,
            "token_span": [self.token_start, self.token_end],
            "char_span": [self.char_start, self.char_end],
            "role": self.role,
            "block": self.block,
            "grounded": self.grounded,
            "claims": [claim.to_dict() for claim in self.claims],
            "has_negation": self.has_negation,
            "has_relation": self.has_relation,
            "verification": self.status,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass
class ChunkedResponse:
    text: str
    chunks: list[SemanticChunk]
    parse_mode: str
    num_tokens: int
    answer_text: Optional[str] = None
    claim_split_count: int = 0

    def token_spans(self) -> list[tuple[int, int]]:
        return [(chunk.token_start, chunk.token_end) for chunk in self.chunks]


# --------------------------------------------------------------------------
# Character-level segmentation
# --------------------------------------------------------------------------


@dataclass
class _Segment:
    start: int
    end: int
    block: str
    claims: list[FieldClaim]
    grounded: bool = True
    num_tokens: int = 0

    def merged(self, other: "_Segment") -> "_Segment":
        first, second = (self, other) if self.start <= other.start else (other, self)
        if first.end != second.start:
            raise AssertionError("Only adjacent segments can be merged.")
        return _Segment(
            start=first.start,
            end=second.end,
            block=first.block,
            claims=first.claims + second.claims,
            grounded=first.grounded and second.grounded,
            num_tokens=first.num_tokens + second.num_tokens,
        )


def _is_formatting_only(text: str) -> bool:
    stripped = _FORMAT_TOKEN_RE.sub("", text)
    return not any(char.isalnum() for char in stripped)


def _inside_claim(position: int, claims: Sequence[FieldClaim]) -> bool:
    return any(claim.char_start < position < claim.char_end for claim in claims)


def _sentence_boundaries(text: str, start: int, end: int, claims: Sequence[FieldClaim]) -> list[int]:
    """Boundary offsets strictly inside ``(start, end)``; whitespace starts the next segment."""

    boundaries: list[int] = []
    position = start
    while position < end:
        char = text[position]
        if char == "\n":
            if position > start and not _inside_claim(position, claims):
                boundaries.append(position)
            while position < end and text[position] == "\n":
                position += 1
            continue
        if char in _SENTENCE_PUNCT:
            run_end = position
            while run_end < end and text[run_end] in _SENTENCE_PUNCT:
                run_end += 1
            boundary = run_end
            if boundary < end and not _inside_claim(boundary, claims) and not _inside_claim(position, claims):
                cjk = any(mark in _CJK_PUNCT for mark in text[position:run_end])
                follows_space = text[boundary].isspace()
                if cjk or follows_space:
                    lookahead = boundary
                    while lookahead < end and text[lookahead] in " \t":
                        lookahead += 1
                    next_char = text[lookahead] if lookahead < end else ""
                    word_match = re.search(r"([A-Za-z][A-Za-z.]*)$", text[max(start, position - 8) : position])
                    previous_word = word_match.group(1).lower().rstrip(".") if word_match else ""
                    abbreviation = previous_word in _ABBREVIATIONS and text[position] == "."
                    lowercase_continuation = text[position] == "." and next_char.islower()
                    if cjk or not (abbreviation or lowercase_continuation):
                        boundaries.append(boundary)
            position = run_end
            continue
        position += 1
    return sorted(set(boundary for boundary in boundaries if start < boundary < end))


def _segment_region(text: str, start: int, end: int, block: str, grounded: bool = True) -> list[_Segment]:
    if end <= start:
        return []
    claims = parse_field_claims(text, start, end)
    cuts = [start] + _sentence_boundaries(text, start, end, claims) + [end]
    segments = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        if right <= left:
            continue
        inside = [claim for claim in claims if left <= claim.char_start and claim.char_end <= right]
        segments.append(_Segment(left, right, block, inside, grounded))
    return segments


def _clause_cut_candidates(text: str, segment: _Segment) -> list[int]:
    candidates: set[int] = set()
    for index in range(segment.start, segment.end - 1):
        if text[index] in _CLAUSE_PUNCT and text[index + 1].isspace():
            candidates.add(index + 1)
    for match in _CONJUNCTION_RE.finditer(text, segment.start, segment.end):
        candidates.add(match.start())
    return sorted(
        candidate
        for candidate in candidates
        if segment.start < candidate < segment.end and not _inside_claim(candidate, segment.claims)
    )


def _structure(text: str) -> tuple[str, list[_Segment], Optional[str]]:
    """Return ``(parse_mode, character segments, answer_text)``."""

    counts = {tag: text.count(tag) for tag in (_THINK_OPEN, _THINK_CLOSE, _ANSWER_OPEN, _ANSWER_CLOSE)}
    if any(count > 1 for count in counts.values()):
        return "fallback_malformed", [], None
    think_open = text.find(_THINK_OPEN)
    think_close = text.find(_THINK_CLOSE)
    answer_open = text.find(_ANSWER_OPEN)
    answer_close = text.find(_ANSWER_CLOSE)
    length = len(text)

    if think_open < 0 and think_close < 0 and answer_open < 0 and answer_close < 0:
        return "untagged", _segment_region(text, 0, length, "reasoning"), None

    if think_close >= 0 and (think_open < 0 or think_close < think_open):
        return "fallback_malformed", [], None
    if answer_close >= 0 and (answer_open < 0 or answer_close < answer_open):
        return "fallback_malformed", [], None
    if answer_open >= 0 and think_open >= 0 and (think_close < 0 or answer_open < think_close):
        return "fallback_malformed", [], None

    segments: list[_Segment] = []
    if answer_open < 0:
        # Reasoning only (possibly truncated before the answer was produced).
        mode = "truncated_reasoning" if think_close < 0 else "missing_answer"
        segments.extend(_segment_region(text, 0, length, "reasoning"))
        return mode, segments, None

    reasoning_end = think_close + len(_THINK_CLOSE) if think_close >= 0 else 0
    if think_close >= 0:
        segments.extend(_segment_region(text, 0, reasoning_end, "reasoning"))
        gap = text[reasoning_end:answer_open]
        if gap and not _is_formatting_only(gap):
            segments.extend(_segment_region(text, reasoning_end, answer_open, "reasoning"))
            answer_start = answer_open
        else:
            answer_start = reasoning_end
        mode = "tagged"
    else:
        # An answer block without a reasoning block.
        if answer_open > 0 and not _is_formatting_only(text[:answer_open]):
            segments.extend(_segment_region(text, 0, answer_open, "reasoning"))
            answer_start = answer_open
        else:
            answer_start = 0
        mode = "answer_only"

    if answer_close < 0:
        answer_end = length
        answer_text = text[answer_open + len(_ANSWER_OPEN) :].strip() or None
        claims = parse_field_claims(text, answer_start, answer_end)
        segments.append(_Segment(answer_start, answer_end, "answer", claims, grounded=False))
        return "truncated_answer", segments, answer_text

    answer_end = answer_close + len(_ANSWER_CLOSE)
    answer_text = text[answer_open + len(_ANSWER_OPEN) : answer_close].strip() or None
    trailing = text[answer_end:]
    if trailing and _is_formatting_only(trailing):
        answer_end = length
        trailing_segments: list[_Segment] = []
    else:
        trailing_segments = _segment_region(text, answer_end, length, "trailing")
    claims = parse_field_claims(text, answer_start, answer_end)
    segments.append(_Segment(answer_start, answer_end, "answer", claims, grounded=True))
    segments.extend(trailing_segments)
    return mode, segments, answer_text


# --------------------------------------------------------------------------
# Token mapping, merging and splitting
# --------------------------------------------------------------------------


def _count_tokens(starts: Sequence[int], left: int, right: int) -> int:
    """Number of tokens whose first character lies in ``[left, right)``."""

    return bisect_left(starts, right) - bisect_left(starts, left)


def _merge_short(text: str, segments: list[_Segment], min_tokens: int) -> list[_Segment]:
    if not segments:
        return []
    merged: list[_Segment] = []
    for segment in segments:
        small = segment.num_tokens < min_tokens or _is_formatting_only(text[segment.start : segment.end])
        if merged and small and merged[-1].block == segment.block and segment.block != "answer":
            merged[-1] = merged[-1].merged(segment)
        elif merged and segment.num_tokens == 0:
            # A segment that owns no token start is covered by the previous token.
            merged[-1] = merged[-1].merged(segment)
        else:
            merged.append(segment)
    if len(merged) >= 2:
        first = merged[0]
        first_small = first.num_tokens < min_tokens or _is_formatting_only(text[first.start : first.end])
        if first_small and merged[1].block == first.block and first.block != "answer":
            merged[1] = first.merged(merged[1])
            merged.pop(0)
    return merged


def _split_long(text: str, segment: _Segment, starts: Sequence[int], cfg: ChunkingConfig) -> list[_Segment]:
    """Greedy clause-boundary splitting; a segment without such boundaries stays long."""

    if segment.block == "answer" or segment.num_tokens <= cfg.max_tokens:
        return [segment]

    def admissible(left: int, candidate: int) -> bool:
        return (
            candidate > left
            and _count_tokens(starts, left, candidate) >= cfg.min_tokens
            and _count_tokens(starts, candidate, segment.end) >= cfg.min_tokens
        )

    cuts: list[int] = []
    left = segment.start
    furthest_fit: Optional[int] = None
    for candidate in _clause_cut_candidates(text, segment):
        if _count_tokens(starts, left, segment.end) <= cfg.max_tokens:
            break
        if not admissible(left, candidate):
            continue
        if _count_tokens(starts, left, candidate) <= cfg.max_tokens:
            furthest_fit = candidate
            continue
        # The piece would exceed the limit: cut at the furthest boundary that
        # still fits, or here when no earlier boundary exists.
        cut = furthest_fit if furthest_fit is not None else candidate
        cuts.append(cut)
        left = cut
        furthest_fit = None
        if admissible(left, candidate) and _count_tokens(starts, left, candidate) <= cfg.max_tokens:
            furthest_fit = candidate
    if furthest_fit is not None and _count_tokens(starts, left, segment.end) > cfg.max_tokens:
        cuts.append(furthest_fit)
    pieces: list[_Segment] = []
    bounds = [segment.start] + sorted(set(cuts)) + [segment.end]
    for piece_start, piece_end in zip(bounds[:-1], bounds[1:]):
        if piece_end <= piece_start:
            continue
        claims = [claim for claim in segment.claims if piece_start <= claim.char_start and claim.char_end <= piece_end]
        pieces.append(
            _Segment(
                piece_start,
                piece_end,
                segment.block,
                claims,
                segment.grounded,
                _count_tokens(starts, piece_start, piece_end),
            )
        )
    return pieces or [segment]


def fallback_length_spans(num_tokens: int, chunk_tokens: int, min_tokens: int = 1) -> list[tuple[int, int]]:
    """Deterministic contiguous-length spans covering ``[0, num_tokens)``."""

    if num_tokens <= 0:
        return []
    spans = [(start, min(start + chunk_tokens, num_tokens)) for start in range(0, num_tokens, chunk_tokens)]
    if len(spans) >= 2 and spans[-1][1] - spans[-1][0] < min_tokens:
        last = spans.pop()
        spans[-1] = (spans[-1][0], last[1])
    return spans


def _role_of(segment: _Segment) -> str:
    if segment.block == "answer":
        return ROLE_ANSWER
    role = ROLE_STATEMENT
    for claim in segment.claims:
        if _ROLE_PRIORITY[claim.role] > _ROLE_PRIORITY[role]:
            role = claim.role
    return role


def _fallback_response(text: str, num_tokens: int, starts: Sequence[int], cfg: ChunkingConfig, mode: str) -> ChunkedResponse:
    chunks = []
    for index, (start, end) in enumerate(fallback_length_spans(num_tokens, cfg.fallback_tokens, cfg.min_tokens)):
        char_start = starts[start] if starts else 0
        char_end = starts[end] if starts and end < len(starts) else len(text)
        chunks.append(
            SemanticChunk(
                index=index,
                token_start=start,
                token_end=end,
                char_start=char_start,
                char_end=char_end,
                role=ROLE_FALLBACK,
                block="fallback",
                grounded=False,
            )
        )
    return ChunkedResponse(text=text, chunks=chunks, parse_mode=mode, num_tokens=num_tokens)


def chunk_response(
    token_ids: Sequence[int],
    aligner: TokenTextAligner,
    cfg: Optional[ChunkingConfig] = None,
) -> ChunkedResponse:
    """Partition one generated response (valid tokens only) into semantic chunks."""

    cfg = cfg or ChunkingConfig()
    token_ids = [int(token_id) for token_id in token_ids]
    num_tokens = len(token_ids)
    if num_tokens == 0:
        return ChunkedResponse(text="", chunks=[], parse_mode="empty", num_tokens=0)
    try:
        aligned = aligner.align(token_ids)
    except AlignmentError:
        return _fallback_response("", num_tokens, [], cfg, "fallback_alignment")
    text, starts = aligned.text, list(aligned.token_char_starts)

    mode, segments, answer_text = _structure(text)
    if mode.startswith("fallback"):
        return _fallback_response(text, num_tokens, starts, cfg, mode)

    for segment in segments:
        segment.num_tokens = _count_tokens(starts, segment.start, segment.end)
    # Tokens are owned by the segment containing their first character.  The
    # first segment also owns any token that starts before its first character.
    if segments:
        segments[0].start = 0
        segments[0].num_tokens = _count_tokens(starts, 0, segments[0].end)
        segments[-1].end = max(segments[-1].end, len(text))
    segments = _merge_short(text, segments, cfg.min_tokens)
    expanded: list[_Segment] = []
    for segment in segments:
        expanded.extend(_split_long(text, segment, starts, cfg))
    segments = [segment for segment in expanded if segment.num_tokens > 0] if expanded else []
    if not segments:
        return _fallback_response(text, num_tokens, starts, cfg, "fallback_empty")

    segment_starts = [segment.start for segment in segments]
    owners = [max(bisect_right(segment_starts, start) - 1, 0) for start in starts]
    chunks: list[SemanticChunk] = []
    token_start = 0
    for owner_index, segment in enumerate(segments):
        token_end = token_start
        while token_end < num_tokens and owners[token_end] == owner_index:
            token_end += 1
        if token_end == token_start:
            continue
        span_text = text[segment.start : segment.end]
        chunks.append(
            SemanticChunk(
                index=len(chunks),
                token_start=token_start,
                token_end=token_end,
                char_start=segment.start,
                char_end=segment.end,
                role=_role_of(segment),
                block=segment.block,
                grounded=segment.grounded,
                claims=list(segment.claims),
                has_negation=bool(_NEGATION_RE.search(span_text)),
                has_relation=bool(_RELATION_RE.search(span_text)),
            )
        )
        token_start = token_end
    if token_start != num_tokens or not chunks:
        return _fallback_response(text, num_tokens, starts, cfg, "fallback_partition")

    # Atomic claims must not be split at token level either.
    claim_splits = 0
    for chunk in chunks:
        for claim in chunk.claims:
            first = bisect_right(starts, claim.char_start) - 1
            last = bisect_right(starts, max(claim.char_end - 1, claim.char_start)) - 1
            if first < chunk.token_start or last >= chunk.token_end:
                claim_splits += 1
    return ChunkedResponse(
        text=text,
        chunks=chunks,
        parse_mode=mode,
        num_tokens=num_tokens,
        answer_text=answer_text,
        claim_split_count=claim_splits,
    )
