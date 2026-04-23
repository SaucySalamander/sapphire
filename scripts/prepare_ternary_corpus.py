#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import gzip
import io
import json
import os
import random
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
TRUST_REMOTE_CODE_DATASETS = frozenset({"EleutherAI/proof-pile-2"})
PROOF_PILE_2_DATASET = "EleutherAI/proof-pile-2"
THE_STACK_DATASET = "bigcode/the-stack-v2"
THE_STACK_S3_PREFIX = "s3://softwareheritage/content"
THE_STACK_HTTPS_PREFIX = "https://softwareheritage.s3.amazonaws.com/content"
AWS_ENV_VARS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
TEXT_FIELD_CANDIDATES = ("text", "content", "body", "document", "article", "markdown")
DEFAULT_STACK_CACHE_DIR = REPO_ROOT / ".cache" / "the-stack-v2"
DEFAULT_SOURCE_CACHE_DIR = REPO_ROOT / ".cache" / "corpus-sources"
STACK_SAMPLE_MULTIPLIER = 16
STACK_SAMPLE_MIN_CANDIDATES = 4096
STACK_RANDOM_SEED = 1337
STACK_PUBLIC_FETCH_RETRIES = 4
STACK_PUBLIC_FETCH_TIMEOUT = 60
STACK_PUBLIC_FETCH_BACKOFF_SECONDS = 1.5

_STACK_CONTENT_FETCHER: Callable[[str, str], str] | None = None
_HF_REPO_FILE_CACHE: dict[str, list[str]] = {}


@dataclass(frozen=True)
class DatasetUnit:
    dataset_id: str
    config_hints: tuple[str, ...]
    split: str
    max_records: int


@dataclass(frozen=True)
class ExportSpec:
    name: str
    output_relpath: str
    units: tuple[DatasetUnit, ...]
    builder_name: str
    description: str


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _cache_token(value: str) -> str:
    lowered = value.strip().lower()
    lowered = lowered.replace("++", "pp")
    lowered = lowered.replace("#", "sharp")
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "default"


def _config_match_tokens(value: str) -> set[str]:
    lowered = value.strip().lower()
    tokens = {
        lowered,
        _normalize_token(lowered),
        _cache_token(lowered),
    }
    if "++" in lowered:
        tokens.add(lowered.replace("++", "pp"))
        tokens.add(lowered.replace("++", "plusplus"))
        tokens.add(_normalize_token(lowered.replace("++", "pp")))
        tokens.add(_normalize_token(lowered.replace("++", "plusplus")))
    if "#" in lowered:
        tokens.add(lowered.replace("#", "sharp"))
        tokens.add(_normalize_token(lowered.replace("#", "sharp")))
    return {token for token in tokens if token}


def _resolve_proof_pile_2_config(hints: tuple[str, ...]) -> str:
    config_aliases = {
        "arxiv": {"arxiv"},
        "open-web-math": {"open-web-math", "openwebmath", "open_web_math", "math"},
        "algebraic-stack": {"algebraic-stack", "algebraicstack", "algebraic_stack"},
    }
    for canonical, aliases in config_aliases.items():
        alias_tokens = set()
        for alias in aliases:
            alias_tokens.update(_config_match_tokens(alias))
        for hint in hints:
            if _config_match_tokens(hint) & alias_tokens:
                return canonical
    raise RuntimeError(
        "Unable to match config for dataset 'EleutherAI/proof-pile-2' using hints "
        f"{hints}. Supported configs: {', '.join(config_aliases.keys())}"
    )


def _first_text(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        if isinstance(value, list):
            parts = [item.strip() for item in value if isinstance(item, str) and item.strip()]
            if parts:
                return "\n".join(parts)
    return None


def _build_generic_text(record: dict[str, Any]) -> str | None:
    return _first_text(record, TEXT_FIELD_CANDIDATES)


def _build_code_text(record: dict[str, Any]) -> str | None:
    return _first_text(record, ("content", "code", "text", "body"))


def _has_stack_aws_credentials() -> bool:
    return all(os.environ.get(name) for name in AWS_ENV_VARS)


def _fetch_stack_content_public(blob_id: str, src_encoding: str) -> str:
    encoding = src_encoding or "utf-8"
    url = f"{THE_STACK_HTTPS_PREFIX}/{blob_id}"
    last_error: Exception | None = None

    for attempt in range(STACK_PUBLIC_FETCH_RETRIES):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "sapphire/1.0"})
            with urllib.request.urlopen(request, timeout=STACK_PUBLIC_FETCH_TIMEOUT) as response:
                payload = response.read()
            if payload.startswith(b"\x1f\x8b"):
                payload = gzip.decompress(payload)
            return payload.decode(encoding, errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt + 1 >= STACK_PUBLIC_FETCH_RETRIES:
                raise
            last_error = exc
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, socket.timeout, OSError) as exc:
            retryable = True
            if isinstance(exc, OSError) and exc.errno not in (
                None,
                errno.ECONNRESET,
                errno.ETIMEDOUT,
                errno.EHOSTUNREACH,
                errno.ENETUNREACH,
            ):
                retryable = False
            if not retryable or attempt + 1 >= STACK_PUBLIC_FETCH_RETRIES:
                raise
            last_error = exc

        time.sleep(STACK_PUBLIC_FETCH_BACKOFF_SECONDS * (attempt + 1))

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Public Software Heritage fetch exhausted retries for blob '{blob_id}'.")


def _get_stack_content_fetcher() -> Callable[[str, str], str]:
    global _STACK_CONTENT_FETCHER

    if _STACK_CONTENT_FETCHER is not None:
        return _STACK_CONTENT_FETCHER

    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config
        from smart_open import open as smart_open
    except ImportError as exc:
        _STACK_CONTENT_FETCHER = _fetch_stack_content_public
        return _STACK_CONTENT_FETCHER

    if _has_stack_aws_credentials():
        session_kwargs: dict[str, str] = {
            "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
            "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        }
        if os.environ.get("AWS_SESSION_TOKEN"):
            session_kwargs["aws_session_token"] = os.environ["AWS_SESSION_TOKEN"]

        session = boto3.Session(**session_kwargs)
        s3_client = session.client("s3")
    else:
        session = boto3.Session()
        s3_client = session.client("s3", config=Config(signature_version=UNSIGNED))

    def _fetch(blob_id: str, src_encoding: str) -> str:
        encoding = src_encoding or "utf-8"
        s3_url = f"{THE_STACK_S3_PREFIX}/{blob_id}"
        try:
            with smart_open(s3_url, "rb", compression=".gz", transport_params={"client": s3_client}) as fin:
                return fin.read().decode(encoding, errors="replace")
        except Exception:
            return _fetch_stack_content_public(blob_id, encoding)

    _STACK_CONTENT_FETCHER = _fetch
    return _STACK_CONTENT_FETCHER


def _build_stack_code_text(record: dict[str, Any]) -> str | None:
    global _STACK_CONTENT_FETCHER

    direct_text = _build_code_text(record)
    if direct_text:
        return direct_text

    blob_id = record.get("blob_id")
    if not isinstance(blob_id, str) or not blob_id.strip():
        return None

    src_encoding = record.get("src_encoding")
    fetch_content = _get_stack_content_fetcher()
    blob_id = blob_id.strip()
    encoding = src_encoding if isinstance(src_encoding, str) else "utf-8"

    try:
        return fetch_content(blob_id, encoding)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404) and fetch_content is _fetch_stack_content_public and _has_stack_aws_credentials():
            _STACK_CONTENT_FETCHER = None
            return _get_stack_content_fetcher()(blob_id, encoding)
        raise RuntimeError(
            f"The Stack v2 public content fetch failed for blob '{blob_id}' with HTTP {exc.code}. "
            "If public access is unavailable for this object, configure Software Heritage AWS credentials."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"The Stack v2 public content fetch failed for blob '{blob_id}': {exc.reason}"
        ) from exc


def _build_gsm8k_prompt(record: dict[str, Any]) -> str | None:
    question = _first_text(record, ("question",))
    if not question:
        return None
    return f"Solve the following math word problem.\n\n{question}"


def _build_humaneval_prompt(record: dict[str, Any]) -> str | None:
    prompt = _first_text(record, ("prompt", "text"))
    if not prompt:
        return None
    return f"Complete the following Python function.\n\n{prompt}"


BUILDERS: dict[str, Callable[[dict[str, Any]], str | None]] = {
    "generic": _build_generic_text,
    "code": _build_code_text,
    "stack_code": _build_stack_code_text,
    "gsm8k": _build_gsm8k_prompt,
    "humaneval": _build_humaneval_prompt,
}


CALIBRATION_SPECS: tuple[ExportSpec, ...] = (
    ExportSpec(
        name="fineweb_edu",
        output_relpath="corpora/calibration/fineweb_edu.txt",
        units=(
            DatasetUnit(
                dataset_id="HuggingFaceFW/fineweb-edu",
                config_hints=("sample-10BT", "10BT", "sample"),
                split="train",
                max_records=4000,
            ),
        ),
        builder_name="generic",
        description="FineWeb-Edu prose replay",
    ),
    ExportSpec(
        name="the_stack_c_cpp",
        output_relpath="corpora/calibration/the_stack_c_cpp.txt",
        units=(
            DatasetUnit(
                dataset_id="bigcode/the-stack-v2",
                config_hints=("c",),
                split="train",
                max_records=1000,
            ),
            DatasetUnit(
                dataset_id="bigcode/the-stack-v2",
                config_hints=("cpp", "c++"),
                split="train",
                max_records=1000,
            ),
        ),
        builder_name="stack_code",
        description="The Stack v2 code replay for C/C++",
    ),
    ExportSpec(
        name="proof_pile2_science",
        output_relpath="corpora/calibration/proof_pile2_science.txt",
        units=(
            DatasetUnit(
                dataset_id="EleutherAI/proof-pile-2",
                config_hints=("arxiv",),
                split="train",
                max_records=1500,
            ),
            DatasetUnit(
                dataset_id="EleutherAI/proof-pile-2",
                config_hints=("openwebmath", "open-web-math", "math"),
                split="train",
                max_records=1500,
            ),
        ),
        builder_name="generic",
        description="Proof-Pile-2 science and math replay",
    ),
)


VALIDATION_SPECS: tuple[ExportSpec, ...] = (
    ExportSpec(
        name="gsm8k_prompts",
        output_relpath="corpora/validation/gsm8k_prompts.txt",
        units=(
            DatasetUnit(
                dataset_id="openai/gsm8k",
                config_hints=("main",),
                split="test",
                max_records=32,
            ),
        ),
        builder_name="gsm8k",
        description="GSM8K held-out prompt set",
    ),
    ExportSpec(
        name="humaneval_prompts",
        output_relpath="corpora/validation/humaneval_prompts.txt",
        units=(
            DatasetUnit(
                dataset_id="openai/openai_humaneval",
                config_hints=(),
                split="test",
                max_records=32,
            ),
        ),
        builder_name="humaneval",
        description="HumanEval held-out prompt set",
    ),
)


def _load_hf_datasets() -> tuple[Callable[..., Any], Callable[..., Any]]:
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError:
        print(
            "Missing dependency: install with 'python3 -m pip install -r scripts/requirements-ternary-corpus.txt'.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return get_dataset_config_names, load_dataset


def _dataset_requires_trust_remote_code(dataset_id: str) -> bool:
    return dataset_id in TRUST_REMOTE_CODE_DATASETS


def _with_dataset_options(dataset_id: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    options = dict(kwargs)
    if _dataset_requires_trust_remote_code(dataset_id):
        options["trust_remote_code"] = True
    return options


def _resolve_config(get_dataset_config_names: Callable[..., Any], dataset_id: str, hints: tuple[str, ...]) -> str | None:
    if dataset_id == PROOF_PILE_2_DATASET:
        return _resolve_proof_pile_2_config(hints)

    config_names = list(get_dataset_config_names(**_with_dataset_options(dataset_id, {"path": dataset_id})))
    if not config_names:
        return None
    if not hints:
        for preferred in ("default", "main", "plain_text"):
            for name in config_names:
                if _normalize_token(name) == _normalize_token(preferred):
                    return name
        if len(config_names) == 1:
            return config_names[0]
        raise RuntimeError(
            f"Dataset '{dataset_id}' has multiple configs; add hints. Available configs include: {', '.join(config_names[:10])}"
        )

    normalized = [(_config_match_tokens(name), name) for name in config_names]
    for hint in hints:
        target_tokens = _config_match_tokens(hint)
        exact = [name for norm_tokens, name in normalized if target_tokens & norm_tokens]
        if exact:
            return exact[0]
    for hint in hints:
        target_tokens = _config_match_tokens(hint)
        partial = [
            name
            for norm_tokens, name in normalized
            if any(
                target in norm or norm in target
                for target in target_tokens
                for norm in norm_tokens
            )
        ]
        if partial:
            return partial[0]

    raise RuntimeError(
        f"Unable to match config for dataset '{dataset_id}' using hints {hints}. Available configs include: {', '.join(config_names[:10])}"
    )


def _clean_sample(text: str) -> str | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if len(cleaned) < 24:
        return None
    return cleaned


def _stream_dataset(load_dataset: Callable[..., Any], unit: DatasetUnit, config_name: str | None):
    if unit.dataset_id == PROOF_PILE_2_DATASET:
        return _stream_proof_pile_2_records(unit, config_name)

    kwargs: dict[str, Any] = {
        "path": unit.dataset_id,
        "split": unit.split,
        "streaming": True,
    }
    if config_name is not None:
        kwargs["name"] = config_name
    return load_dataset(**_with_dataset_options(unit.dataset_id, kwargs))


def _get_hf_repo_files(repo_id: str, repo_type: str = "dataset") -> list[str]:
    cache_key = f"{repo_type}:{repo_id}"
    cached = _HF_REPO_FILE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "Missing dependency: install with 'python3 -m pip install -r scripts/requirements-ternary-corpus.txt'.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    files = list(HfApi().list_repo_files(repo_id, repo_type=repo_type))
    _HF_REPO_FILE_CACHE[cache_key] = files
    return files


def _stream_proof_pile_2_records(unit: DatasetUnit, config_name: str | None):
    try:
        import zstandard as zstd
        from huggingface_hub import hf_hub_url
    except ImportError:
        print(
            "Missing dependency: install with 'python3 -m pip install -r scripts/requirements-ternary-corpus.txt'.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    prefix = config_name or _resolve_proof_pile_2_config(unit.config_hints)
    split_prefix = f"{prefix}/{unit.split}/"
    repo_files = _get_hf_repo_files(PROOF_PILE_2_DATASET, repo_type="dataset")
    shard_paths = sorted(path for path in repo_files if path.startswith(split_prefix) and path.endswith(".jsonl.zst"))
    if not shard_paths:
        raise RuntimeError(
            f"No Proof-Pile-2 shard files found for config '{prefix}' split '{unit.split}'."
        )

    request_headers = {"User-Agent": "sapphire/1.0"}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        request_headers["Authorization"] = f"Bearer {hf_token}"

    def _iter_records():
        for shard_path in shard_paths:
            print(f"[corpus] streaming Proof-Pile-2 shard: {shard_path}", file=sys.stderr, flush=True)
            shard_url = hf_hub_url(repo_id=PROOF_PILE_2_DATASET, filename=shard_path, repo_type="dataset")
            request = urllib.request.Request(shard_url, headers=request_headers)
            with urllib.request.urlopen(request, timeout=STACK_PUBLIC_FETCH_TIMEOUT) as response:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(response) as reader:
                    with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
                        for line in text_stream:
                            line = line.strip()
                            if not line:
                                continue
                            payload = json.loads(line)
                            if isinstance(payload, dict):
                                yield payload

    return _iter_records()


def _format_dataset_error(dataset_id: str, exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()

    if "gated dataset" in lowered or "you must be authenticated" in lowered:
        return (
            f"Dataset '{dataset_id}' is gated. Accept the dataset terms on Hugging Face and authenticate this environment "
            f"(for example with 'huggingface-cli login' or by exporting HF_TOKEN), then rerun the helper."
        )
    if "trust_remote_code" in lowered:
        return (
            f"Dataset '{dataset_id}' requires remote dataset code. The helper now enables this automatically; "
            f"if you still see this error, update the 'datasets' package in .venv and retry."
        )
    if dataset_id == THE_STACK_DATASET and ("software heritage" in lowered or "aws credentials" in lowered or "hf_token only" in lowered):
        return message
    return message


def _stack_cache_filename(config_name: str | None) -> str:
    token = _cache_token(config_name or "default")
    return f"{token}.jsonl"


def _stack_cache_path(cache_dir: Path, config_name: str | None) -> Path:
    return cache_dir / _stack_cache_filename(config_name)


def _source_cache_filename(unit: DatasetUnit, config_name: str | None, builder_name: str) -> str:
    dataset_token = _cache_token(unit.dataset_id) or "dataset"
    config_token = _cache_token(config_name or "default") or "default"
    split_token = _cache_token(unit.split) or "split"
    builder_token = _cache_token(builder_name) or "builder"
    return f"{dataset_token}__{config_token}__{split_token}__{builder_token}.jsonl"


def _source_cache_path(cache_dir: Path, unit: DatasetUnit, config_name: str | None, builder_name: str) -> Path:
    return cache_dir / _source_cache_filename(unit, config_name, builder_name)


def _load_cached_text_samples(cache_path: Path) -> list[str]:
    samples: list[str] = []
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            text = payload.get("text") if isinstance(payload, dict) else None
            if isinstance(text, str) and text and _looks_like_text_sample(text):
                samples.append(text)
    return samples


def _looks_like_text_sample(text: str) -> bool:
    if not text:
        return False
    if "\ufffd" in text:
        return False

    control_count = 0
    printable_count = 0
    for ch in text:
        if ch in ("\n", "\r", "\t"):
            printable_count += 1
            continue
        if ch.isprintable():
            printable_count += 1
            continue
        control_count += 1

    total = len(text)
    if total <= 0:
        return False
    if control_count > 0 and (control_count / total) > 0.01:
        return False
    return (printable_count / total) >= 0.95


def _store_cached_text_samples(cache_path: Path, samples: list[str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps({"text": sample}, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temp_path.replace(cache_path)


def _emit_samples_to_handle(handle: Any, seen_prefixes: set[str], samples: list[str], target_count: int) -> int:
    written = 0
    for sample in samples:
        dedupe_key = sample[:512]
        if dedupe_key in seen_prefixes:
            continue
        seen_prefixes.add(dedupe_key)
        handle.write(sample)
        handle.write("\n\n")
        written += 1
        if written >= target_count:
            break
    return written


def _collect_streamed_samples(
    dataset: Any,
    builder: Callable[[dict[str, Any]], str | None],
    label: str,
    target_count: int,
) -> list[str]:
    samples: list[str] = []
    local_seen: set[str] = set()
    progress_state, stop_event, progress_thread = _start_progress_reporter(label, target_count)

    try:
        for record in dataset:
            progress_state["scanned"] = int(progress_state["scanned"]) + 1
            sample = builder(record)
            if not sample:
                continue
            sample = _clean_sample(sample)
            if not sample:
                continue
            dedupe_key = sample[:512]
            if dedupe_key in local_seen:
                continue
            local_seen.add(dedupe_key)
            samples.append(sample)
            progress_state["written"] = len(samples)
            if len(samples) >= target_count:
                break
    finally:
        _stop_progress_reporter(label, target_count, progress_state, stop_event, progress_thread)

    return samples


def _extract_stack_metadata(record: dict[str, Any]) -> dict[str, Any] | None:
    blob_id = record.get("blob_id")
    if not isinstance(blob_id, str) or not blob_id.strip():
        return None
    if record.get("is_generated") is True or record.get("is_vendor") is True:
        return None

    metadata: dict[str, Any] = {"blob_id": blob_id.strip()}
    for key in ("src_encoding", "path", "language", "extension", "license_type"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            metadata[key] = value.strip()
    length_bytes = record.get("length_bytes")
    if isinstance(length_bytes, int) and length_bytes > 0:
        metadata["length_bytes"] = length_bytes
    return metadata


def _ensure_stack_metadata_cache(
    repo_root: Path,
    cache_dir: Path,
    unit: DatasetUnit,
    config_name: str | None,
    load_dataset: Callable[..., Any],
    refresh_cache: bool,
    cache_limit: int,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _stack_cache_path(cache_dir, config_name)
    if cache_path.exists() and cache_path.stat().st_size > 0 and not refresh_cache:
        print(f"[corpus] using cached stack metadata: {cache_path}", file=sys.stderr, flush=True)
        return cache_path

    dataset = _stream_dataset(load_dataset, unit, config_name)
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    progress_label = f"stack-cache/{config_name or '<default>'}"
    progress_target = cache_limit if cache_limit > 0 else 0
    progress_state, stop_event, progress_thread = _start_progress_reporter(progress_label, progress_target)
    cached_count = 0

    print(
        f"[corpus] caching The Stack metadata: dataset={unit.dataset_id} config={config_name or '<default>'} path={cache_path}",
        file=sys.stderr,
        flush=True,
    )

    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            for record in dataset:
                progress_state["scanned"] = int(progress_state["scanned"]) + 1
                metadata = _extract_stack_metadata(record)
                if not metadata:
                    continue
                handle.write(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
                cached_count += 1
                progress_state["written"] = cached_count
                if cache_limit > 0 and cached_count >= cache_limit:
                    break
    finally:
        _stop_progress_reporter(progress_label, progress_target, progress_state, stop_event, progress_thread)

    if cached_count <= 0:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"No The Stack metadata rows were cached for config '{config_name}'.")

    temp_path.replace(cache_path)
    print(f"[corpus] cached {cached_count} stack metadata rows -> {cache_path}", file=sys.stderr, flush=True)
    return cache_path


def _sample_stack_metadata(cache_path: Path, sample_size: int, seed: int) -> tuple[list[dict[str, Any]], int]:
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    total_rows = 0

    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            metadata = json.loads(line)
            if total_rows < sample_size:
                reservoir.append(metadata)
            else:
                pick = rng.randint(0, total_rows)
                if pick < sample_size:
                    reservoir[pick] = metadata
            total_rows += 1

    rng.shuffle(reservoir)
    return reservoir, total_rows


def _stack_sample_pool_size(target: int) -> int:
    scaled = target * STACK_SAMPLE_MULTIPLIER
    if scaled < STACK_SAMPLE_MIN_CANDIDATES:
        scaled = STACK_SAMPLE_MIN_CANDIDATES
    return scaled


def _export_stack_unit_from_cache(
    spec: ExportSpec,
    unit: DatasetUnit,
    config_name: str | None,
    handle: Any,
    seen_prefixes: set[str],
    repo_root: Path,
    cache_dir: Path,
    source_cache_dir: Path,
    load_dataset: Callable[..., Any],
    refresh_cache: bool,
    cache_limit: int,
    refresh_source_cache: bool,
) -> int:
    source_cache_path = _source_cache_path(source_cache_dir, unit, config_name, spec.builder_name)
    if source_cache_path.exists() and source_cache_path.stat().st_size > 0 and not refresh_source_cache:
        cached_samples = _load_cached_text_samples(source_cache_path)
        if cached_samples:
            print(f"[corpus] using cached source samples: {source_cache_path}", file=sys.stderr, flush=True)
            written = _emit_samples_to_handle(handle, seen_prefixes, cached_samples, unit.max_records)
            if written > 0:
                return written
        print(f"[corpus] stale source cache detected, rebuilding: {source_cache_path}", file=sys.stderr, flush=True)
        source_cache_path.unlink(missing_ok=True)

    cache_path = _ensure_stack_metadata_cache(
        repo_root=repo_root,
        cache_dir=cache_dir,
        unit=unit,
        config_name=config_name,
        load_dataset=load_dataset,
        refresh_cache=refresh_cache,
        cache_limit=cache_limit,
    )
    sample_pool_size = _stack_sample_pool_size(unit.max_records)
    sample_seed = STACK_RANDOM_SEED + sum(ord(ch) for ch in (config_name or "default"))
    candidates, total_rows = _sample_stack_metadata(cache_path, sample_pool_size, sample_seed)

    print(
        f"[corpus] sampled {len(candidates)} candidate metadata rows from {cache_path} (cached_rows={total_rows})",
        file=sys.stderr,
        flush=True,
    )

    progress_label = f"stack-fetch/{config_name or '<default>'}"
    progress_state, stop_event, progress_thread = _start_progress_reporter(progress_label, unit.max_records)
    fetched_samples: list[str] = []
    local_seen: set[str] = set()

    try:
        for metadata in candidates:
            progress_state["scanned"] = int(progress_state["scanned"]) + 1
            try:
                sample = _build_stack_code_text(metadata)
            except Exception as exc:
                raise RuntimeError(_format_dataset_error(THE_STACK_DATASET, exc)) from exc
            if not sample:
                continue
            sample = _clean_sample(sample)
            if not sample:
                continue
            dedupe_key = sample[:512]
            if dedupe_key in local_seen:
                continue
            local_seen.add(dedupe_key)
            fetched_samples.append(sample)
            progress_state["written"] = len(fetched_samples)
            if len(fetched_samples) >= unit.max_records:
                break
    finally:
        _stop_progress_reporter(progress_label, unit.max_records, progress_state, stop_event, progress_thread)

    if not fetched_samples:
        raise RuntimeError(
            f"No The Stack samples were exported for config '{config_name}'. Ensure metadata is cached and S3 content access is configured."
        )
    if len(fetched_samples) < unit.max_records:
        raise RuntimeError(
            f"Only exported {len(fetched_samples)}/{unit.max_records} The Stack samples for config '{config_name}'. "
            "Increase the cache size or sampling pool if needed."
        )
    _store_cached_text_samples(source_cache_path, fetched_samples)
    print(f"[corpus] cached source samples -> {source_cache_path}", file=sys.stderr, flush=True)
    written = _emit_samples_to_handle(handle, seen_prefixes, fetched_samples, unit.max_records)
    if written <= 0:
        raise RuntimeError(f"Fetched The Stack samples for config '{config_name}' produced no usable output.")
    return written


def _start_progress_reporter(label: str, target: int) -> tuple[dict[str, Any], threading.Event, threading.Thread]:
    state: dict[str, Any] = {
        "scanned": 0,
        "written": 0,
        "started_at": time.monotonic(),
    }
    stop_event = threading.Event()

    def _reporter() -> None:
        last_written = -1
        last_scanned = -1
        while not stop_event.wait(5.0):
            scanned = int(state["scanned"])
            written = int(state["written"])
            elapsed = max(time.monotonic() - float(state["started_at"]), 0.001)
            if scanned == last_scanned and written == last_written:
                print(
                    f"[corpus] {label}: waiting... scanned={scanned} written={written}/{target} elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"[corpus] {label}: progress scanned={scanned} written={written}/{target} elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                last_scanned = scanned
                last_written = written

    thread = threading.Thread(target=_reporter, name="corpus-progress", daemon=True)
    thread.start()
    return state, stop_event, thread


def _stop_progress_reporter(
    label: str,
    target: int,
    state: dict[str, Any],
    stop_event: threading.Event,
    thread: threading.Thread,
) -> None:
    stop_event.set()
    thread.join(timeout=0.2)
    elapsed = max(time.monotonic() - float(state["started_at"]), 0.001)
    print(
        f"[corpus] {label}: done scanned={int(state['scanned'])} written={int(state['written'])}/{target} elapsed={elapsed:.1f}s",
        file=sys.stderr,
        flush=True,
    )


def _write_export_spec(
    spec: ExportSpec,
    repo_root: Path,
    get_dataset_config_names: Callable[..., Any],
    load_dataset: Callable[..., Any],
    stack_cache_dir: Path,
    source_cache_dir: Path,
    refresh_stack_cache: bool,
    stack_cache_limit: int,
    refresh_source_cache: bool,
) -> int:
    builder = BUILDERS[spec.builder_name]
    output_path = repo_root / spec.output_relpath
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    seen_prefixes: set[str] = set()

    with output_path.open("w", encoding="utf-8") as handle:
        for unit in spec.units:
            try:
                config_name = _resolve_config(get_dataset_config_names, unit.dataset_id, unit.config_hints)
            except Exception as exc:
                raise RuntimeError(_format_dataset_error(unit.dataset_id, exc)) from exc

            print(
                f"[corpus] exporting {spec.name}: dataset={unit.dataset_id} config={config_name or '<default>'} split={unit.split} limit={unit.max_records}",
                file=sys.stderr,
            )

            if unit.dataset_id == THE_STACK_DATASET:
                unit_written = _export_stack_unit_from_cache(
                    spec=spec,
                    unit=unit,
                    config_name=config_name,
                    handle=handle,
                    seen_prefixes=seen_prefixes,
                    repo_root=repo_root,
                    cache_dir=stack_cache_dir,
                    source_cache_dir=source_cache_dir,
                    load_dataset=load_dataset,
                    refresh_cache=refresh_stack_cache,
                    cache_limit=stack_cache_limit,
                    refresh_source_cache=refresh_source_cache,
                )
                total_written += unit_written
                continue

            source_cache_path = _source_cache_path(source_cache_dir, unit, config_name, spec.builder_name)
            if source_cache_path.exists() and source_cache_path.stat().st_size > 0 and not refresh_source_cache:
                cached_samples = _load_cached_text_samples(source_cache_path)
                if cached_samples:
                    print(f"[corpus] using cached source samples: {source_cache_path}", file=sys.stderr, flush=True)
                    unit_written = _emit_samples_to_handle(handle, seen_prefixes, cached_samples, unit.max_records)
                    total_written += unit_written
                    if unit_written > 0:
                        continue
                print(f"[corpus] stale source cache detected, rebuilding: {source_cache_path}", file=sys.stderr, flush=True)
                source_cache_path.unlink(missing_ok=True)

            dataset = _stream_dataset(load_dataset, unit, config_name)
            progress_label = f"{spec.name}/{config_name or '<default>'}"
            collected_samples = _collect_streamed_samples(dataset, builder, progress_label, unit.max_records)
            _store_cached_text_samples(source_cache_path, collected_samples)
            print(f"[corpus] cached source samples -> {source_cache_path}", file=sys.stderr, flush=True)
            unit_written = _emit_samples_to_handle(handle, seen_prefixes, collected_samples, unit.max_records)
            total_written += unit_written

            if unit_written == 0:
                raise RuntimeError(
                    f"No samples were exported for {spec.name} from dataset '{unit.dataset_id}' config '{config_name}'."
                )

    print(f"[corpus] wrote {total_written} samples -> {output_path}", file=sys.stderr)
    return total_written


def _select_specs(include_validation: bool) -> tuple[ExportSpec, ...]:
    if include_validation:
        return CALIBRATION_SPECS + VALIDATION_SPECS
    return CALIBRATION_SPECS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export high-signal plain-text corpora for Sapphire ternary conversion manifests."
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repository root where ./corpora and ./configs live (default: repo root inferred from script path).",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Export only calibration corpora and skip GSM8K/HumanEval validation prompts.",
    )
    parser.add_argument(
        "--stack-cache-dir",
        default=str(DEFAULT_STACK_CACHE_DIR),
        help="Directory for local The Stack v2 metadata caches (default: ./.cache/the-stack-v2).",
    )
    parser.add_argument(
        "--source-cache-dir",
        default=str(DEFAULT_SOURCE_CACHE_DIR),
        help="Directory for cached sampled source texts for all corpus inputs (default: ./.cache/corpus-sources).",
    )
    parser.add_argument(
        "--stack-cache-only",
        action="store_true",
        help="Build or refresh the local The Stack v2 metadata caches and exit without exporting corpora.",
    )
    parser.add_argument(
        "--refresh-stack-cache",
        action="store_true",
        help="Rebuild The Stack v2 metadata caches even if cache files already exist.",
    )
    parser.add_argument(
        "--stack-cache-limit",
        type=int,
        default=0,
        help="Optional cap on cached The Stack metadata rows per language unit (0 = cache the full filtered metadata stream).",
    )
    parser.add_argument(
        "--refresh-source-cache",
        action="store_true",
        help="Rebuild cached sampled source texts for all corpus inputs even if cache files already exist.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    stack_cache_dir = Path(args.stack_cache_dir).resolve()
    source_cache_dir = Path(args.source_cache_dir).resolve()
    get_dataset_config_names, load_dataset = _load_hf_datasets()

    try:
        specs = _select_specs(include_validation=not args.skip_validation)
        if args.stack_cache_only:
            for spec in specs:
                for unit in spec.units:
                    if unit.dataset_id != THE_STACK_DATASET:
                        continue
                    config_name = _resolve_config(get_dataset_config_names, unit.dataset_id, unit.config_hints)
                    _ensure_stack_metadata_cache(
                        repo_root=repo_root,
                        cache_dir=stack_cache_dir,
                        unit=unit,
                        config_name=config_name,
                        load_dataset=load_dataset,
                        refresh_cache=args.refresh_stack_cache,
                        cache_limit=args.stack_cache_limit,
                    )
            print("[corpus] stack metadata cache complete", file=sys.stderr)
            return 0
        for spec in specs:
            _write_export_spec(
                spec,
                repo_root,
                get_dataset_config_names,
                load_dataset,
                stack_cache_dir,
                source_cache_dir,
                args.refresh_stack_cache,
                args.stack_cache_limit,
                args.refresh_source_cache,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("[corpus] complete", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
