#!/usr/bin/env python3
"""Build and verify the STGR decoded-frame cache outside training.

Every uncached sample is decoded in a fresh subprocess.  The parent enforces a
wall-clock timeout and records both Python decoder errors and native decoder
hangs as ``failure.json`` sentinels.  A complete cache therefore contains one
terminal entry per dataset row: either RGB PNGs plus ``metadata.json``, or a
known-failure sentinel that the training dataset maps to dummy visuals and
zero reward.  No failed row is accepted implicitly: ``--max-failed-rows`` must
explicitly authorize the observed number before a manifest is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mesd.preprocess import (  # noqa: E402
    STGRFrameCacheError,
    STGR_FRAME_CACHE_VERSION,
    STGRVisionError,
    load_stgr_frame_cache_entry,
    prepare_stgr_sample,
    stgr_frame_cache_entry,
    store_stgr_frame_cache_entry,
    store_stgr_frame_cache_failure,
)
from mesd.schema import STGRRecord, load_stgr_json_records  # noqa: E402


SCHEMA_VERSION = 1
BUILDER_VERSION = 1
SUMMARY_FILENAME = "build_summary.json"
RESULTS_FILENAME = "build_results.jsonl"
FAILURE_FILENAME = "failure.json"

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


@dataclass
class PendingTask:
    row_index: int
    record: STGRRecord
    cache_entry: Path


@dataclass
class RunningTask:
    pending: PendingTask
    process: subprocess.Popen[Any]
    started_monotonic: float
    data_root: Path
    cache_root: Path
    task_path: Path
    result_path: Path
    log_path: Path
    log_handle: Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}-{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}-{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tail_text(path: Path, max_bytes: int = 16_384) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("decord", "numpy", "pillow", "qwen-vl-utils", "torch", "torchvision"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _validate_decoder_runtime() -> None:
    """Fail before queuing rows when the training decoder stack is unusable."""

    try:
        import decord  # noqa: F401
        from mesd import vision_process  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "STGR cache building requires a working Decord/Qwen vision runtime; "
            f"runtime import failed: {type(exc).__name__}: {exc}"
        ) from exc


def _vision_env() -> dict[str, str | None]:
    names = (
        "VIDEO_PIXELS_FACTOR",
        "VIDEO_MAX_PIXELS",
        "FORCE_QWENVL_VIDEO_READER",
        "DECORD_EOF_RETRY_MAX",
    )
    return {name: os.environ.get(name) for name in names}


def _read_failure(cache_entry: Path, record: STGRRecord) -> dict[str, Any] | None:
    path = cache_entry / FAILURE_FILENAME
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping):
        return None
    required = ("version", "fingerprint", "record_id", "error_type", "error_message")
    if any(key not in value for key in required):
        return None
    if value["version"] != STGR_FRAME_CACHE_VERSION:
        return None
    if str(value["fingerprint"]) != cache_entry.name:
        return None
    if str(value["record_id"]) != record.id:
        return None
    return dict(value)


def _classify_entry(
    record: STGRRecord,
    data_root: Path,
    cache_root: Path,
) -> tuple[str, dict[str, Any] | None]:
    entry = stgr_frame_cache_entry(record, data_root, cache_root)
    metadata_exists = (entry / "metadata.json").exists()
    failure_exists = (entry / FAILURE_FILENAME).exists()
    if metadata_exists:
        try:
            sample = load_stgr_frame_cache_entry(record, data_root, cache_root, required=True)
        except Exception as exc:
            return "corrupt", {
                "reason": "invalid metadata or PNG frame",
                "error_message": str(exc),
            }
        if sample is None:  # Defensive: required=True should make this impossible.
            return "corrupt", {"reason": "required cache loader returned no sample"}
        return "ready", {"frames": len(sample.images)}
    if failure_exists:
        try:
            load_stgr_frame_cache_entry(record, data_root, cache_root, required=True)
        except STGRVisionError:
            failure = _read_failure(entry, record)
            if failure is not None:
                return "failed", failure
            return "corrupt", {"reason": "validated failure sentinel could not be reread"}
        except STGRFrameCacheError as exc:
            return "corrupt", {"reason": "invalid failure sentinel", "error_message": str(exc)}
        except Exception as exc:
            return "corrupt", {"reason": "failure validation error", "error_message": str(exc)}
        return "corrupt", {"reason": "failure sentinel unexpectedly loaded as a success"}
    return "missing", None


def _terminate_process(process: subprocess.Popen[Any], grace_seconds: float) -> bool:
    if process.poll() is not None:
        return True
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return True
    try:
        process.wait(timeout=grace_seconds)
        return True
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=max(grace_seconds, 0.1))
        return True
    except subprocess.TimeoutExpired:
        # A task stuck in uninterruptible kernel I/O cannot be reaped yet.  Do
        # not dispatch more workers in that state; the caller aborts the build.
        return False


def _launch_task(pending: PendingTask, work_dir: Path, data_root: Path, cache_root: Path) -> RunningTask:
    token = f"{pending.row_index:08d}-{uuid4().hex}"
    task_path = work_dir / f"{token}.task.json"
    result_path = work_dir / f"{token}.result.json"
    log_path = work_dir / f"{token}.log"
    _atomic_json(
        task_path,
        {
            "row_index": pending.row_index,
            # Cache entries contain vision data only.  Avoid copying large
            # key_items/answer payloads through every isolated subprocess.
            "record": {
                "id": pending.record.id,
                "source": pending.record.source,
                "task": pending.record.task,
                "question": "",
                "answer": "",
                "video_path": pending.record.video_path,
                "image_path": pending.record.image_path,
                "image_size": list(pending.record.image_size),
                "key_frames": list(pending.record.key_frames),
            },
            "data_root": str(data_root),
            "cache_root": str(cache_root),
        },
    )
    log_handle = log_path.open("wb")
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_worker",
            "--task-path",
            str(task_path),
            "--result-path",
            str(result_path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=(os.name == "posix"),
    )
    return RunningTask(
        pending=pending,
        process=process,
        started_monotonic=time.monotonic(),
        data_root=data_root,
        cache_root=cache_root,
        task_path=task_path,
        result_path=result_path,
        log_path=log_path,
        log_handle=log_handle,
    )


def _record_failure(
    pending: PendingTask,
    *,
    error_type: str,
    error_message: str,
    timed_out: bool = False,
    worker_log_tail: str = "",
) -> tuple[str, dict[str, Any]]:
    extra: dict[str, Any] = {
        "row_index": pending.row_index,
        "timed_out": timed_out,
        "created_at": _utc_now(),
    }
    if worker_log_tail:
        extra["worker_log_tail"] = worker_log_tail
    try:
        store_stgr_frame_cache_failure(
            pending.cache_entry,
            record_id=pending.record.id,
            error_type=error_type,
            error_message=error_message,
            **extra,
        )
    except OSError as exc:
        return "unrecoverable", {
            "error_type": type(exc).__name__,
            "error_message": f"could not write failure sentinel: {exc}",
        }
    return "failed", {
        "error_type": error_type,
        "error_message": error_message,
        "timed_out": timed_out,
    }


def _finalize_task(
    running: RunningTask,
    *,
    timeout_seconds: float,
    kill_grace_seconds: float,
) -> dict[str, Any]:
    process = running.process
    elapsed = time.monotonic() - running.started_monotonic
    timed_out = process.poll() is None and elapsed >= timeout_seconds
    if timed_out:
        if not _terminate_process(process, kill_grace_seconds):
            running.log_handle.close()
            raise RuntimeError(
                "decoder worker remained alive after SIGKILL; aborting to avoid "
                f"accumulating uninterruptible processes (pid={process.pid})"
            )
    else:
        process.wait()
    running.log_handle.close()
    log_tail = _tail_text(running.log_path)

    worker_result: dict[str, Any] | None = None
    if not timed_out and running.result_path.is_file():
        try:
            value = json.loads(running.result_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                worker_result = value
        except (OSError, json.JSONDecodeError):
            worker_result = None

    if timed_out:
        status, detail = _record_failure(
            running.pending,
            error_type="TimeoutError",
            error_message=f"decoder exceeded {timeout_seconds:.1f}s",
            timed_out=True,
            worker_log_tail=log_tail,
        )
    elif worker_result is None and isinstance(process.returncode, int) and process.returncode < 0:
        status, detail = _record_failure(
            running.pending,
            error_type="NativeWorkerCrash",
            error_message=f"decoder worker was terminated by signal {-process.returncode}",
            worker_log_tail=log_tail,
        )
    elif worker_result is None:
        status, detail = "unrecoverable", {
            "error_type": "WorkerProtocolError",
            "error_message": (
                "worker exited without a valid result "
                f"(returncode={process.returncode})"
            ),
        }
        if log_tail:
            detail["worker_log_tail"] = log_tail
    elif worker_result.get("status") == "unrecoverable":
        status, detail = "unrecoverable", {
            "error_type": str(worker_result.get("error_type", "WorkerError")),
            "error_message": str(worker_result.get("error_message", "unknown worker error")),
        }
    elif worker_result.get("status") == "failed":
        traceback_tail = str(worker_result.get("traceback", ""))[-16_384:]
        diagnostic_tail = "\n".join(value for value in (log_tail, traceback_tail) if value)
        status, detail = _record_failure(
            running.pending,
            error_type=str(worker_result.get("error_type", "DecodeError")),
            error_message=str(worker_result.get("error_message", "unknown decoder error")),
            worker_log_tail=diagnostic_tail,
        )
    elif worker_result.get("status") != "ready":
        status, detail = "unrecoverable", {
            "error_type": "WorkerProtocolError",
            "error_message": f"unsupported worker status {worker_result.get('status')!r}",
        }
    else:
        entry_status, entry_detail = _classify_entry(
            running.pending.record,
            running.data_root,
            running.cache_root,
        )
        if entry_status == "ready":
            status, detail = "ready", entry_detail or {}
        else:
            status, detail = "unrecoverable", {
                "error_type": "CacheWriteError",
                "error_message": "worker returned ready but the cache entry did not validate",
            }

    result = {
        "row_index": running.pending.row_index,
        "record_id": running.pending.record.id,
        "fingerprint": running.pending.cache_entry.name,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - running.started_monotonic, 6),
        **detail,
    }
    for path in (running.task_path, running.result_path, running.log_path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    return result


def _worker_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task-path", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        task = json.loads(args.task_path.read_text(encoding="utf-8"))
        record = STGRRecord.from_mapping(task["record"])
        data_root = Path(task["data_root"])
        cache_root = Path(task["cache_root"])
        cache_entry = stgr_frame_cache_entry(record, data_root, cache_root)
        sample = prepare_stgr_sample(record, data_root)
        store_stgr_frame_cache_entry(cache_entry, sample)
        result = {
            "status": "ready",
            "record_id": record.id,
            "frames": len(sample.images),
            "data_root": str(data_root),
            "cache_root": str(cache_root),
        }
        returncode = 0
    except STGRVisionError as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        returncode = 10
    except Exception as exc:
        result = {
            "status": "unrecoverable",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        returncode = 11
    try:
        _atomic_json(args.result_path, result)
    except Exception:
        traceback.print_exc()
        return 12
    return returncode


def _failure_example(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in ("row_index", "record_id", "status", "error_type", "error_message")
        if result.get(key) is not None
    }


def _summary(
    *,
    args: argparse.Namespace,
    created_at: str,
    dataset_sha256: str,
    rows_total: int,
    results: Sequence[Mapping[str, Any]],
    fatal_error: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    counts = Counter(str(result.get("status", "unrecoverable")) for result in results)
    ready = counts["ready"]
    failed = counts["failed"]
    unrecoverable = rows_total - ready - failed
    failure_limit_exceeded = failed > args.max_failed_rows
    complete = (
        fatal_error is None
        and rows_total > 0
        and ready > 0
        and unrecoverable == 0
        and len(results) == rows_total
        and not failure_limit_exceeded
    )
    failure_examples = [
        _failure_example(result)
        for result in results
        if result.get("status") != "ready"
    ][:100]
    return {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "cache_version": STGR_FRAME_CACHE_VERSION,
        "mode": "verify" if args.verify_existing else "build",
        "complete": complete,
        "created_at": created_at,
        "finished_at": _utc_now(),
        "dataset_path": str(args.dataset),
        "dataset_sha256": dataset_sha256,
        "media_root": str(args.data_root),
        "cache_dir": str(args.cache_dir),
        "rows_total": rows_total,
        "ready": ready,
        "failed": failed,
        "unrecoverable": unrecoverable,
        "max_failed_rows": args.max_failed_rows,
        "failure_limit_exceeded": failure_limit_exceeded,
        "result_counts": dict(sorted(counts.items())),
        "fatal_error": fatal_error,
        "failure_examples": failure_examples,
        "jobs": args.jobs,
        "timeout_seconds": args.timeout_seconds,
        "resume": args.resume,
        "retry_failures": args.retry_failures,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "package_versions": _package_versions(),
        "vision_env": _vision_env(),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode STGR samples into a validated PNG cache before training."
    )
    parser.add_argument("dataset", type=Path, help="STGR-RL.json or JSONL annotation")
    parser.add_argument("--data-root", type=Path, required=True, help="STGR root containing videos/")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Output/shared frame cache")
    parser.add_argument("--jobs", type=int, default=4, help="Concurrent isolated decoder subprocesses")
    parser.add_argument("--timeout-seconds", type=float, default=300.0, help="Wall timeout per sample")
    parser.add_argument("--kill-grace-seconds", type=float, default=5.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--max-failed-rows",
        type=int,
        default=0,
        help="Maximum known-bad rows accepted as a complete cache (default: 0)",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse valid success and failure entries")
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="With --resume, retry known failure sentinels instead of reusing them",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Only verify that every row has a valid cache entry or failure sentinel",
    )
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test the first N rows")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.dataset.is_file():
        parser.error(f"dataset does not exist: {args.dataset}")
    if not args.data_root.is_dir():
        parser.error(f"data root does not exist: {args.data_root}")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.timeout_seconds <= 0 or args.kill_grace_seconds < 0:
        parser.error("timeout values must be positive")
    if args.progress_every < 1 or args.limit < 0 or args.max_failed_rows < 0:
        parser.error("progress, limit, and max-failed-rows values must be non-negative")
    if args.retry_failures and not args.resume:
        parser.error("--retry-failures requires --resume")
    if args.verify_existing and args.limit:
        parser.error("--verify-existing always checks the full dataset; remove --limit")


def _iter_prepared_tasks(
    rows: Sequence[dict[str, Any]], data_root: Path, cache_root: Path
) -> Iterable[PendingTask | dict[str, Any]]:
    for row_index, raw_row in enumerate(rows):
        try:
            record = STGRRecord.from_mapping(raw_row)
            # Retain only vision-relevant fields while thousands of tasks wait
            # in the parent queue.  The full row is validated above, but
            # answer/key_items are neither cached nor part of the fingerprint.
            record = replace(record, question="", answer="", key_items={})
            entry = stgr_frame_cache_entry(record, data_root, cache_root)
        except Exception as exc:
            yield {
                "row_index": row_index,
                "record_id": str(raw_row.get("id", "")),
                "status": "unrecoverable",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            continue
        yield PendingTask(row_index, record, entry)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.dataset = args.dataset.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    _validate_args(parser, args)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_sha256 = _sha256_file(args.dataset)
    rows = load_stgr_json_records(args.dataset)
    rows_total = len(rows)
    if args.limit:
        rows = rows[: args.limit]
    created_at = _utc_now()
    results: list[dict[str, Any]] = []
    pending: deque[PendingTask] = deque()
    seen_fingerprints: dict[str, int] = {}

    for value in _iter_prepared_tasks(rows, args.data_root, args.cache_dir):
        if isinstance(value, dict):
            results.append(value)
            continue
        fingerprint = value.cache_entry.name
        previous_row = seen_fingerprints.get(fingerprint)
        if previous_row is not None:
            results.append(
                {
                    "row_index": value.row_index,
                    "record_id": value.record.id,
                    "fingerprint": fingerprint,
                    "status": "unrecoverable",
                    "error_type": "DuplicateCacheFingerprint",
                    "error_message": f"fingerprint already belongs to row {previous_row}",
                }
            )
            continue
        seen_fingerprints[fingerprint] = value.row_index
        entry_status, detail = _classify_entry(value.record, args.data_root, args.cache_dir)
        if args.verify_existing:
            result_status = entry_status if entry_status in ("ready", "failed") else "unrecoverable"
            results.append(
                {
                    **(detail or {}),
                    "row_index": value.row_index,
                    "record_id": value.record.id,
                    "fingerprint": value.cache_entry.name,
                    "status": result_status,
                }
            )
        elif args.resume and (
            entry_status == "ready" or (entry_status == "failed" and not args.retry_failures)
        ):
            results.append(
                {
                    **(detail or {}),
                    "row_index": value.row_index,
                    "record_id": value.record.id,
                    "fingerprint": value.cache_entry.name,
                    "status": entry_status,
                    "resumed": True,
                }
            )
        else:
            pending.append(value)

    print(
        f"STGR cache: rows={rows_total}, queued={len(pending)}, "
        f"terminal_existing={len(results)}, jobs={args.jobs}",
        flush=True,
    )

    work_dir = args.cache_dir / ".builder_tmp" / f"{int(time.time())}-{os.getpid()}"
    active: dict[int, RunningTask] = {}
    completed_workers = 0
    fatal_error: dict[str, str] | None = None
    try:
        if pending:
            _validate_decoder_runtime()
        if pending:
            work_dir.mkdir(parents=True, exist_ok=True)
        while pending or active:
            while pending and len(active) < args.jobs:
                running = _launch_task(pending.popleft(), work_dir, args.data_root, args.cache_dir)
                active[running.process.pid] = running

            now = time.monotonic()
            finished = [
                pid
                for pid, running in active.items()
                if running.process.poll() is not None
                or now - running.started_monotonic >= args.timeout_seconds
            ]
            if not finished:
                time.sleep(0.2)
                continue
            for pid in finished:
                result = _finalize_task(
                    active.pop(pid),
                    timeout_seconds=args.timeout_seconds,
                    kill_grace_seconds=args.kill_grace_seconds,
                )
                results.append(result)
                completed_workers += 1
                if (
                    result["status"] != "ready"
                    or completed_workers % args.progress_every == 0
                    or not (pending or active)
                ):
                    print(
                        f"cache {completed_workers}: status={result['status']} "
                        f"row={result['row_index']} id={result['record_id']}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        fatal_error = {"type": "KeyboardInterrupt", "message": "cache build interrupted"}
        print("Interrupted; stopping active decoder subprocesses.", file=sys.stderr)
    except Exception as exc:
        fatal_error = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    finally:
        for running in active.values():
            _terminate_process(running.process, args.kill_grace_seconds)
            running.log_handle.close()
        results.sort(key=lambda item: int(item.get("row_index", -1)))
        _atomic_jsonl(args.cache_dir / RESULTS_FILENAME, results)
        summary = _summary(
            args=args,
            created_at=created_at,
            dataset_sha256=dataset_sha256,
            rows_total=rows_total,
            results=results,
            fatal_error=fatal_error,
        )
        _atomic_json(args.cache_dir / SUMMARY_FILENAME, summary)
        print(
            f"Summary: ready={summary['ready']}, failed={summary['failed']}, "
            f"unrecoverable={summary['unrecoverable']}, "
            f"max_failed_rows={summary['max_failed_rows']}, complete={summary['complete']}",
            flush=True,
        )

    if fatal_error and fatal_error["type"] == "KeyboardInterrupt":
        return 130
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        raise SystemExit(_worker_main(sys.argv[2:]))
    raise SystemExit(main())
