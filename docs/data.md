# STGR training data

Use the `STGR-RL.json` annotation file from the original Open-o3 Video release or an STGR-compatible JSON or JSONL manifest. The annotation manifest and media are separate. Obtain media through its original sources and follow their usage terms. To use a particular training mixture, supply its own manifest and corresponding media. [Open-o3 Video data preparation](https://github.com/marinero4972/Open-o3-Video#data-preparation) · [Open-o3 Video dataset](https://huggingface.co/datasets/marinero4972/Open-o3-Video)

## Records

Each record has a unique `id` and the fields `source`, `task`, `question`, and `answer`. The task string controls the student prompt and reward. The filenames below are examples. [Synthetic records](../configs/stgr_samples.json)

| `task` | Media and additional fields | Example `answer` field |
| --- | --- | --- |
| `visual QA` | `image_path`, `image_size` as `[width, height]` in pixels | `<obj>person</obj><box>[64, 48, 320, 432]</box>` |
| `temporal-spatial free-form QA` | `video_path`, `key_frames`, `key_items` | `The person picks up the cup.` |
| `temporal QA` | `video_path` | `[1.0, 3.0]` |
| `temporal QA (MCQ)` | `video_path` and answer options in `question` | `B\n[1.0, 3.0]` |
| `General video QA MCQ` | `video_path` and answer options in `question` | `B` |
| `General video QA Free-form` | `video_path` | `The person picks up the cup.` |

`answer` is a string, including when it encodes a time interval. For temporal-spatial rows, each `key_frames` entry has `path`, `time` in seconds, and `idx`. `key_items` maps the string form of `idx` to object names and each name to a list of normalized `[x1, y1, x2, y2]` boxes. The trainer constructs the task-specific `<think>` and `<answer>` prompt format. [Schema](../mesd/schema.py) · [Prompts](../mesd/prompts.py)

## Media paths

Set `data.media_root` to the directory containing `videos/`. The loader resolves each row's `image_path`, `video_path`, or keyframe `path` relative to its source-specific directory. Source prefixes below are case-sensitive.

| `source` | Image or video directory under `videos/` | Keyframe directory under `videos/` |
| --- | --- | --- |
| `gqa` | `gqa/` | — |
| `TreeVGR` | `treevgr/` | — |
| `videoespresso_train_video` | `videoespresso/videos/` | `videoespresso/kfs/` |
| `timerft` | `timerft/` | — |
| `STR_plm…` | `stgr/plm/videos/` | `stgr/plm/kfs/` |
| Other `STR…` sources | `stgr/temporal_grounding/videos/` | `stgr/temporal_grounding/kfs/` |
| `TVG…` | `tvg_r1/` | — |
| `videor1…` | `videor1/` | — |

## Optional frame cache

Build the cache with the same environment used for training. Set `data.stgr_frame_cache_dir` to its path, and set `data.stgr_frame_cache_required=true` if you want training to require cached samples. Use a cache directory visible to all Ray workers.

```bash
python tools/build_stgr_frame_cache.py /data/stgr/json_data/STGR-RL.json \
  --data-root /data/stgr --cache-dir /shared/recast/frame-cache
```
