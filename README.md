# RECAST

**Training-Based Recursive Self-Improvement in VideoLLM**

RECAST alternates task-reward updates with self-teacher-guided updates. Future-Aware Semantic Credit (FASC) redistributes token advantages using later semantic chunks. Outcome-Grounded Decision Refinement (ODR) scores candidate tokens by sampling their continuations.

## Training efficiency

RECAST uses vLLM 0.11.0 to generate batched student rollouts and ODR continuations. PyTorch FSDP updates the model, and Ray coordinates the distributed workers.

On 32 NVIDIA A800 GPUs, one epoch on 37K STGR-RL examples took **10.93 hours** over 582 rollout steps (67.61 s/step). The paper's 100K run took **31.92 hours** over 1,563 rollout steps (73.53 s/step). These measurements exclude checkpoint saving.

## Install

Use Linux with NVIDIA GPUs, CUDA, NCCL, and Python 3.11. Install the same environment on each node. Building FlashAttention requires a matching CUDA toolkit and a C++ compiler.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install ninja==1.13.0
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

## Train

Prepare [Open-o3-Video-SFT-7B](https://huggingface.co/marinero4972/Open-o3-Video-SFT-7B) and an STGR annotation file with its media. The [data guide](docs/data.md) documents the layout and optional frame cache.

The [default recipe](configs/train_video_trsi_fast.yaml) uses four nodes with eight GPUs each, 64 prompts per rollout batch, and four responses per prompt. Use the same absolute paths for the checkout, environment, model, data, and shared output on all nodes.

Start Ray on the head with `ray start --head --port=6379`. On each worker, run `ray start --address=HEAD_IP:6379` with the head's IP. Launch training from the head:

```bash
RAY_ADDRESS=auto bash scripts/train.sh \
  data.train_files=/data/stgr/json_data/STGR-RL.json \
  data.media_root=/data/stgr \
  worker.actor.model.model_path=/models/Open-o3-Video-SFT-7B \
  trainer.save_checkpoint_path=/shared/recast/full
```

For a standalone eight-GPU node, unset `RAY_ADDRESS` and add `trainer.nnodes=1 trainer.n_gpus_per_node=8`. This keeps the batch layout compatible. GPU memory use depends on the model and sequence lengths.

For a short training run, add `--config configs/train_video_trsi_fast_smoke.yaml`, use a separate output directory, and supply at least 512 annotated records with media for eight full batches. The [short-run configuration](configs/train_video_trsi_fast_smoke.yaml) covers both training phases.

## Checkpoints

Use `trainer.load_checkpoint_path=/path/to/run/global_step_N` with the original config and GPU topology to resume. Choose a new output directory for each fresh run. TensorBoard logs default to `tensorboard_log/`.

Export a completed FSDP actor checkpoint to Hugging Face format:

```bash
python scripts/model_merger.py --local_dir /shared/recast/full/global_step_N/actor
```

## Acknowledgments

RECAST shares evidence-conditioned prompt and training utilities with MeSD. [MeSD code](https://github.com/WhiteVentral/MeSD-code)

The training runtime builds on RLSD and veRL. The STGR adapters build on Open-o3 Video, and the reward and teacher design draws on VISD. [LICENSE](LICENSE) · [NOTICE](NOTICE)
