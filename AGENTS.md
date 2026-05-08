# AGENTS.md
please chat with user in chinese
## Two separate workflows (no shared entrypoint)

- **DeiT (ViT) attention rollout** — `vit_explain.py` (CLI), driven by `vit_rollout.py` + `vit_grad_rollout.py` 现在暂时不需要管这条链路。
- **Qwen2-VL attention visualization** — `qwen2vl_explain.py` (CLI), driven by `qwen2vl_rollout.py`

## 运行说明
不要在本地运行，本地没有GPU支持ai推理。

## DeviT (ViT) commands

```bash
# Attention Rollout (no class index → no gradient)
python vit_explain.py --image_path <path> --head_fusion max --discard_ratio 0.9

# Gradient Attention Rollout (class-specific, e.g. dog=243)
python vit_explain.py --image_path <path> --head_fusion max --discard_ratio 0.9 --category_index 243
```

The script always loads `deit_tiny_patch16_224` from torch hub. It uses `MVITAttentionRollout` (monkey-patch `timm.models.vision_transformer.Attention.forward`) for standard rollout, and `VITAttentionGradRollout` (forward+backward hooks on `attn_drop`) for gradient rollout.

## Qwen2-VL commands

```bash
python qwen2vl_explain.py \
  --image_path "./photoes/R.jpg" \
  --prompt "图中最显著的目标是什么？" \
  --query_mode last \
  --head_fusion mean \
  --last_n_layers 4
```

Or just `run.bat` (same command). Main args: `--query_mode` (`last`/`keyword`/`generated`), `--keyword` (required with `keyword` mode), `--rollout` (optional rollout-style layer composition), `--min_pixels`/`--max_pixels` (control image token budget).

## Critical model loading quirks

1. `Qwen2VLAttentionExtractor` **must** load with `attn_implementation="eager"` or attention weights won't be returned. Hardcoded in `qwen2vl_rollout.py:42`.
2. **dtype 必须为 `torch.float32`**（`qwen2vl_rollout.py:34`）。eager attention 在 float16 下数值不稳定，大量 visual token 会导致 softmax 下溢 → 模型输出乱码。已硬编码为 float32（不再根据 GPU/CPU 条件切换）。

## generated 模式注意事项

`generated` 模式使用 `gen_out.attentions[-1]`（最后一个生成步）提取 attention（`qwen2vl_rollout.py:148`）。第一步 attention 模型尚未聚焦图像，热力图会平坦无区分度。若需调试，`qwen2vl_rollout.py:275` 有 `[mask]` 日志输出 max/min/mean。

## Dependencies

All in `requirements.txt`. Key ones: `torch`, `torchvision`, `transformers>=4.45.0`, `accelerate`, `qwen-vl-utils`, `opencv-python`, `timm` (note: `timm` is imported by `vit_rollout.py` and `vit_grad_rollout.py` but NOT listed in `requirements.txt` — install it manually).

## Architecture notes

- `vit_rollout.py`: `MVITAttentionRollout` replaces `Attention.forward` via monkey-patching (the default used by `vit_explain.py`). `VITAttentionRollout` uses forward hooks on `attn_drop` (legacy, not used by default).
- `vit_grad_rollout.py`: Uses forward+backward hooks on `attn_drop` layers. Gradients multiply attention to isolate class-relevant flow.
- `qwen2vl_rollout.py`: Full extractor for Qwen2-VL. Does a forward pass with `output_attentions=True` (or `generate` with `return_dict_in_generate=True` for `generated` mode). Resolves query token position in the sequence, extracts attention row for image tokens, reshapes to 2D heatmap.
- `test_hook.py`: Minimal standalone test verifying PyTorch forward hooks fire in both train/eval modes. Not part of any pipeline.

## Output files

- `vit_explain.py` writes `input.png` and `attention_rollout_*.png` or `grad_rollout_*.png` to cwd.
- `qwen2vl_explain.py` writes `input_qwen2vl.png` and `<stem>_qwen2vl_attn_*.png` next to the input image, or to `--output`.
- `heatmaps/` directory exists but is empty — used as output target for some runs.

## No tests, no lint, no CI

This is a research/exploration repo. There is no test suite, no formatter config, no CI. `test_hook.py` is the only test-like file and it's a standalone sanity check.
