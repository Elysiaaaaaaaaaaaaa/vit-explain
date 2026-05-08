import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None


@dataclass
class AttentionMeta:
    query_index: int
    query_token: str
    image_span: Tuple[int, int]
    grid_hw: Tuple[int, int]
    generated_text: str = ""


class Qwen2VLAttentionExtractor:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 512 * 28 * 28,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()

    def _build_messages(self, image_path: str, prompt: str) -> List[Dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _load_vision_inputs(self, messages: List[Dict]):
        if process_vision_info is not None:
            return process_vision_info(messages)
        image_path = messages[0]["content"][0]["image"]
        image = Image.open(image_path).convert("RGB")
        return [image], None

    def _prepare_inputs(self, image_path: str, prompt: str):
        messages = self._build_messages(image_path, prompt)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._load_vision_inputs(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.device)

    @staticmethod
    def _find_image_span(input_ids: torch.Tensor, image_token_id: int) -> Tuple[int, int]:
        idx = (input_ids == image_token_id).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            raise ValueError("未在 input_ids 中找到 image token，请检查输入模板。")
        return int(idx[0].item()), int(idx[-1].item() + 1)

    def _grid_from_inputs(self, inputs) -> Tuple[int, int]:
        if "image_grid_thw" not in inputs:
            raise ValueError("inputs 中没有 image_grid_thw，无法恢复二维网格。")
        thw = inputs["image_grid_thw"][0].tolist()
        _, h_patches, w_patches = [int(x) for x in thw]
        return h_patches // 2, w_patches // 2

    def _resolve_query_index(
        self,
        inputs,
        mode: str,
        keyword: Optional[str],
        image_span: Tuple[int, int],
    ) -> Tuple[int, str, Optional[Sequence[torch.Tensor]], str]:
        input_ids = inputs["input_ids"][0]
        special_ids = set(self.processor.tokenizer.all_special_ids)
        image_start, image_end = image_span

        if mode == "last":
            for i in range(input_ids.numel() - 1, -1, -1):
                tid = int(input_ids[i].item())
                if tid in special_ids:
                    continue
                if image_start <= i < image_end:
                    continue
                token = self.processor.tokenizer.decode([tid], skip_special_tokens=False)
                return i, token, None, ""
            raise ValueError("找不到可用的文本 query token。")

        if mode == "keyword":
            if not keyword:
                raise ValueError("query_mode=keyword 时必须传 --keyword。")
            keyword_ids = self.processor.tokenizer.encode(keyword, add_special_tokens=False)
            if not keyword_ids:
                raise ValueError("keyword 分词后为空，请换一个关键词。")
            seq = input_ids.tolist()
            k = len(keyword_ids)
            for i in range(len(seq) - k + 1):
                if seq[i : i + k] == keyword_ids and not (image_start <= i < image_end):
                    token = self.processor.tokenizer.decode(keyword_ids, skip_special_tokens=False)
                    return i, token, None, ""
            raise ValueError(f"未在输入 token 序列中找到关键词: {keyword}")

        if mode == "generated":
            gen_inputs = {k: v for k, v in inputs.items()}
            with torch.no_grad():
                gen_out = self.model.generate(
                    **gen_inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    output_attentions=True,
                    return_dict_in_generate=True,
                )
            if not gen_out.attentions:
                raise ValueError("generate 未返回 attentions，请确认 attn_implementation='eager'。")
            seq = gen_out.sequences[0]
            query_idx = int(seq.numel() - 1)
            token_id = int(seq[-1].item())
            token = self.processor.tokenizer.decode([token_id], skip_special_tokens=False)
            text = self.processor.tokenizer.decode(
                seq[inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            return query_idx, token, gen_out.attentions[0], text

        raise ValueError(f"不支持的 query_mode: {mode}")

    @staticmethod
    def _fuse_heads(attn: torch.Tensor, head_fusion: str) -> torch.Tensor:
        if head_fusion == "mean":
            return attn.mean(dim=1)
        if head_fusion == "max":
            return attn.max(dim=1).values
        if head_fusion == "min":
            return attn.min(dim=1).values
        raise ValueError("head_fusion 必须是 mean/max/min。")

    def _extract_row_from_forward_attn(
        self,
        attentions: Sequence[torch.Tensor],
        query_idx: int,
        image_span: Tuple[int, int],
        last_n_layers: int,
        head_fusion: str,
        use_rollout: bool,
    ) -> torch.Tensor:
        n = max(1, min(last_n_layers, len(attentions)))
        layer_attn = torch.stack(attentions[-n:], dim=0).float()  # (N, B, H, S, S)

        if not use_rollout:
            fused_per_layer = self._fuse_heads(layer_attn, head_fusion)  # (N, B, S, S)
            fused = fused_per_layer.mean(dim=0)  # (B, S, S)
            row = fused[0, query_idx, image_span[0] : image_span[1]]
            return row

        s = layer_attn.shape[-1]
        result = torch.eye(s, device=layer_attn.device, dtype=layer_attn.dtype)
        for li in range(layer_attn.shape[0]):
            a = self._fuse_heads(layer_attn[li], head_fusion)[0]  # (S, S)
            a = (a + torch.eye(s, device=a.device, dtype=a.dtype)) / 2.0
            a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            result = a @ result
        return result[query_idx, image_span[0] : image_span[1]]

    @staticmethod
    def _extract_row_from_generate_attn(
        gen_step_attn: Sequence[torch.Tensor],
        image_span: Tuple[int, int],
        head_fusion: str,
        last_n_layers: int,
    ) -> torch.Tensor:
        n = max(1, min(last_n_layers, len(gen_step_attn)))
        # generate 返回每层形状通常是 (B, H, 1, S_prev)
        selected = [a.float() for a in gen_step_attn[-n:]]
        stacked = torch.stack(selected, dim=0)  # (N, B, H, 1, S_prev)
        if head_fusion == "mean":
            fused = stacked.mean(dim=2)
        elif head_fusion == "max":
            fused = stacked.max(dim=2).values
        elif head_fusion == "min":
            fused = stacked.min(dim=2).values
        else:
            raise ValueError("head_fusion 必须是 mean/max/min。")
        fused = fused.mean(dim=0)  # (B, 1, S_prev)
        row = fused[0, 0, image_span[0] : image_span[1]]
        return row

    def explain(
        self,
        image_path: str,
        prompt: str,
        query_mode: str = "last",
        keyword: Optional[str] = None,
        head_fusion: str = "mean",
        last_n_layers: int = 4,
        rollout: bool = False,
    ) -> Tuple[np.ndarray, AttentionMeta]:
        inputs = self._prepare_inputs(image_path=image_path, prompt=prompt)
        image_token_id = int(self.model.config.image_token_id)
        image_span = self._find_image_span(inputs["input_ids"][0], image_token_id)
        grid_h, grid_w = self._grid_from_inputs(inputs)

        query_idx, query_token, gen_attn, generated_text = self._resolve_query_index(
            inputs=inputs,
            mode=query_mode,
            keyword=keyword,
            image_span=image_span,
        )

        print("[model output]"+generated_text)
        print("[gen_attn type]", type(gen_attn))
        print("[gen_attn len]", len(gen_attn))
        if isinstance(gen_attn, tuple) and len(gen_attn) > 0:
            print("[num_layers]", len(gen_attn))
            print("[gen_attn[0] type]", type(gen_attn[0]))
            if isinstance(gen_attn[0], tuple) and len(gen_attn[0]) > 0:
                print("[steps_per_layer]", len(gen_attn[0]))
                step = gen_attn[0][-1]
                print("[gen_attn[0][last_step] shape]", step.shape)
                print("[gen_attn[0][last_step]", step)

        if query_mode == "generated":
            row = self._extract_row_from_generate_attn(
                gen_step_attn=gen_attn,
                image_span=image_span,
                head_fusion=head_fusion,
                last_n_layers=last_n_layers,
            )
        else:
            with torch.no_grad():
                out = self.model(**inputs, output_attentions=True, return_dict=True)
            row = self._extract_row_from_forward_attn(
                attentions=out.attentions,
                query_idx=query_idx,
                image_span=image_span,
                last_n_layers=last_n_layers,
                head_fusion=head_fusion,
                use_rollout=rollout,
            )

        expected_tokens = grid_h * grid_w
        if row.numel() != expected_tokens:
            # 某些版本 image_span 可能比 grid 更长，优先裁到匹配长度。
            row = row[:expected_tokens]
            if row.numel() != expected_tokens:
                raise ValueError(
                    f"image token 数量和 grid 不匹配: {row.numel()} vs {expected_tokens}."
                )

        mask = row.reshape(grid_h, grid_w).detach().cpu().numpy()
        denom = float(np.max(mask))
        if denom <= 0:
            denom = 1.0
        mask = mask / denom

        meta = AttentionMeta(
            query_index=query_idx,
            query_token=query_token,
            image_span=image_span,
            grid_hw=(grid_h, grid_w),
            generated_text=generated_text,
        )
        return mask, meta
