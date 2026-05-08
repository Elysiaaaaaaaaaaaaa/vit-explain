import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from qwen2vl_rollout import Qwen2VLAttentionExtractor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="输入图像路径")
    parser.add_argument("--prompt", type=str, required=True, help="文本提示词")
    parser.add_argument(
        "--head_fusion",
        type=str,
        default="mean",
        choices=["mean", "max", "min"],
        help="多头融合方式",
    )
    parser.add_argument("--last_n_layers", type=int, default=4, help="最后 N 层做融合")
    parser.add_argument(
        "--query_mode",
        type=str,
        default="last",
        choices=["last", "keyword", "generated"],
        help="query token 选择模式",
    )
    parser.add_argument("--keyword", type=str, default=None, help="query_mode=keyword 时使用")
    parser.add_argument("--rollout", action="store_true", help="是否启用 rollout 连乘")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace 模型名称",
    )
    parser.add_argument("--output", type=str, default=None, help="输出热力图路径")
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=256 * 28 * 28,
        help="processor 最小像素预算，用于控制 token 数",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=512 * 28 * 28,
        help="processor 最大像素预算，用于控制 token 数",
    )
    return parser.parse_args()


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def main():
    args = get_args()
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"找不到图像: {image_path}")

    extractor = Qwen2VLAttentionExtractor(
        model_name=args.model_name,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    mask, meta = extractor.explain(
        image_path=str(image_path),
        prompt=args.prompt,
        query_mode=args.query_mode,
        keyword=args.keyword,
        head_fusion=args.head_fusion,
        last_n_layers=args.last_n_layers,
        rollout=True,
    )

    img_rgb = Image.open(image_path).convert("RGB")
    np_img = np.array(img_rgb)[:, :, ::-1]
    mask_up = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    cam = show_mask_on_image(np_img, mask_up)

    if args.output is None:
        stem = image_path.stem
        mode_name = args.query_mode
        out_name = f"{stem}_qwen2vl_attn_{mode_name}_{args.head_fusion}_L{args.last_n_layers}.png"
        out_path = Path.cwd() / out_name
    else:
        out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_path), cam)
    cv2.imwrite(str(Path.cwd() / "input_qwen2vl.png"), np_img)

    print("完成 Qwen2-VL attention 可视化")
    print(f"- 输出文件: {out_path}")
    print(f"- query_index: {meta.query_index}")
    print(f"- query_token: {meta.query_token}")
    print(f"- image_span: {meta.image_span}")
    print(f"- grid_hw: {meta.grid_hw}")
    if meta.generated_text:
        print(f"- generated_text: {meta.generated_text}")


if __name__ == "__main__":
    main()
