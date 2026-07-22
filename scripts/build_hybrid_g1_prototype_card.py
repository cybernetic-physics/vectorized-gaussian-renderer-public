#!/usr/bin/env python3
"""Build a labeled, non-composited custom-Gaussian plus RTX-G1 evidence card."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussian-poster", type=Path, required=True)
    parser.add_argument("--g1-rtx-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (20, 23, 28))
    offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def main() -> None:
    args = parse_args()
    poster = Image.open(args.gaussian_poster).convert("RGB")
    custom = poster.crop((0, 0, poster.width // 2, poster.height))
    rtx = Image.open(args.g1_rtx_image).convert("RGB")

    panel_size = (560, 440)
    card = Image.new("RGB", (1160, 610), (10, 12, 16))
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default()
    title = "Home Scan + Unitree G1 rendering boundary"
    subtitle = "Two separate GPU outputs — not a completed depth/alpha composite"
    draw.text((30, 24), title, fill=(240, 243, 248), font=font)
    draw.text((30, 48), subtitle, fill=(176, 188, 204), font=font)
    card.paste(fit(custom, panel_size), (20, 100))
    card.paste(fit(rtx, panel_size), (580, 100))
    draw.text((30, 555), "Custom CUDA Gaussian renderer: static Home Scan RGB/depth/alpha tensors", fill=(203, 215, 230), font=font)
    draw.text((590, 555), "Isaac Sim RTX: imported Unitree G1 USD mesh (default pose)", fill=(203, 215, 230), font=font)
    draw.text((30, 578), "Missing: GPU-resident depth-aware compositor and Gaussian collision proxy validation.", fill=(250, 184, 103), font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    card.save(args.output)


if __name__ == "__main__":
    main()
