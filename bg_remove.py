#!/usr/bin/env python3
import argparse
import os
import sys
import glob
from typing import Optional, Tuple

import cv2
import numpy as np


def imread_color(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def save_rgba(path: str, bgr: np.ndarray, alpha: np.ndarray) -> None:
    if alpha.dtype != np.uint8:
        alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    ensure_dir(path)
    if not cv2.imwrite(path, bgra):
        raise IOError(f"Failed to save image: {path}")


def save_mask(path: str, mask: np.ndarray) -> None:
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    ensure_dir(path)
    if not cv2.imwrite(path, mask):
        raise IOError(f"Failed to save mask: {path}")


def refine_mask(mask: np.ndarray, open_size: int, close_size: int, blur: int) -> np.ndarray:
    """Refine a binary/soft mask with morphology and optional feathering.

    mask: uint8 [0..255]
    open_size/close_size: kernel size in pixels (0 to disable)
    blur: Gaussian kernel size (odd int, 0 to disable)
    """
    m = mask.copy()
    if open_size and open_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_size and close_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if blur and blur > 0:
        if blur % 2 == 0:
            blur += 1
        m = cv2.GaussianBlur(m, (blur, blur), 0)
    return m


def parse_rect(rect_str: str, img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    try:
        x, y, w, h = map(int, rect_str.split(','))
    except Exception as e:
        raise ValueError("--rect must be 'x,y,w,h'") from e
    H, W = img_shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError("rect width and height must be positive")
    if x < 0 or y < 0 or x + w > W or y + h > H:
        raise ValueError("rect must lie within the image bounds")
    return x, y, w, h


def auto_key_from_border_hsv(img_bgr: np.ndarray, border: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate HSV lower/upper for background by sampling image borders."""
    H, W = img_bgr.shape[:2]
    b = max(1, min(border, min(H, W)//4))
    border_pixels = np.concatenate([
        img_bgr[:b, :, :].reshape(-1, 3),
        img_bgr[-b:, :, :].reshape(-1, 3),
        img_bgr[:, :b, :].reshape(-1, 3),
        img_bgr[:, -b:, :].reshape(-1, 3),
    ], axis=0)
    hsv = cv2.cvtColor(border_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    # Robust range: use percentiles to reduce outlier impact
    low = np.percentile(hsv, 5, axis=0)
    high = np.percentile(hsv, 95, axis=0)
    # Expand slightly
    pad = np.array([10, 40, 40])
    lo = np.maximum(low - pad, [0, 0, 0])
    hi = np.minimum(high + pad, [179, 255, 255])
    return lo.astype(np.uint8), hi.astype(np.uint8)


def chroma_key_mask(img_bgr: np.ndarray,
                    hsv_lo: Optional[Tuple[int, int, int]] = None,
                    hsv_hi: Optional[Tuple[int, int, int]] = None,
                    auto_border: bool = False,
                    morph_open: int = 3,
                    morph_close: int = 7,
                    edge_blur: int = 5) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if auto_border or hsv_lo is None or hsv_hi is None:
        lo, hi = auto_key_from_border_hsv(img_bgr)
    else:
        lo = np.array(hsv_lo, dtype=np.uint8)
        hi = np.array(hsv_hi, dtype=np.uint8)

    bg_mask = cv2.inRange(hsv, lo, hi)  # 255 where background
    fg_mask = 255 - bg_mask
    fg_mask = refine_mask(fg_mask, morph_open, morph_close, edge_blur)
    return fg_mask


def grabcut_mask(img_bgr: np.ndarray,
                 rect: Optional[Tuple[int, int, int, int]] = None,
                 iters: int = 5,
                 morph_open: int = 3,
                 morph_close: int = 7,
                 edge_blur: int = 5) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    # Initialize mask with probable background
    mask = np.full((H, W), cv2.GC_PR_BGD, dtype=np.uint8)

    if rect is None:
        # Auto rectangle: shrink by 5% margin assuming subject is central
        mx = int(W * 0.05)
        my = int(H * 0.05)
        rect = (mx, my, max(1, W - 2*mx), max(1, H - 2*my))
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT)

    # Convert labels to mask: 1 or 3 are foreground/probable foreground
    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    m = refine_mask(m, morph_open, morph_close, edge_blur)
    return m


def compose_alpha(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Ensure mask in 0..255
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    return mask


def process_image(img_path: str,
                  out_dir: str,
                  method: str,
                  rect: Optional[Tuple[int, int, int, int]] = None,
                  hsv_lo: Optional[Tuple[int, int, int]] = None,
                  hsv_hi: Optional[Tuple[int, int, int]] = None,
                  auto_key: bool = False,
                  morph_open: int = 3,
                  morph_close: int = 7,
                  edge_blur: int = 5,
                  grabcut_iters: int = 5,
                  save_mask_path: Optional[str] = None) -> Tuple[str, Optional[str]]:
    img = imread_color(img_path)
    if method == 'grabcut':
        mask = grabcut_mask(img, rect, grabcut_iters, morph_open, morph_close, edge_blur)
    elif method == 'chroma':
        mask = chroma_key_mask(img, hsv_lo, hsv_hi, auto_key, morph_open, morph_close, edge_blur)
    else:
        raise ValueError(f"Unknown method: {method}")

    alpha = compose_alpha(img, mask)

    # Build output paths
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(out_dir, f"{base}_rgba.png")
    ensure_dir(out_path)
    save_rgba(out_path, img, alpha)

    saved_mask = None
    if save_mask_path:
        # If a directory provided, save as <base>_mask.png in that directory
        if os.path.isdir(save_mask_path) or save_mask_path.endswith(os.sep):
            mask_out = os.path.join(save_mask_path, f"{base}_mask.png")
        else:
            # Treat as file path if ends with .png, otherwise as directory
            if os.path.splitext(save_mask_path)[1].lower() == '.png' and not os.path.isdir(os.path.dirname(save_mask_path)):
                ensure_dir(save_mask_path)
                mask_out = save_mask_path
            else:
                ensure_dir(save_mask_path)
                mask_out = os.path.join(save_mask_path, f"{base}_mask.png")
        save_mask(mask_out, mask)
        saved_mask = mask_out

    return out_path, saved_mask


def expand_inputs(input_path: str) -> list:
    # Accept file, directory, or glob
    if any(ch in input_path for ch in '*?[]'):
        files = glob.glob(input_path)
        return [f for f in files if os.path.isfile(f)]
    if os.path.isdir(input_path):
        files = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.webp'):
            files.extend(glob.glob(os.path.join(input_path, ext)))
        return sorted(files)
    if os.path.isfile(input_path):
        return [input_path]
    raise FileNotFoundError(f"Input not found: {input_path}")


def parse_hsv_range(hsv_str: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if not hsv_str:
        return None
    try:
        parts = [int(p.strip()) for p in hsv_str.split(',')]
        if len(parts) != 3:
            raise ValueError
        h, s, v = parts
        h = int(np.clip(h, 0, 179))
        s = int(np.clip(s, 0, 255))
        v = int(np.clip(v, 0, 255))
        return (h, s, v)
    except Exception as e:
        raise ValueError("HSV must be 'H,S,V' with H in [0,179], S/V in [0,255]") from e


def main(argv=None):
    p = argparse.ArgumentParser(description="Background removal without AI: GrabCut and Chroma Key.")
    p.add_argument('-i', '--input', required=True, help='Input file, directory, or glob pattern')
    p.add_argument('-o', '--out-dir', default='out', help='Output directory for RGBA PNGs (default: out)')
    p.add_argument('-m', '--method', choices=['grabcut', 'chroma'], default='grabcut', help='Removal method')

    # GrabCut options
    p.add_argument('--rect', help="Foreground rectangle 'x,y,w,h' (if omitted, auto from central region)")
    p.add_argument('--grabcut-iters', type=int, default=5, help='GrabCut iterations (default: 5)')

    # Chroma options
    p.add_argument('--hsv-lo', help="HSV lower bound 'H,S,V' (0-179,0-255,0-255)")
    p.add_argument('--hsv-hi', help="HSV upper bound 'H,S,V' (0-179,0-255,0-255)")
    p.add_argument('--auto-key', action='store_true', help='Auto-estimate chroma key from image borders')

    # Refinement
    p.add_argument('--open', dest='morph_open', type=int, default=3, help='Morphological open kernel size (0 disable)')
    p.add_argument('--close', dest='morph_close', type=int, default=7, help='Morphological close kernel size (0 disable)')
    p.add_argument('--blur', dest='edge_blur', type=int, default=5, help='Gaussian blur for edge feather (0 disable)')

    # Extras
    p.add_argument('--save-mask', help='Directory or file path to also save raw mask as PNG')

    args = p.parse_args(argv)

    inputs = expand_inputs(args.input)
    if not inputs:
        print('No input images found.', file=sys.stderr)
        return 2

    rect = None
    if args.rect:
        # parse rect per-image later to validate against image dims; here just store string
        rect_str = args.rect
    else:
        rect_str = None

    hsv_lo = parse_hsv_range(args.hsv_lo)
    hsv_hi = parse_hsv_range(args.hsv_hi)

    # Process images
    ok = 0
    errors = []
    for path in inputs:
        try:
            img = imread_color(path)
            r = None
            if args.method == 'grabcut':
                if rect_str:
                    r = parse_rect(rect_str, img.shape)
            out_path, mask_path = process_image(
                img_path=path,
                out_dir=args.out_dir,
                method=args.method,
                rect=r,
                hsv_lo=hsv_lo,
                hsv_hi=hsv_hi,
                auto_key=args.auto_key,
                morph_open=args.morph_open,
                morph_close=args.morph_close,
                edge_blur=args.edge_blur,
                grabcut_iters=args.grabcut_iters,
                save_mask_path=args.save_mask,
            )
            print(f"Processed: {path} -> {out_path}" + (f" (mask: {mask_path})" if mask_path else ""))
            ok += 1
        except Exception as e:
            errors.append((path, str(e)))

    if errors:
        print("\nErrors:")
        for pth, msg in errors:
            print(f" - {pth}: {msg}")
    return 0 if ok > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
