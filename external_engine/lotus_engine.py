#!/usr/bin/env python3
"""
LOTUS standalone engine for Colab / local: depth + learned normals, EXR + optional MP4.
Expects LOT repo root on sys.path (pipeline.py, utils).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# HF model ids (Lotus-D regression, same family as LOT/app.py defaults)
DEFAULT_DEPTH_MODEL_D = "jingheya/lotus-depth-d-v2-0-disparity"
DEFAULT_NORMAL_MODEL_D = "jingheya/lotus-normal-d-v1-0"


def _resolve_auto_processing_res(processing_res: Optional[int]) -> Optional[int]:
    """-1 means choose the highest quality setting this GPU is likely to handle."""
    if processing_res != -1:
        return processing_res
    if not torch.cuda.is_available():
        print("[LOTUS] Auto quality: CUDA unavailable, using 768 processing resolution")
        return 768
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    if vram_gb >= 40.0:
        chosen = 0  # Native LOTUS processing; outputs still match input res.
    elif vram_gb >= 24.0:
        chosen = 2048
    elif vram_gb >= 16.0:
        chosen = 1536
    elif vram_gb >= 10.0:
        chosen = 1024
    else:
        chosen = 768
    print(f"[LOTUS] Auto quality: GPU VRAM {vram_gb:.1f} GB -> processing_res={chosen}")
    return chosen


def _task_embedding(device: torch.device) -> torch.Tensor:
    task_emb = torch.tensor([[1.0, 0.0]], device=device, dtype=torch.float32)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
    return task_emb


def _load_rgb_frame(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        from PIL import Image

        with Image.open(path) as im:
            return np.array(im.convert("RGB"))
    except Exception as exc:
        raise FileNotFoundError(path) from exc


def _build_frame_path_from_pattern(pattern: str, frame_num: int) -> str:
    return re.sub(
        r"%0?(\d+)d",
        lambda m: f"{frame_num:0{int(m.group(1))}d}",
        pattern,
    )


def _discover_frame_numbers(pattern: str) -> List[int]:
    """List frame indices by scanning the pattern's directory (handles gaps / wrong batch range)."""
    directory = os.path.dirname(pattern) or "."
    basename = os.path.basename(pattern)
    m = re.search(r"%0(\d+)d", basename)
    if not m:
        m = re.search(r"%(\d+)d", basename)
    if not m:
        return []
    token = m.group(0)
    prefix, suffix = basename.split(token, 1)
    if suffix is None:
        return []
    rx = re.compile("^" + re.escape(prefix) + r"(\d+)" + re.escape(suffix) + "$", re.IGNORECASE)
    try:
        names = os.listdir(directory)
    except (FileNotFoundError, NotADirectoryError):
        return []
    nums: List[int] = []
    for name in names:
        mm = rx.match(name)
        if mm:
            nums.append(int(mm.group(1)))
    return sorted(set(nums))


def _resolve_frame_numbers(pattern: str, first: int, last: int) -> List[int]:
    """
    Build the list of frames to process. Uses dense [first, last] when every file exists; otherwise
    falls back to listing the folder so JPG/PNG plates work with missing frames, Drive sync gaps,
    or batch JSON that does not match on-disk numbering.
    """
    dense = list(range(first, last + 1))
    existing = [n for n in dense if os.path.exists(_build_frame_path_from_pattern(pattern, n))]
    if len(existing) == len(dense):
        return dense
    if existing:
        missing = len(dense) - len(existing)
        print(
            f"[LOTUS] Warning: {missing} missing file(s) in range [{first},{last}]; "
            f"processing {len(existing)} existing frame(s)."
        )
        return existing
    discovered = _discover_frame_numbers(pattern)
    if not discovered:
        sample = _build_frame_path_from_pattern(pattern, first)
        raise FileNotFoundError(
            f"No input frames found for pattern {pattern!r} (tried {sample!r} and directory listing)."
        )
    in_range = [n for n in discovered if first <= n <= last]
    if in_range:
        if len(in_range) != len(dense):
            print(
                f"[LOTUS] Warning: batch range [{first},{last}] has {len(dense)} steps but only "
                f"{len(in_range)} files on disk; processing those."
            )
        return in_range
    print(
        f"[LOTUS] Warning: no frames in [{first},{last}] on disk; using {len(discovered)} files "
        f"actually present ({discovered[0]}–{discovered[-1]})."
    )
    return discovered


def _read_frame_from_pattern(pattern: str, frame_num: int, padding: int) -> np.ndarray:
    frame_path = _build_frame_path_from_pattern(pattern, frame_num)
    lower = frame_path.lower()
    if lower.endswith(".exr"):
        if not os.path.exists(frame_path):
            raise FileNotFoundError(frame_path)
        try:
            import Imath
            import OpenEXR
        except ImportError:
            raise RuntimeError(
                "EXR input requires OpenEXR. Install OpenEXR or use JPEG/PNG plates."
            ) from None
        exr = OpenEXR.InputFile(frame_path)
        dw = exr.header()["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        channels = list(exr.header()["channels"].keys())
        ch0 = "R" if "R" in channels else channels[0]
        extype = Imath.PixelType(Imath.PixelType.FLOAT)
        r = np.frombuffer(exr.channel(ch0, extype), dtype=np.float32).reshape((h, w))
        if "G" in channels and "B" in channels:
            g = np.frombuffer(exr.channel("G", extype), dtype=np.float32).reshape((h, w))
            b = np.frombuffer(exr.channel("B", extype), dtype=np.float32).reshape((h, w))
            rgb = np.stack([r, g, b], axis=-1)
        else:
            rgb = np.stack([r, r, r], axis=-1)
        rgb = np.clip(rgb, 0.0, None)
        mx = float(rgb.max()) if rgb.size else 1.0
        if mx > 1.0:
            rgb = rgb / (mx + 1e-8)
        return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    if not os.path.exists(frame_path):
        raise FileNotFoundError(frame_path)
    return _load_rgb_frame(frame_path)


def _save_depth_exr(path: str, depth_hw: np.ndarray, floating_point: str = "float32") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = depth_hw.astype(np.float32) if floating_point == "float32" else depth_hw.astype(np.float16)
    ok = cv2.imwrite(path, d)
    if not ok:
        try:
            import OpenEXR
            import Imath
        except ImportError:
            raise RuntimeError("Failed to write depth EXR and OpenEXR not available") from None
        h, w = d.shape[:2]
        hdr = OpenEXR.Header(w, h)
        hdr["channels"] = {"R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
        out = OpenEXR.OutputFile(path, hdr)
        out.writePixels({"R": d.astype(np.float32)})
        out.close()


def _save_normal_exr(path: str, rgb01: np.ndarray, normal_exr_range: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if normal_exr_range == "neg1_1":
        data = rgb01.astype(np.float32) * 2.0 - 1.0
        lo, hi = -1.0, 1.0
    else:
        data = rgb01.astype(np.float32)
        lo, hi = 0.0, 1.0
    data = np.clip(data, lo, hi).astype(np.float32)
    bgr = data[..., ::-1].copy()
    cv2.imwrite(path, bgr)


def _stabilize_depth(prev: Optional[np.ndarray], curr: np.ndarray, ema: float) -> np.ndarray:
    if prev is None or ema <= 0.0:
        return curr
    return ema * curr + (1.0 - ema) * prev


def _stabilize_normal(
    prev: Optional[np.ndarray],
    curr01: np.ndarray,
    ema: float,
) -> np.ndarray:
    if prev is None or ema <= 0.0:
        return curr01
    v0 = prev * 2.0 - 1.0
    v1 = curr01 * 2.0 - 1.0
    v = ema * v1 + (1.0 - ema) * v0
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-8)
    v = v / n
    return np.clip((v + 1.0) * 0.5, 0.0, 1.0)


def _normalize_normal01(normal01: np.ndarray) -> np.ndarray:
    v = normal01.astype(np.float32) * 2.0 - 1.0
    mag = np.linalg.norm(v, axis=-1, keepdims=True)
    v = v / np.maximum(mag, 1e-8)
    return np.clip((v + 1.0) * 0.5, 0.0, 1.0)


def _warp_scalar_backward(prev_hw: np.ndarray, backward_flow: np.ndarray) -> np.ndarray:
    """Warp previous scalar frame into current coordinates using current -> previous flow."""
    assert prev_hw.ndim == 2
    assert backward_flow.shape[0] == 2
    h, w = prev_hw.shape
    prev_t = torch.from_numpy(prev_hw.astype(np.float32, copy=False))[None, None]
    flow_t = torch.from_numpy(backward_flow.astype(np.float32, copy=False))
    y_coords = torch.arange(h, dtype=torch.float32)
    x_coords = torch.arange(w, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    src_x = xx + flow_t[0]
    src_y = yy + flow_t[1]
    grid = torch.stack(
        [
            src_x / max(w - 1, 1) * 2.0 - 1.0,
            src_y / max(h - 1, 1) * 2.0 - 1.0,
        ],
        dim=-1,
    )[None]
    warped = F.grid_sample(prev_t, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return warped[0, 0].numpy().astype(np.float32, copy=False)


def _warp_rgb_backward(prev_hwc: np.ndarray, backward_flow: np.ndarray) -> np.ndarray:
    chans = [
        _warp_scalar_backward(prev_hwc[..., idx], backward_flow)
        for idx in range(prev_hwc.shape[-1])
    ]
    return np.stack(chans, axis=-1).astype(np.float32, copy=False)


def _fb_occlusion_mask(
    forward_prev: np.ndarray,
    backward_cur: np.ndarray,
    threshold_px: float,
) -> np.ndarray:
    """Return [0,1] occlusion mask in current-frame coords; 1 rejects smoothing."""
    assert forward_prev.shape == backward_cur.shape
    h, w = backward_cur.shape[1:]
    fp = torch.from_numpy(forward_prev.astype(np.float32, copy=False))[None]
    bc = torch.from_numpy(backward_cur.astype(np.float32, copy=False))[None]
    y_coords = torch.arange(h, dtype=torch.float32)
    x_coords = torch.arange(w, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    src_x = xx + bc[0, 0]
    src_y = yy + bc[0, 1]
    grid = torch.stack(
        [
            src_x / max(w - 1, 1) * 2.0 - 1.0,
            src_y / max(h - 1, 1) * 2.0 - 1.0,
        ],
        dim=-1,
    )[None]
    fwd_at_src = F.grid_sample(fp, grid, mode="bilinear", padding_mode="border", align_corners=True)
    residual = bc + fwd_at_src
    err = torch.sqrt(residual[0, 0] ** 2 + residual[0, 1] ** 2)
    return torch.clamp(err / max(float(threshold_px), 1e-6), 0.0, 1.0).numpy().astype(np.float32)


def _resize_rgb_for_raft(rgb: np.ndarray, max_side: int) -> tuple[torch.Tensor, float, float]:
    h, w = rgb.shape[:2]
    if max_side and max(h, w) > max_side:
        scale = max_side / float(max(h, w))
    else:
        scale = 1.0
    rh = max(8, int(round(h * scale / 8.0)) * 8)
    rw = max(8, int(round(w * scale / 8.0)) * 8)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255.0
    if (rh, rw) != (h, w):
        tensor = F.interpolate(tensor, size=(rh, rw), mode="bilinear", align_corners=False)
    return tensor, w / float(rw), h / float(rh)


class LOTUSDepthNormalEngine:
    """Load Lotus-D pipelines and run a frame range from an image sequence pattern."""

    def __init__(self, lotus_root: Optional[str] = None, lot_root: Optional[str] = None):
        # `lot_root` is accepted for older Colab cellcode compatibility.
        root = lotus_root if lotus_root is not None else lot_root
        self.lotus_root = Path(root) if root else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe_depth = None
        self._pipe_normal = None
        self._raft_model = None
        self._raft_transforms = None

    def _ensure_lotus_imports(self) -> None:
        if self.lotus_root and str(self.lotus_root) not in sys.path:
            sys.path.insert(0, str(self.lotus_root))

    def load_models(
        self,
        generate_depth: bool,
        generate_normals: bool,
        variant: str = "d",
        half_precision: bool = True,
        depth_model_id: Optional[str] = None,
        normal_model_id: Optional[str] = None,
    ) -> None:
        if variant.lower() != "d":
            raise ValueError("Standalone test supports lotus_variant=d (regression) only for now.")
        self._ensure_lotus_imports()
        from pipeline import LotusDPipeline

        dtype = torch.float16 if half_precision and self.device.type == "cuda" else torch.float32
        if generate_depth:
            mid = depth_model_id or DEFAULT_DEPTH_MODEL_D
            self._pipe_depth = LotusDPipeline.from_pretrained(mid, torch_dtype=dtype)
            self._pipe_depth = self._pipe_depth.to(self.device)
            self._pipe_depth.set_progress_bar_config(disable=True)
        if generate_normals:
            mid = normal_model_id or DEFAULT_NORMAL_MODEL_D
            self._pipe_normal = LotusDPipeline.from_pretrained(mid, torch_dtype=dtype)
            self._pipe_normal = self._pipe_normal.to(self.device)
            self._pipe_normal.set_progress_bar_config(disable=True)

    def _run_frame(
        self,
        pipe,
        rgb_uint8: np.ndarray,
        timestep: int,
        processing_res: Optional[int],
        match_input_res: bool,
        generator: Optional[torch.Generator],
    ) -> np.ndarray:
        x = rgb_uint8.astype(np.float32)
        x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
        x = x / 127.5 - 1.0
        task_emb = _task_embedding(self.device)
        if torch.backends.mps.is_available():
            ctx = nullcontext()
        else:
            ctx = torch.autocast(self.device.type, enabled=self.device.type == "cuda")
        with torch.no_grad():
            with ctx:
                pred = pipe(
                    rgb_in=x,
                    prompt="",
                    num_inference_steps=1,
                    generator=generator,
                    output_type="np",
                    timesteps=[timestep],
                    task_emb=task_emb,
                    processing_res=processing_res,
                    match_input_res=match_input_res,
                    resample_method="bilinear",
                ).images[0]
        return pred

    def _load_raft(self, backend: str = "raft_small") -> None:
        if self._raft_model is not None:
            return
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )

        if backend == "raft_large":
            weights = Raft_Large_Weights.DEFAULT
            model = raft_large(weights=weights, progress=False)
        elif backend == "raft_small":
            weights = Raft_Small_Weights.DEFAULT
            model = raft_small(weights=weights, progress=False)
        else:
            raise ValueError(f"Unknown RAFT backend: {backend!r}")
        self._raft_transforms = weights.transforms()
        self._raft_model = model.to(self.device).eval()

    def _compute_raft_flow(
        self,
        source_rgb: np.ndarray,
        target_rgb: np.ndarray,
        backend: str,
        max_side: int,
        num_flow_updates: int,
    ) -> np.ndarray:
        """Compute optical flow source -> target in original-resolution pixels."""
        self._load_raft(backend)
        src, sx, sy = _resize_rgb_for_raft(source_rgb, max_side)
        tgt, _, _ = _resize_rgb_for_raft(target_rgb, max_side)
        src, tgt = self._raft_transforms(src, tgt)
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        with torch.no_grad():
            preds = self._raft_model(src, tgt, num_flow_updates=int(num_flow_updates))
        flow = preds[-1]
        h, w = source_rgb.shape[:2]
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)[0]
        flow[0] *= sx
        flow[1] *= sy
        return flow.detach().cpu().numpy().astype(np.float32, copy=False)

    def _raft_temporal_weight(
        self,
        prev_rgb: Optional[np.ndarray],
        curr_rgb: np.ndarray,
        backend: str,
        max_side: int,
        num_flow_updates: int,
        fb_threshold_px: float,
        alpha: float,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if prev_rgb is None or alpha <= 0.0:
            return None, None
        backward_cur = self._compute_raft_flow(
            curr_rgb,
            prev_rgb,
            backend,
            max_side,
            num_flow_updates,
        )
        forward_prev = self._compute_raft_flow(
            prev_rgb,
            curr_rgb,
            backend,
            max_side,
            num_flow_updates,
        )
        occlusion = _fb_occlusion_mask(forward_prev, backward_cur, fb_threshold_px)
        weight = (1.0 - occlusion) * float(alpha)
        return backward_cur, np.clip(weight, 0.0, 1.0).astype(np.float32)

    def process_sequence(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        pattern = job_data["input_video"]
        first = int(job_data["first_frame"])
        last = int(job_data["last_frame"])
        pad = int(job_data.get("frame_padding", 4))
        gen_depth = bool(job_data.get("generate_lotus_depth", True))
        gen_norm = bool(job_data.get("generate_lotus_normals", True))
        if not gen_depth and not gen_norm:
            return {"status": "error", "message": "Nothing to generate"}

        depth_exr_dir = job_data["lotus_depth_exr_dir_local"]
        norm_exr_dir = job_data["normal_exr_dir_local"]
        depth_mp4_dir = job_data.get("lotus_depth_mp4_dir_local")
        norm_mp4_dir = job_data.get("normal_mp4_dir_local")
        processing_res = job_data.get("lotus_processing_res")
        if processing_res is not None:
            processing_res = int(processing_res)
        processing_res = _resolve_auto_processing_res(processing_res)
        match_input_res = True
        timestep = int(job_data.get("timestep", 999))
        seed = job_data.get("lotus_seed")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        floating_point = job_data.get("floating_point", "float32")
        normal_range = job_data.get("normal_exr_range", "0_1")
        disparity_vis = bool(job_data.get("lotus_depth_disparity_vis", True))
        stabilize = bool(job_data.get("lotus_temporal_stabilization", True))
        ema = float(job_data.get("lotus_temporal_ema", 0.35))
        temporal_mode = str(job_data.get("lotus_temporal_mode", "ema")).lower()
        use_raft = stabilize and temporal_mode == "raft"
        raft_failed = False
        raft_backend = str(job_data.get("raft_backend", "raft_small"))
        raft_max_side = int(job_data.get("raft_inference_resolution", 520))
        raft_updates = int(job_data.get("raft_num_flow_updates", 12))
        raft_threshold = float(job_data.get("raft_fb_threshold_px", 1.0))
        raft_alpha = float(job_data.get("raft_alpha", ema))

        from utils.image_utils import colorize_depth_map

        os.makedirs(depth_exr_dir, exist_ok=True)
        if gen_norm:
            os.makedirs(norm_exr_dir, exist_ok=True)

        depth_vis_frames: List[np.ndarray] = []
        norm_vis_frames: List[np.ndarray] = []

        prev_d: Optional[np.ndarray] = None
        prev_n: Optional[np.ndarray] = None
        prev_rgb: Optional[np.ndarray] = None
        depth_count = 0
        norm_count = 0

        frame_numbers = _resolve_frame_numbers(pattern, first, last)
        for frame_num in tqdm(frame_numbers, desc="LOTUS frames"):
            rgb = _read_frame_from_pattern(pattern, frame_num, pad)
            backward_flow = None
            temporal_weight = None
            if use_raft and not raft_failed and (gen_depth or gen_norm):
                try:
                    backward_flow, temporal_weight = self._raft_temporal_weight(
                        prev_rgb,
                        rgb,
                        raft_backend,
                        raft_max_side,
                        raft_updates,
                        raft_threshold,
                        raft_alpha,
                    )
                except Exception as exc:
                    raft_failed = True
                    print(
                        "[WARN] RAFT temporal smoothing failed; falling back to normal EMA temporal. "
                        f"{type(exc).__name__}: {exc}"
                    )
            if gen_depth and self._pipe_depth is not None:
                pred = self._run_frame(
                    self._pipe_depth,
                    rgb,
                    timestep,
                    processing_res,
                    match_input_res,
                    generator,
                )
                dep = np.asarray(pred.mean(axis=-1), dtype=np.float32)
                if stabilize:
                    if use_raft and prev_d is not None and backward_flow is not None and temporal_weight is not None:
                        warped = _warp_scalar_backward(prev_d, backward_flow)
                        dep = temporal_weight * warped + (1.0 - temporal_weight) * dep
                    else:
                        dep = _stabilize_depth(prev_d, dep, ema)
                    prev_d = dep
                exr_path = os.path.join(depth_exr_dir, f"lotus_depth.{frame_num:04d}.exr")
                _save_depth_exr(exr_path, dep, floating_point)
                depth_count += 1
                vis = colorize_depth_map(dep, reverse_color=disparity_vis)
                depth_vis_frames.append(np.array(vis.convert("RGB")))
            if gen_norm and self._pipe_normal is not None:
                pred = self._run_frame(
                    self._pipe_normal,
                    rgb,
                    timestep,
                    processing_res,
                    match_input_res,
                    generator,
                )
                n01 = np.clip(pred.astype(np.float32), 0.0, 1.0)
                if stabilize:
                    if use_raft and prev_n is not None and backward_flow is not None and temporal_weight is not None:
                        warped = _warp_rgb_backward(prev_n, backward_flow)
                        n01 = temporal_weight[..., None] * warped + (1.0 - temporal_weight[..., None]) * n01
                        n01 = _normalize_normal01(n01)
                    else:
                        n01 = _stabilize_normal(prev_n, n01, ema)
                    prev_n = n01
                exr_path = os.path.join(norm_exr_dir, f"normal.{frame_num:04d}.exr")
                _save_normal_exr(exr_path, n01, normal_range)
                norm_count += 1
                norm_vis_frames.append((n01 * 255.0).astype(np.uint8))
            prev_rgb = rgb
            torch.cuda.empty_cache()

        fps = float(job_data.get("fps", 24.0))
        if depth_mp4_dir and depth_vis_frames:
            os.makedirs(depth_mp4_dir, exist_ok=True)
            p = os.path.join(depth_mp4_dir, "lotus_depth_vis.mp4")
            self._write_mp4_rgb(p, depth_vis_frames, fps)
        if norm_mp4_dir and norm_vis_frames:
            os.makedirs(norm_mp4_dir, exist_ok=True)
            p = os.path.join(norm_mp4_dir, "normal_vis.mp4")
            self._write_mp4_rgb(p, norm_vis_frames, fps)

        return {
            "status": "success",
            "lotus_depth_frames": depth_count,
            "normal_frames": norm_count,
            "lotus_depth_exr_dir": depth_exr_dir,
            "normal_exr_dir": norm_exr_dir,
            "lotus_depth_mp4_dir": depth_mp4_dir,
            "normal_mp4_dir": norm_mp4_dir,
        }

    @staticmethod
    def _write_mp4_rgb(path: str, frames: List[np.ndarray], fps: float) -> None:
        if not frames:
            return
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for fr in frames:
            if fr.shape[-1] == 3:
                bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            else:
                bgr = fr
            writer.write(bgr)
        writer.release()
