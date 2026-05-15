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

import cv2
import numpy as np
import torch
from contextlib import nullcontext
from PIL import Image
from tqdm import tqdm

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# HF model ids (Lotus-D regression, same family as LOT/app.py defaults)
DEFAULT_DEPTH_MODEL_D = "jingheya/lotus-depth-d-v2-0-disparity"
DEFAULT_NORMAL_MODEL_D = "jingheya/lotus-normal-d-v1-0"


def _task_embedding(device: torch.device) -> torch.Tensor:
    task_emb = torch.tensor([[1.0, 0.0]], device=device, dtype=torch.float32)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
    return task_emb


def _load_rgb_frame(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_frame_from_pattern(pattern: str, frame_num: int, padding: int) -> np.ndarray:
    frame_path = re.sub(
        r"%0?(\d+)d",
        lambda m: f"{frame_num:0{int(m.group(1))}d}",
        pattern,
    )
    if os.path.exists(frame_path):
        return _load_rgb_frame(frame_path)
    lower = frame_path.lower()
    if lower.endswith(".exr"):
        try:
            import OpenEXR
            import Imath
        except ImportError:
            raise RuntimeError("EXR input requires OpenEXR. Enable create_jpg in app or install OpenEXR.") from None
        exr = OpenEXR.InputFile(frame_path)
        dw = exr.header()["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        channels = list(exr.header()["channels"].keys())
        ch0 = channels[0]
        extype = Imath.PixelType(Imath.PixelType.FLOAT)
        raw = np.frombuffer(exr.channel(ch0, extype), dtype=np.float32).reshape((h, w))
        rgb = np.stack([raw, raw, raw], axis=-1)
        rgb = np.clip(rgb, 0.0, None)
        mx = float(rgb.max()) if rgb.size else 1.0
        if mx > 1.0:
            rgb = rgb / (mx + 1e-8)
        return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    raise FileNotFoundError(frame_path)


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


class LOTUSDepthNormalEngine:
    """Load Lotus-D pipelines and run a frame range from an image sequence pattern."""

    def __init__(self, lotus_root: Optional[str] = None):
        self.lotus_root = Path(lotus_root) if lotus_root else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe_depth = None
        self._pipe_normal = None

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

        from utils.image_utils import colorize_depth_map

        os.makedirs(depth_exr_dir, exist_ok=True)
        if gen_norm:
            os.makedirs(norm_exr_dir, exist_ok=True)

        depth_vis_frames: List[np.ndarray] = []
        norm_vis_frames: List[np.ndarray] = []

        prev_d: Optional[np.ndarray] = None
        prev_n: Optional[np.ndarray] = None
        depth_count = 0
        norm_count = 0

        for frame_num in tqdm(range(first, last + 1), desc="LOTUS frames"):
            rgb = _read_frame_from_pattern(pattern, frame_num, pad)
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
                    n01 = _stabilize_normal(prev_n, n01, ema)
                    prev_n = n01
                exr_path = os.path.join(norm_exr_dir, f"normal.{frame_num:04d}.exr")
                _save_normal_exr(exr_path, n01, normal_range)
                norm_count += 1
                norm_vis_frames.append((n01 * 255.0).astype(np.uint8))
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
