"""Zero-copy torch bindings for the batched OptiX gaussian tracer.

Wraps libgs_tracer.so (build.sh) via ctypes. Rendered RGBA, depth, and
semantic outputs are written straight into caller-provided CUDA torch tensors.
CUDA view matrices, intrinsics, and sparse rays are also consumed directly;
CPU camera inputs remain supported through an explicit staged path. Calls run
on torch's current CUDA stream and are synchronous on that stream.

Example:
    tracer = GsTracer("/workspace/datasets/gaussians.bin")
    out = tracer.render_dense(viewmats, intrinsics, width=256, height=256)
    refl = tracer.render_sparse(origins, dirs)   # CUDA [N,3] tensors
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
import threading

import numpy as np
import torch

_LIB_DIR = Path(__file__).resolve().parent


class GsTracer:
    def __init__(
        self,
        dump_path: str | Path,
        ptx_path: str | Path | None = None,
        lib_path: str | Path | None = None,
    ) -> None:
        # The native context owns one launch-parameter buffer and one set of
        # camera scratch buffers. Serialize calls so two Python threads cannot
        # overwrite those buffers on different CUDA streams.
        self._lock = threading.RLock()
        lib_path = Path(lib_path or _LIB_DIR / "libgs_tracer.so")
        ptx_path = Path(ptx_path or _LIB_DIR / "gs_tracer_device.ptx")
        if not lib_path.is_file():
            raise FileNotFoundError(f"{lib_path} — run build.sh first")
        if not ptx_path.is_file():
            raise FileNotFoundError(f"{ptx_path} — run build.sh first")
        self._lib = ctypes.CDLL(str(lib_path))
        self._lib.gst_create.restype = ctypes.c_void_p
        self._lib.gst_create.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double),
        ]
        self._lib.gst_gas_build_ms.restype = ctypes.c_double
        self._lib.gst_gas_build_ms.argtypes = [ctypes.c_void_p]
        self._lib.gst_render_dense.restype = ctypes.c_int
        self._lib.gst_render_dense.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_ulonglong,
            ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_ulonglong,
        ]
        self._lib.gst_render_dense_device.restype = ctypes.c_int
        self._lib.gst_render_dense_device.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_ulonglong, ctypes.c_ulonglong,
            ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_ulonglong,
            ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_ulonglong,
        ]
        self._lib.gst_render_sparse.restype = ctypes.c_int
        self._lib.gst_render_sparse.argtypes = [
            ctypes.c_void_p, ctypes.c_longlong,
            ctypes.c_ulonglong, ctypes.c_ulonglong,
            ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_ulonglong,
            ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_ulonglong,
        ]
        self._lib.gst_destroy.argtypes = [ctypes.c_void_p]

        if not torch.cuda.is_available():
            raise RuntimeError("GsTracer requires an available CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device())
        info = (ctypes.c_double * 6)()
        with torch.cuda.device(self.device):
            self._ctx = self._lib.gst_create(
                str(dump_path).encode(), str(ptx_path).encode(), info
            )
        if not self._ctx:
            raise RuntimeError(
                f"Failed to create OptiX tracer from dump={dump_path} ptx={ptx_path}"
            )
        self.count = int(info[0])
        self.center = (info[1], info[2], info[3])
        self.radius = float(info[4])
        self.has_semantics = bool(info[5])
        self.gas_build_ms = float(self._lib.gst_gas_build_ms(self._ctx))

    def close(self) -> None:
        with self._lock:
            if self._ctx:
                with torch.cuda.device(self.device):
                    self._lib.gst_destroy(self._ctx)
                self._ctx = None

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        try:
            self.close()
        except Exception:
            pass

    def _check_out(self, t: torch.Tensor, dtype: torch.dtype, numel: int,
                   name: str) -> int:
        if (
            t.device != self.device
            or t.dtype != dtype
            or not t.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be a contiguous {self.device} {dtype} tensor"
            )
        if t.numel() != numel:
            raise ValueError(f"{name} numel {t.numel()} != {numel}")
        return t.data_ptr()

    def _stream_handle(self) -> int:
        return int(torch.cuda.current_stream(self.device).cuda_stream)

    @staticmethod
    def _check_trace_range(near_plane: float, far_plane: float) -> None:
        if (
            not math.isfinite(near_plane)
            or not math.isfinite(far_plane)
            or near_plane < 0.0
            or far_plane < 0.0
        ):
            raise ValueError("near_plane and far_plane must be finite and non-negative")
        effective_near = near_plane if near_plane > 0.0 else 1.0e-4
        if far_plane > 0.0 and far_plane <= effective_near:
            raise ValueError("far_plane must be greater than near_plane")

    def render_dense(
        self,
        viewmats: torch.Tensor,
        intrinsics: torch.Tensor,
        *,
        width: int,
        height: int,
        out_rgba: torch.Tensor | None = None,
        out_depth: torch.Tensor | None = None,
        out_semantic: torch.Tensor | None = None,
        kbuffer: bool = True,
        max_iters: int = 0,
        alpha_min: float = 0.0,
        tmin: float = 0.0,
        near_plane: float = 0.0,
        far_plane: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Batched pinhole render for OpenCV world->camera viewmats [B,4,4]."""
        if self._ctx is None:
            raise RuntimeError("GsTracer is closed")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self._check_trace_range(near_plane, far_plane)
        if viewmats.ndim != 3 or tuple(viewmats.shape[1:]) != (4, 4):
            raise ValueError("viewmats must have shape [B,4,4]")
        if intrinsics.ndim != 3 or tuple(intrinsics.shape[1:]) != (3, 3):
            raise ValueError("intrinsics must have shape [B,3,3]")
        envs = int(viewmats.shape[0])
        if envs <= 0 or intrinsics.shape[0] != envs:
            raise ValueError("viewmats and intrinsics need the same positive batch")
        if viewmats.device.type != intrinsics.device.type:
            raise ValueError("viewmats and intrinsics must both be CPU or both be CUDA")
        n = envs * height * width
        with self._lock, torch.cuda.device(self.device):
            if self._ctx is None:
                raise RuntimeError("GsTracer is closed")
            if out_rgba is None:
                out_rgba = torch.empty(
                    (envs, height, width, 4),
                    device=self.device,
                    dtype=torch.float32,
                )
            if out_depth is None:
                out_depth = torch.empty(
                    (envs, height, width),
                    device=self.device,
                    dtype=torch.float32,
                )
            if out_semantic is None:
                out_semantic = torch.empty(
                    (envs, height, width),
                    device=self.device,
                    dtype=torch.int32,
                )
            output_args = (
                self._check_out(out_rgba, torch.float32, n * 4, "out_rgba"),
                self._check_out(out_depth, torch.float32, n, "out_depth"),
                self._check_out(out_semantic, torch.int32, n, "out_semantic"),
                1 if kbuffer else 0,
                max_iters,
                alpha_min,
                tmin,
                near_plane,
                far_plane,
                self._stream_handle(),
            )
            if viewmats.device.type == "cuda":
                for name, tensor, shape in (
                    ("viewmats", viewmats, (envs, 4, 4)),
                    ("intrinsics", intrinsics, (envs, 3, 3)),
                ):
                    if (
                        tensor.device != self.device
                        or tensor.dtype != torch.float32
                        or not tensor.is_contiguous()
                        or tuple(tensor.shape) != shape
                    ):
                        raise ValueError(
                            f"CUDA {name} must be contiguous {self.device} float32"
                        )
                rc = self._lib.gst_render_dense_device(
                    self._ctx,
                    envs,
                    width,
                    height,
                    viewmats.data_ptr(),
                    intrinsics.data_ptr(),
                    *output_args,
                )
            else:
                vm = np.ascontiguousarray(
                    viewmats.detach().to("cpu", torch.float32).numpy()
                )
                K = np.ascontiguousarray(
                    intrinsics.detach().to("cpu", torch.float32).numpy()
                )
                rotations = vm[:, :3, :3]
                translations = vm[:, :3, 3]
                origins = -np.einsum("eji,ej->ei", rotations, translations)
                cams = np.ascontiguousarray(
                    np.concatenate(
                        (origins, rotations.transpose(0, 2, 1).reshape(envs, 9)),
                        axis=1,
                    ),
                    dtype=np.float32,
                )
                intrinsics4 = np.ascontiguousarray(
                    np.stack(
                        (K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]),
                        axis=1,
                    ),
                    dtype=np.float32,
                )
                rc = self._lib.gst_render_dense(
                    self._ctx,
                    envs,
                    width,
                    height,
                    cams.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    intrinsics4.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    *output_args,
                )
        if rc != 0:
            raise RuntimeError(f"gst_render_dense failed rc={rc}")
        return {"rgba": out_rgba, "depth": out_depth, "semantic": out_semantic}

    def render_sparse(
        self,
        origins: torch.Tensor,
        dirs: torch.Tensor,
        *,
        out_rgba: torch.Tensor | None = None,
        out_depth: torch.Tensor | None = None,
        out_semantic: torch.Tensor | None = None,
        kbuffer: bool = True,
        max_iters: int = 0,
        alpha_min: float = 0.0,
        tmin: float = 0.0,
        near_plane: float = 0.0,
        far_plane: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Arbitrary ray-list render; origins/dirs are CUDA [N,3] float32."""
        if self._ctx is None:
            raise RuntimeError("GsTracer is closed")
        self._check_trace_range(near_plane, far_plane)
        for name, t in (("origins", origins), ("dirs", dirs)):
            if (
                t.device != self.device or t.dtype != torch.float32
                or not t.is_contiguous() or t.ndim != 2 or t.shape[1] != 3
            ):
                raise ValueError(
                    f"{name} must be contiguous {self.device} float32 [N,3]"
                )
        if origins.shape[0] != dirs.shape[0]:
            raise ValueError("origins and dirs need the same ray count")
        n = int(origins.shape[0])
        with self._lock, torch.cuda.device(self.device):
            if self._ctx is None:
                raise RuntimeError("GsTracer is closed")
            if out_rgba is None:
                out_rgba = torch.empty(
                    (n, 4), device=self.device, dtype=torch.float32
                )
            if out_depth is None:
                out_depth = torch.empty(
                    (n,), device=self.device, dtype=torch.float32
                )
            if out_semantic is None:
                out_semantic = torch.empty(
                    (n,), device=self.device, dtype=torch.int32
                )
            rc = self._lib.gst_render_sparse(
                self._ctx, n, origins.data_ptr(), dirs.data_ptr(),
                self._check_out(out_rgba, torch.float32, n * 4, "out_rgba"),
                self._check_out(out_depth, torch.float32, n, "out_depth"),
                self._check_out(out_semantic, torch.int32, n, "out_semantic"),
                1 if kbuffer else 0, max_iters, alpha_min, tmin,
                near_plane, far_plane,
                self._stream_handle(),
            )
        if rc != 0:
            raise RuntimeError(f"gst_render_sparse failed rc={rc}")
        return {"rgba": out_rgba, "depth": out_depth, "semantic": out_semantic}
