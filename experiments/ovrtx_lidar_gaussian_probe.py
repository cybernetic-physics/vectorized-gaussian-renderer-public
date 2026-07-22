"""Probe whether OVRTX LiDAR observes Gaussian ParticleFields and mesh walls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import ovrtx
import torch

from isaacsim_gaussian_renderer import GaussianLidarService
from isaacsim_gaussian_renderer.cuda_lidar_backend import CudaLidarBackend


OVRTX_SOURCE_COMMIT = "29d11037fbcaed0f0f53e7f32d17bd0486fd453b"
PRODUCT = "/World/Render/Products/LidarProduct"
SENSOR_ORIGIN = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def gaussian_field() -> str:
    coordinates = [(10.0, float(y), float(z)) for y in np.linspace(-4, 4, 9) for z in np.linspace(0, 4, 9)]
    positions = ",\n            ".join(f"({x}, {y}, {z})" for x, y, z in coordinates)
    scales = ",\n            ".join("(0.01, 0.55, 0.55)" for _ in coordinates)
    orientations = ",\n            ".join("(1, 0, 0, 0)" for _ in coordinates)
    opacities = ", ".join("0.99" for _ in coordinates)
    sh_white = 0.5 / 0.28209479177387814
    coefficients = ",\n            ".join(f"({sh_white}, {sh_white}, {sh_white})" for _ in coordinates)
    return f'''    def ParticleField3DGaussianSplat "GaussianWall"
    {{
        point3f[] positions = [
            {positions}
        ]
        float3[] scales = [
            {scales}
        ]
        quatf[] orientations = [
            {orientations}
        ]
        float[] opacities = [{opacities}]
        uniform int radiance:sphericalHarmonicsDegree = 0
        float3[] radiance:sphericalHarmonicsCoefficients = [
            {coefficients}
        ] (
            elementSize = 1
            interpolation = "vertex"
        )
        float3[] extent = [(9.9, -4.6, -0.6), (10.1, 4.6, 4.6)]
        uniform token projectionModeHint = "perspective"
        uniform token sortingModeHint = "zDepth"
    }}'''


def mesh_wall() -> str:
    return '''    def Mesh "MeshWall" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(10, -5, -1), (10, 5, -1), (10, 5, 5), (10, -5, 5)]
        normal3f[] normals = [(-1, 0, 0), (-1, 0, 0), (-1, 0, 0), (-1, 0, 0)] (
            interpolation = "faceVarying"
        )
        float3[] extent = [(10, -5, -1), (10, 5, 5)]
        rel material:binding = </World/Materials/Concrete>
    }'''


def build_scene(mode: str) -> str:
    geometry = []
    if mode in ("gaussian", "both"):
        geometry.append(gaussian_field())
    if mode in ("mesh", "both"):
        geometry.append(mesh_wall())
    return f'''#usda 1.0
(
    defaultPrim = "World"
    startTimeCode = 0
    endTimeCode = 1
    timeCodesPerSecond = 10
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def OmniLidar "Lidar" (
        prepend apiSchemas = ["OmniSensorGenericLidarCoreAPI"]
    )
    {{
        token omni:sensor:Core:elementsCoordsType = "CARTESIAN"
        double2 omni:sensor:frameRate = (10, 1)
        double3 xformOp:translate = (0, 0, 1)
        float3 xformOp:rotateXYZ = (90, 0, -90)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    }}

{chr(10).join(geometry)}

    def "Render"
    {{
        def "Products"
        {{
            def RenderProduct "LidarProduct"
            {{
                rel camera = </World/Lidar>
                rel orderedVars = [<../../Vars/PointCloud>]
            }}
        }}
        def "Vars"
        {{
            def RenderVar "PointCloud"
            {{
                uniform string sourceName = "PointCloud"
                string[] channels = ["Coordinates", "Intensity", "Counts", "TimeOffsetNs"]
            }}
        }}
    }}

    def Xform "Materials"
    {{
        def Material "Concrete"
        {{
            token omni:simready:nonvisual:base = "concrete"
            token omni:simready:nonvisual:coating = "none"
            token[] omni:simready:nonvisual:attributes = ["none"]
            custom string inputs:nonvisual:base = "concrete"
            custom string inputs:nonvisual:coating = "none"
            custom string inputs:nonvisual:attributes = "none"
        }}
    }}
}}
'''


def run_probe(mode: str, output_dir: Path) -> dict[str, object]:
    scene = build_scene(mode)
    (output_dir / f"{mode}.usda").write_text(scene, encoding="utf-8")
    renderer = ovrtx.Renderer(
        ovrtx.RendererConfig(
            log_file_path=str(output_dir / f"{mode}.log"),
            log_level="info",
            enable_motion_bvh=True,
        )
    )
    renderer.open_usd_from_string(scene)
    for _ in range(3):
        renderer.step(render_products={PRODUCT}, delta_time=0.1)
    products = renderer.step(render_products={PRODUCT}, delta_time=0.1)
    frame = products[PRODUCT].frames[0]
    with frame.render_vars["PointCloud"].map(device=ovrtx.Device.CPU) as pointcloud:
        count = int(np.from_dlpack(pointcloud["Counts"])[0])
        coordinates = np.from_dlpack(pointcloud["Coordinates"])[:, :count].T.copy()
        intensity = np.from_dlpack(pointcloud["Intensity"])[:count].copy()
        time_offsets = np.from_dlpack(pointcloud["TimeOffsetNs"])[:count].copy()
    np.savez_compressed(
        output_dir / f"{mode}.npz",
        coordinates=coordinates,
        intensity=intensity,
        time_offsets_ns=time_offsets,
    )
    ranges = np.linalg.norm(coordinates - SENSOR_ORIGIN[None], axis=1) if count else np.empty((0,))
    return {
        "mode": mode,
        "valid_points": count,
        "mean_range_m": float(np.mean(ranges)) if count else None,
        "min_range_m": float(np.min(ranges)) if count else None,
        "max_range_m": float(np.max(ranges)) if count else None,
        "mean_intensity": float(np.mean(intensity)) if count else None,
        "max_time_offset_ns": int(np.max(time_offsets)) if count else None,
    }


def custom_mesh_range_comparison(mesh_points: np.ndarray) -> dict[str, object]:
    if len(mesh_points) == 0:
        raise RuntimeError("OVRTX mesh probe returned no points.")
    device = torch.device("cuda:0")
    directions = mesh_points.astype(np.float64) - SENSOR_ORIGIN[None]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    ray_tensor = torch.from_numpy(directions.astype(np.float32)).to(device).contiguous()
    count = 81
    yz = [(float(y), float(z)) for y in np.linspace(-4, 4, 9) for z in np.linspace(0, 4, 9)]
    service = GaussianLidarService(
        CudaLidarBackend(max_scenes=1),
        max_sensors=1,
        max_rays=len(directions),
        returns=1,
    )
    service.initialize(stage=None, device=device)
    service.load_scene(
        1,
        means=torch.tensor([[10.0, y, z] for y, z in yz], device=device, dtype=torch.float32),
        scales=torch.tensor([[0.01, 0.55, 0.55]] * count, device=device),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count, device=device),
        opacities=torch.full((count,), 0.99, device=device),
        semantic_ids=torch.ones((count,), device=device, dtype=torch.int64),
        reflectivity=torch.full((count,), 0.5, device=device),
    )
    transform = torch.eye(4, device=device).unsqueeze(0).contiguous()
    transform[0, 2, 3] = 1.0
    outputs = service.render_lidar(
        ray_tensor,
        torch.zeros((len(directions),), device=device, dtype=torch.int64),
        transform,
        torch.tensor([1], device=device, dtype=torch.int64),
    )
    service.synchronize()
    service.backend.check_errors(synchronize=False)
    valid = outputs["valid"][0, :, 0].cpu().numpy()
    custom_ranges = outputs["range_m"][0, :, 0].cpu().numpy()
    analytic_ranges = 10.0 / directions[:, 0]
    ovrtx_ranges = np.linalg.norm(mesh_points - SENSOR_ORIGIN[None], axis=1)
    result = {
        "ray_count": len(directions),
        "custom_valid_count": int(np.count_nonzero(valid)),
        "custom_vs_analytic_max_abs_m": float(np.max(np.abs(custom_ranges[valid] - analytic_ranges[valid]))),
        "ovrtx_mesh_vs_analytic_max_abs_m": float(np.max(np.abs(ovrtx_ranges - analytic_ranges))),
        "intensity_compared": False,
        "intensity_reason": "OVRTX material and custom reflectivity normalization are not calibrated",
    }
    service.shutdown()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lidar/ovrtx-probes"))
    parser.add_argument("--mode", choices=("gaussian", "mesh", "both"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = (args.mode,) if args.mode is not None else ("gaussian", "mesh", "both")
    probes = {mode: run_probe(mode, args.output_dir) for mode in modes}
    comparison = None
    if "mesh" in probes:
        mesh_arrays = np.load(args.output_dir / "mesh.npz")
        comparison = custom_mesh_range_comparison(mesh_arrays["coordinates"])
    gaussian_seen = bool(
        "gaussian" in probes
        and probes["gaussian"]["valid_points"]
        and probes["gaussian"]["min_range_m"] is not None
        and 8.0 <= float(probes["gaussian"]["min_range_m"]) <= 12.0
    )
    result = {
        "schema_version": "ovrtx-gaussian-lidar-probes/v1",
        "pass": True,
        "ovrtx_source_commit": OVRTX_SOURCE_COMMIT,
        "probes": probes,
        "ovrtx_lidar_sees_gaussian_particle_fields": gaussian_seen,
        "mesh_range_comparison": comparison,
        "speedup_claimed": False,
    }
    result_name = "results.json" if args.mode is None else f"results-{args.mode}.json"
    (args.output_dir / result_name).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("OVRTX_GAUSSIAN_LIDAR_PROBES_OK")


if __name__ == "__main__":
    main()
