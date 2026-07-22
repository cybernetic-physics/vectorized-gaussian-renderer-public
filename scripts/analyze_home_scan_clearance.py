#!/usr/bin/env python3
"""Build an eye-level XZ occupancy and clearance map for Home Scan."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaacsim_gaussian_renderer.ply_loader import load_ply_to_gaussians


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-path",
        type=Path,
        default=Path("/workspace/datasets/home-scan-lod0.ply"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/flyby/home-scan-clearance"),
    )
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--grid-width", type=int, default=384)
    parser.add_argument("--lower-quantile", type=float, default=0.005)
    parser.add_argument("--upper-quantile", type=float, default=0.995)
    parser.add_argument("--height-min-fraction", type=float, default=0.30)
    parser.add_argument("--height-max-fraction", type=float, default=0.82)
    parser.add_argument("--occupied-count", type=int, default=2)
    parser.add_argument("--dilation-cells", type=int, default=3)
    parser.add_argument("--footprint-close-cells", type=int, default=12)
    parser.add_argument("--candidate-count", type=int, default=24)
    parser.add_argument("--candidate-separation-cells", type=int, default=18)
    parser.add_argument("--interior-component-count", type=int, default=16)
    parser.add_argument("--interior-component-min-cells", type=int, default=120)
    parser.add_argument("--route-waypoint-count", type=int, default=8)
    parser.add_argument("--route-output-points", type=int, default=192)
    parser.add_argument("--eye-height-fraction", type=float, default=0.40)
    return parser.parse_args()


def colorize_clearance(
    clearance: np.ndarray,
    occupied: np.ndarray,
) -> np.ndarray:
    scale = max(float(np.percentile(clearance, 99.0)), 1.0)
    normalized = np.clip(clearance / scale, 0.0, 1.0)
    image = np.zeros((*clearance.shape, 3), dtype=np.uint8)
    image[..., 0] = (35.0 + 40.0 * normalized).astype(np.uint8)
    image[..., 1] = (35.0 + 190.0 * normalized).astype(np.uint8)
    image[..., 2] = (45.0 + 210.0 * normalized).astype(np.uint8)
    image[occupied] = np.array([245, 245, 245], dtype=np.uint8)
    return image


def pick_candidates(
    clearance: np.ndarray,
    count: int,
    separation: int,
) -> list[tuple[int, int, float]]:
    working = clearance.copy()
    candidates: list[tuple[int, int, float]] = []
    yy, xx = np.ogrid[: clearance.shape[0], : clearance.shape[1]]
    for _ in range(count):
        flat_index = int(np.argmax(working))
        y, x = np.unravel_index(flat_index, working.shape)
        value = float(working[y, x])
        if value <= 0:
            break
        candidates.append((int(y), int(x), value))
        mask = (yy - y) ** 2 + (xx - x) ** 2 <= separation**2
        working[mask] = 0
    return candidates


def astar_path(
    free: np.ndarray,
    clearance: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    if not free[start] or not free[goal]:
        raise ValueError("A* endpoints must be free cells.")
    neighbors = (
        (-1, -1, math.sqrt(2.0)),
        (-1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, math.sqrt(2.0)),
        (1, 0, 1.0),
        (1, 1, math.sqrt(2.0)),
    )
    queue: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(queue, (0.0, 0.0, start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    best_cost = {start: 0.0}
    height, width = free.shape
    while queue:
        _, cost, current = heapq.heappop(queue)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        if cost > best_cost.get(current, float("inf")):
            continue
        for dz, dx, step_distance in neighbors:
            neighbor = (current[0] + dz, current[1] + dx)
            if (
                neighbor[0] < 0
                or neighbor[0] >= height
                or neighbor[1] < 0
                or neighbor[1] >= width
                or not free[neighbor]
            ):
                continue
            clearance_penalty = 2.5 / (float(clearance[neighbor]) + 1.0)
            next_cost = cost + step_distance * (1.0 + clearance_penalty)
            if next_cost >= best_cost.get(neighbor, float("inf")):
                continue
            best_cost[neighbor] = next_cost
            came_from[neighbor] = current
            heuristic = math.hypot(
                goal[0] - neighbor[0],
                goal[1] - neighbor[1],
            )
            heapq.heappush(
                queue,
                (next_cost + heuristic, next_cost, neighbor),
            )
    raise RuntimeError(f"No free-space route exists from {start} to {goal}.")


def resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("Polyline points must have shape [N, 2], N >= 2.")
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 1.0e-8))
    points = points[keep]
    if points.shape[0] < 2:
        raise ValueError("Polyline has no nonzero-length segments.")
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    sample_distances = np.linspace(0.0, cumulative[-1], count)
    result = np.empty((count, 2), dtype=np.float64)
    result[:, 0] = np.interp(sample_distances, cumulative, points[:, 0])
    result[:, 1] = np.interp(sample_distances, cumulative, points[:, 1])
    return result


def main() -> None:
    args = parse_args()
    if min(
        args.sample_stride,
        args.grid_width,
        args.occupied_count,
        args.candidate_count,
        args.candidate_separation_cells,
        args.interior_component_count,
        args.interior_component_min_cells,
        args.footprint_close_cells,
        args.route_waypoint_count,
        args.route_output_points,
    ) <= 0:
        raise ValueError("Clearance-map integer arguments must be positive.")
    if not (
        0.0
        <= args.lower_quantile
        < args.upper_quantile
        <= 1.0
        and 0.0
        <= args.height_min_fraction
        < args.height_max_fraction
        <= 1.0
        and 0.0
        < args.eye_height_fraction
        < 1.0
    ):
        raise ValueError("Quantiles and height fractions are invalid.")

    scene = load_ply_to_gaussians(args.scene_path)
    means = scene.means[:: args.sample_stride].numpy()
    bounds = np.quantile(
        means,
        [args.lower_quantile, args.upper_quantile],
        axis=0,
    )
    bounds_min = bounds[0]
    bounds_max = bounds[1]
    extent = bounds_max - bounds_min
    y_min = bounds_min[1] + extent[1] * args.height_min_fraction
    y_max = bounds_min[1] + extent[1] * args.height_max_fraction
    band = means[
        (means[:, 1] >= y_min)
        & (means[:, 1] <= y_max)
        & (means[:, 0] >= bounds_min[0])
        & (means[:, 0] <= bounds_max[0])
        & (means[:, 2] >= bounds_min[2])
        & (means[:, 2] <= bounds_max[2])
    ]

    aspect = float(extent[2] / extent[0])
    grid_height = max(64, int(round(args.grid_width * aspect)))
    x = np.clip(
        ((band[:, 0] - bounds_min[0]) / extent[0] * args.grid_width).astype(
            np.int64
        ),
        0,
        args.grid_width - 1,
    )
    z = np.clip(
        ((band[:, 2] - bounds_min[2]) / extent[2] * grid_height).astype(
            np.int64
        ),
        0,
        grid_height - 1,
    )
    counts = np.zeros((grid_height, args.grid_width), dtype=np.int32)
    np.add.at(counts, (z, x), 1)
    occupied_raw = counts >= args.occupied_count
    occupied = ndimage.binary_dilation(
        occupied_raw,
        iterations=args.dilation_cells,
    )
    free = ~occupied
    labels, label_count = ndimage.label(free)
    if label_count <= 0:
        raise RuntimeError("No free-space component was found.")
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    largest_label = int(np.argmax(component_sizes))
    clearance_all = ndimage.distance_transform_edt(free)
    largest_clearance = np.where(
        labels == largest_label,
        clearance_all,
        0.0,
    )
    outside_candidates = pick_candidates(
        largest_clearance,
        args.candidate_count,
        args.candidate_separation_cells,
    )
    boundary_labels = set(labels[0, :].tolist())
    boundary_labels.update(labels[-1, :].tolist())
    boundary_labels.update(labels[:, 0].tolist())
    boundary_labels.update(labels[:, -1].tolist())
    boundary_labels.discard(0)
    interior_labels = [
        int(label)
        for label in np.argsort(component_sizes)[::-1]
        if label != 0
        and label not in boundary_labels
        and component_sizes[label] >= args.interior_component_min_cells
    ][: args.interior_component_count]
    interior_candidates: list[tuple[int, int, float, int]] = []
    for label in interior_labels:
        component_clearance = np.where(
            labels == label,
            clearance_all,
            0.0,
        )
        flat_index = int(np.argmax(component_clearance))
        grid_z, grid_x = np.unravel_index(
            flat_index,
            component_clearance.shape,
        )
        interior_candidates.append(
            (
                int(grid_z),
                int(grid_x),
                float(component_clearance[grid_z, grid_x]),
                label,
            )
        )

    footprint_closed = ndimage.binary_closing(
        occupied_raw,
        iterations=args.footprint_close_cells,
    )
    footprint_filled = ndimage.binary_fill_holes(footprint_closed)
    footprint_labels, footprint_label_count = ndimage.label(footprint_filled)
    if footprint_label_count <= 0:
        raise RuntimeError("No scan footprint was found.")
    footprint_sizes = np.bincount(footprint_labels.ravel())
    footprint_sizes[0] = 0
    footprint_label = int(np.argmax(footprint_sizes))
    footprint = footprint_labels == footprint_label
    footprint_free = footprint & ~occupied
    footprint_clearance = ndimage.distance_transform_edt(footprint_free)
    footprint_candidates = pick_candidates(
        footprint_clearance,
        args.candidate_count,
        args.candidate_separation_cells,
    )
    route_labels, route_label_count = ndimage.label(footprint_free)
    if route_label_count <= 0:
        raise RuntimeError("No connected free-space route component was found.")
    route_component_sizes = np.bincount(route_labels.ravel())
    route_component_sizes[0] = 0
    route_component_label = int(np.argmax(route_component_sizes))
    route_free = route_labels == route_component_label
    route_clearance = np.where(
        route_free,
        footprint_clearance,
        0.0,
    )
    route_candidates = pick_candidates(
        route_clearance,
        args.route_waypoint_count,
        args.candidate_separation_cells,
    )
    if len(route_candidates) < 2:
        raise RuntimeError("The route planner found fewer than two waypoints.")
    route_candidate_cells = np.asarray(
        [(item[0], item[1]) for item in route_candidates],
        dtype=np.float64,
    )
    centered_candidates = (
        route_candidate_cells - route_candidate_cells.mean(axis=0)
    )
    _, _, axes = np.linalg.svd(centered_candidates, full_matrices=False)
    principal_axis = axes[0]
    if principal_axis[1] < 0:
        principal_axis = -principal_axis
    route_order = np.argsort(centered_candidates @ principal_axis)
    ordered_route_candidates = [
        route_candidates[int(index)]
        for index in route_order
    ]
    route_cells: list[tuple[int, int]] = []
    for index in range(len(ordered_route_candidates) - 1):
        start = ordered_route_candidates[index][:2]
        goal = ordered_route_candidates[index + 1][:2]
        segment = astar_path(
            route_free,
            footprint_clearance,
            start,
            goal,
        )
        route_cells.extend(segment if index == 0 else segment[1:])
    route_grid = np.asarray(route_cells, dtype=np.float64)
    route_world = np.empty_like(route_grid)
    route_world[:, 0] = bounds_min[0] + (
        (route_grid[:, 1] + 0.5) / args.grid_width
    ) * extent[0]
    route_world[:, 1] = bounds_min[2] + (
        (route_grid[:, 0] + 0.5) / grid_height
    ) * extent[2]
    route_world = resample_polyline(
        route_world,
        args.route_output_points,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    density_scale = max(float(np.percentile(np.log1p(counts), 99.5)), 1.0)
    density = (
        np.clip(np.log1p(counts) / density_scale, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    Image.fromarray(np.flipud(density)).save(args.output_dir / "density.png")

    clearance_rgb = colorize_clearance(clearance_all, occupied)
    clearance_image = Image.fromarray(np.flipud(clearance_rgb))
    draw = ImageDraw.Draw(clearance_image)
    candidate_payload: list[dict[str, float | int]] = []
    for index, (grid_z, grid_x, clearance_cells) in enumerate(
        outside_candidates
    ):
        draw_y = grid_height - 1 - grid_z
        radius = 5
        draw.ellipse(
            (
                grid_x - radius,
                draw_y - radius,
                grid_x + radius,
                draw_y + radius,
            ),
            fill=(255, 64, 64),
            outline=(0, 0, 0),
            width=1,
        )
        draw.text((grid_x + 7, draw_y - 7), str(index), fill=(0, 0, 0))
        world_x = bounds_min[0] + (
            (grid_x + 0.5) / args.grid_width
        ) * extent[0]
        world_z = bounds_min[2] + (
            (grid_z + 0.5) / grid_height
        ) * extent[2]
        candidate_payload.append(
            {
                "index": index,
                "grid_x": grid_x,
                "grid_z": grid_z,
                "world_x": float(world_x),
                "world_z": float(world_z),
                "clearance_cells": clearance_cells,
                "component_kind": "boundary-connected",
                "component_label": largest_label,
            }
        )
    clearance_image.save(args.output_dir / "clearance.png")

    component_rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    component_rgb[occupied] = np.array([245, 245, 245], dtype=np.uint8)
    palette = np.array(
        [
            [38, 139, 210],
            [42, 161, 152],
            [133, 153, 0],
            [181, 137, 0],
            [203, 75, 22],
            [211, 54, 130],
            [108, 113, 196],
            [88, 110, 117],
        ],
        dtype=np.uint8,
    )
    interior_payload: list[dict[str, float | int]] = []
    for index, (grid_z, grid_x, clearance_cells, label) in enumerate(
        interior_candidates
    ):
        component_rgb[labels == label] = palette[index % len(palette)]
        interior_payload.append(
            {
                "index": index,
                "grid_x": grid_x,
                "grid_z": grid_z,
                "world_x": float(
                    bounds_min[0]
                    + ((grid_x + 0.5) / args.grid_width) * extent[0]
                ),
                "world_z": float(
                    bounds_min[2]
                    + ((grid_z + 0.5) / grid_height) * extent[2]
                ),
                "clearance_cells": clearance_cells,
                "component_cells": int(component_sizes[label]),
                "component_kind": "interior",
                "component_label": label,
            }
        )
    component_image = Image.fromarray(np.flipud(component_rgb))
    component_draw = ImageDraw.Draw(component_image)
    for candidate in interior_payload:
        grid_x = int(candidate["grid_x"])
        draw_y = grid_height - 1 - int(candidate["grid_z"])
        radius = 6
        component_draw.ellipse(
            (
                grid_x - radius,
                draw_y - radius,
                grid_x + radius,
                draw_y + radius,
            ),
            fill=(255, 80, 80),
            outline=(0, 0, 0),
            width=1,
        )
        component_draw.text(
            (grid_x + 8, draw_y - 8),
            str(candidate["index"]),
            fill=(0, 0, 0),
        )
    component_image.save(args.output_dir / "components.png")

    footprint_rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    footprint_rgb[footprint] = colorize_clearance(
        footprint_clearance,
        occupied,
    )[footprint]
    footprint_rgb[occupied & footprint] = np.array(
        [245, 245, 245],
        dtype=np.uint8,
    )
    footprint_image = Image.fromarray(np.flipud(footprint_rgb))
    footprint_draw = ImageDraw.Draw(footprint_image)
    footprint_payload: list[dict[str, float | int]] = []
    for index, (grid_z, grid_x, clearance_cells) in enumerate(
        footprint_candidates
    ):
        draw_y = grid_height - 1 - grid_z
        radius = 5
        footprint_draw.ellipse(
            (
                grid_x - radius,
                draw_y - radius,
                grid_x + radius,
                draw_y + radius,
            ),
            fill=(255, 80, 80),
            outline=(0, 0, 0),
            width=1,
        )
        footprint_draw.text(
            (grid_x + 7, draw_y - 7),
            str(index),
            fill=(255, 255, 0),
        )
        footprint_payload.append(
            {
                "index": index,
                "grid_x": grid_x,
                "grid_z": grid_z,
                "world_x": float(
                    bounds_min[0]
                    + ((grid_x + 0.5) / args.grid_width) * extent[0]
                ),
                "world_z": float(
                    bounds_min[2]
                    + ((grid_z + 0.5) / grid_height) * extent[2]
                ),
                "clearance_cells": clearance_cells,
                "component_kind": "closed-scan-footprint",
            }
        )
    footprint_image.save(args.output_dir / "footprint.png")

    route_image = Image.fromarray(np.flipud(footprint_rgb))
    route_draw = ImageDraw.Draw(route_image)
    route_pixels = [
        (int(grid_x), grid_height - 1 - int(grid_z))
        for grid_z, grid_x in route_cells
    ]
    route_draw.line(
        route_pixels,
        fill=(255, 128, 0),
        width=3,
        joint="curve",
    )
    ordered_route_payload: list[dict[str, float | int]] = []
    for order_index, (grid_z, grid_x, clearance_cells) in enumerate(
        ordered_route_candidates
    ):
        draw_y = grid_height - 1 - grid_z
        radius = 6
        route_draw.ellipse(
            (
                grid_x - radius,
                draw_y - radius,
                grid_x + radius,
                draw_y + radius,
            ),
            fill=(255, 80, 80),
            outline=(0, 0, 0),
            width=1,
        )
        route_draw.text(
            (grid_x + 8, draw_y - 8),
            str(order_index),
            fill=(255, 255, 0),
        )
        ordered_route_payload.append(
            {
                "order": order_index,
                "grid_x": grid_x,
                "grid_z": grid_z,
                "clearance_cells": clearance_cells,
            }
        )
    route_image.save(args.output_dir / "route.png")

    eye_height_fraction = args.eye_height_fraction
    eye_height = float(
        bounds_min[1] + extent[1] * eye_height_fraction
    )
    route_contract = {
        "schema_version": "home-scan-clearance-route/v1",
        "coordinate_system": "world-xz-y-up",
        "scene_path": str(args.scene_path),
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "eye_height": eye_height,
        "eye_height_fraction": eye_height_fraction,
        "world_path_xz": route_world.tolist(),
        "route_length": float(
            np.linalg.norm(np.diff(route_world, axis=0), axis=1).sum()
        ),
        "route_grid_cell_count": len(route_cells),
        "route_output_points": args.route_output_points,
        "ordered_waypoints": ordered_route_payload,
        "planner": {
            "sample_stride": args.sample_stride,
            "grid_width": args.grid_width,
            "grid_height": grid_height,
            "height_min_fraction": args.height_min_fraction,
            "height_max_fraction": args.height_max_fraction,
            "occupied_count": args.occupied_count,
            "dilation_cells": args.dilation_cells,
            "footprint_close_cells": args.footprint_close_cells,
            "route_component_cells": int(
                route_component_sizes[route_component_label]
            ),
        },
    }
    route_hash_payload = json.dumps(
        route_contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    route_contract["route_sha256"] = hashlib.sha256(
        route_hash_payload
    ).hexdigest()
    (args.output_dir / "walkthrough-route.json").write_text(
        json.dumps(route_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "home-scan-clearance/v1",
        "scene_path": str(args.scene_path),
        "sample_stride": args.sample_stride,
        "sample_count": int(means.shape[0]),
        "height_band_count": int(band.shape[0]),
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "height_min": float(y_min),
        "height_max": float(y_max),
        "grid_width": args.grid_width,
        "grid_height": grid_height,
        "occupied_count": args.occupied_count,
        "dilation_cells": args.dilation_cells,
        "footprint_close_cells": args.footprint_close_cells,
        "footprint_cells": int(footprint.sum()),
        "largest_free_component_cells": int(component_sizes[largest_label]),
        "boundary_connected_candidates": candidate_payload,
        "interior_candidates": interior_payload,
        "footprint_candidates": footprint_payload,
        "route_contract": "walkthrough-route.json",
        "route_sha256": route_contract["route_sha256"],
        "route_length": route_contract["route_length"],
    }
    (args.output_dir / "clearance.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "HOME_SCAN_CLEARANCE_OK "
        + json.dumps(
            {
                "output_dir": str(args.output_dir),
                "candidate_count": len(candidate_payload),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
