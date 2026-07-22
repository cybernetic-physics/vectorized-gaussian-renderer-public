"""Warp kernels for zero-copy Fabric camera ingestion inside Isaac Sim."""

from __future__ import annotations

import warp as wp


@wp.kernel(enable_backward=False)
def fabric_world_to_opencv_viewmats(
    world_matrices: wp.fabricarray(dtype=wp.mat44d),
    camera_indices: wp.fabricarray(dtype=wp.int32),
    viewmats: wp.array(dtype=wp.mat44f),
):
    """Convert Gf row-vector USD camera matrices into OpenCV view matrices."""
    source_index = wp.tid()
    output_index = camera_indices[source_index]
    world = world_matrices[source_index]
    world_x = wp.float32(world[3, 0])
    world_y = wp.float32(world[3, 1])
    world_z = wp.float32(world[3, 2])

    r00 = wp.float32(world[0, 0])
    r01 = wp.float32(world[0, 1])
    r02 = wp.float32(world[0, 2])
    r10 = -wp.float32(world[1, 0])
    r11 = -wp.float32(world[1, 1])
    r12 = -wp.float32(world[1, 2])
    r20 = -wp.float32(world[2, 0])
    r21 = -wp.float32(world[2, 1])
    r22 = -wp.float32(world[2, 2])

    viewmats[output_index] = wp.mat44f(
        r00,
        r01,
        r02,
        -(r00 * world_x + r01 * world_y + r02 * world_z),
        r10,
        r11,
        r12,
        -(r10 * world_x + r11 * world_y + r12 * world_z),
        r20,
        r21,
        r22,
        -(r20 * world_x + r21 * world_y + r22 * world_z),
        0.0,
        0.0,
        0.0,
        1.0,
    )
