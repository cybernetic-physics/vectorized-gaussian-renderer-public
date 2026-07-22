# Third-party notices

This distribution contains a FlashGS-derived matched-contract rasterization
backend under `isaacsim_gaussian_renderer/native/flashgs/`. The port is derived
from InternLandMark's public FlashGS repository at commit
`cdfc4e4002318423eda356eed02df8e01fa32cb6` and remains subject to its MIT
License. The complete MIT text is distributed as
`isaacsim_gaussian_renderer/native/flashgs/LICENSE.flashgs` and as a wheel
license file.

The port retains FlashGS's emission, sorting, range, and tile-compositor
topology but changes projection and image formation to a pinned-gsplat target.
It is not represented as an upstream-faithful or integration-only FlashGS
build.

The separate `UpstreamFaithfulFlashGSBackend` control contains project-owned
integration glue under
`isaacsim_gaussian_renderer/native/flashgs_upstream_faithful/` and compiles the
hash-pinned public FlashGS `sort.cu` and `render.cu` from a clean external
checkout at runtime. Its generated `preprocess.cu` starts from that same MIT
source and changes only GPU camera/color input and bounded key-emission
plumbing. The public source is not vendored a second time in this distribution;
the same included FlashGS MIT license governs it.

FlashGS: <https://github.com/InternLandMark/FlashGS>

Optional Unitree G1 integration scripts accept a user-supplied robot asset;
this repository and its Python distribution do not include that asset. Unitree
Robotics publishes the `unitree_ros` robot descriptions under the BSD
3-Clause License. Users sourcing a G1 model from that repository must retain
its copyright and license notice and must not imply Unitree endorsement.

Unitree ROS: <https://github.com/unitreerobotics/unitree_ros>
Unitree BSD 3-Clause License: <https://github.com/unitreerobotics/unitree_ros/blob/master/LICENSE>

The repository also uses third-party dependencies and public datasets under
their own licenses. Dataset attribution is recorded in each file under
`datasets/*.manifest.json`; dataset content is not included in the Python
distribution.
