"""Finite Isaac Sim lifecycle smoke test for the remote development machine.

The minimal lane proves that a standalone Kit application can start, step,
stop, drain an update, and exit cleanly.  The default lane repeats that proof
with Isaac's normal RTX renderer.  Neither lane substitutes for the separate
OVRTX activation gate's Gaussian-output assertions.
"""

import argparse
import sys
import traceback

from isaacsim import SimulationApp

if __package__:
    from ._isaac_launch import close_simulation_app, isaac_app_config
else:
    from _isaac_launch import close_simulation_app, isaac_app_config


parser = argparse.ArgumentParser()
parser.add_argument(
    "--renderer",
    choices=("default", "minimal"),
    default="default",
    help="Use Isaac's default RTX renderer or the cheap lifecycle-only renderer.",
)
args, kit_args = parser.parse_known_args()
# SimulationApp consumes any remaining Kit arguments from sys.argv.
sys.argv = [sys.argv[0], *kit_args]


app_config = isaac_app_config(
    renderer="MinimalRendering" if args.renderer == "minimal" else None,
)
simulation_app = SimulationApp(app_config)

exit_code = 1
try:
    from isaacsim.core.api import World

    world = World()
    world.scene.add_default_ground_plane()
    world.reset()

    for _ in range(10):
        world.step(render=True)

    # World.reset() starts the timeline.  Stop it before Kit's fast shutdown;
    # stop() also performs the standalone app update needed to drain this work.
    world.stop()
    print("ISAAC_HEADLESS_SMOKE_OK", flush=True)
    exit_code = 0
except BaseException:
    traceback.print_exc()
    raise
finally:
    # SimulationApp.close() owns the process exit in fast-shutdown mode.  The
    # parent must therefore require both this marker and shell return code 0.
    close_simulation_app(simulation_app, failed=exit_code != 0)
