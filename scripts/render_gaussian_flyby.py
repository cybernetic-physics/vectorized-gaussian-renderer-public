#!/usr/bin/env python3
"""Generic entrypoint for the hash-pinned matched Gaussian flyby runner.

``render_home_scan_flyby.py`` remains a supported Home Scan-compatible name;
this entrypoint makes it explicit that the same contract now accepts any
prepared canonical PLY with provenance supplied through the scene arguments.
"""

from render_home_scan_flyby import main


if __name__ == "__main__":
    main()
