"""PolaRiS server entry point for EgoVerse checkpoint inference.

The model implementation stays in the EgoVerse checkout because it depends on that
environment. This entry point follows the PolaRiS policy-server layout and delegates
to ``egomimic.scripts.serve_egoverse_policy`` after adding EgoVerse to ``sys.path``.

Example:
    /scratch/madhavai/emimic/bin/python src/polaris/server/egoverse_server.py \
        --egoverse-root /home/madhavan/EgoVerse \
        --checkpoint /path/to/checkpoints/last.ckpt \
        --port 5557 \
        --image-scale-factor 2 \
        --crop-shape 360 480 \
        --wrist-crop-left 160
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--egoverse-root",
        type=Path,
        default=Path("/home/madhavan/EgoVerse"),
    )
    args, remaining = bootstrap.parse_known_args()

    root = args.egoverse_root.expanduser().resolve()
    if not (root / "egomimic").is_dir():
        raise FileNotFoundError(f"EgoVerse checkout not found at {root}")
    sys.path.insert(0, str(root))

    from egomimic.scripts.serve_egoverse_policy import main as serve

    # Let the delegated server parse its checkpoint, transport, and preprocessing flags.
    sys.argv = [sys.argv[0], *remaining]
    serve()


if __name__ == "__main__":
    main()
