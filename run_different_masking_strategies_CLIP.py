"""Run clean/random fixed-mask CLIP strategies in isolated processes."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="All other options are forwarded to adv_score_CLIP_fixed_masked.py (use its --help).",
        allow_abbrev=False,
    )
    parser.add_argument("--mask-sources", nargs="+", choices=["clean", "random"], default=["clean", "random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args, forwarded = parser.parse_known_args()
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be non-negative")
    if any(option.split("=")[0] in ("--mask-source", "--seed") for option in forwarded):
        parser.error("Use --mask-sources and --seeds instead of --mask-source and --seed")
    runner = Path(__file__).resolve().with_name("adv_score_CLIP_fixed_masked.py")
    for source in dict.fromkeys(args.mask_sources):
        seeds = list(dict.fromkeys(args.seeds)) if source == "random" else [None]
        for seed in seeds:
            command = [sys.executable, str(runner), *forwarded, "--mask-source", source]
            if seed is not None:
                command.extend(["--seed", str(seed)])
            print(f"Running fixed mask: {source}, seed={seed}", flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
