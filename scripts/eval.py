from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute an evaluation notebook.")
    parser.add_argument(
        "--notebook",
        default="notebooks/Analize_Results.ipynb",
        help="Notebook to execute.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/reports",
        help="Directory for executed notebook outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--output-dir",
            str(outdir),
            args.notebook,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
