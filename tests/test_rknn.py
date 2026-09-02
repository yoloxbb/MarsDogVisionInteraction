"""Manual Ultralytics/RKNN hardware smoke test (not a pytest test)."""

from __future__ import annotations

import argparse
from pathlib import Path

from marsdog_vision_interaction.utils.config_loader import load_config


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    config = load_config(project_dir / "config" / "vision.yaml")
    default_model = config["providers"]["object"]["config"]["object_model"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="input image")
    parser.add_argument("--model", type=Path, default=Path(default_model))
    parser.add_argument(
        "--output",
        type=Path,
        help="result image; defaults beside the input image",
    )
    args = parser.parse_args()
    output = args.output or args.image.with_name(
        f"{args.image.stem}_result{args.image.suffix or '.jpg'}"
    )

    from ultralytics import YOLOE

    model = YOLOE(str(args.model))
    results = model.predict(
        str(args.image),
        conf=0.2,
        iou=0.2,
        max_det=50,
        imgsz=640,
    )

    print(results[0].boxes)
    print(results[0].names)
    results[0].save(filename=str(output))


if __name__ == "__main__":
    main()
