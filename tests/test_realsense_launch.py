from pathlib import Path


def test_vision_subscribes_to_external_realsense_stream() -> None:
    root = Path(__file__).parents[1]
    launch_source = (root / "launch" / "vision.launch.py").read_text(
        encoding="utf-8"
    )
    config_source = (root / "config" / "vision.yaml").read_text(
        encoding="utf-8"
    )

    assert "IncludeLaunchDescription" not in launch_source
    assert "realsense2_camera" not in launch_source
    assert "camera_image: /camera/camera/color/image_raw" in config_source
