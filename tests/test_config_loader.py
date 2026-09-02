from pathlib import Path

from marsdog_vision_interaction.utils.config_loader import load_config


def _write_config(path: Path) -> None:
    path.write_text(
        "model: ${MARSDOG_VISION_MODEL_DIR}/model.task\n"
        "data: ${MARSDOG_VISION_DATA_DIR}/faces\n",
        encoding="utf-8",
    )


def test_path_variables_honor_environment_overrides(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "vision.yaml"
    _write_config(config_path)
    model_dir = tmp_path / "external-models"
    data_dir = tmp_path / "runtime-data"
    monkeypatch.setenv("MARSDOG_VISION_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("MARSDOG_VISION_DATA_DIR", str(data_dir))

    config = load_config(config_path)

    assert config["model"] == str(model_dir / "model.task")
    assert config["data"] == str(data_dir / "faces")


def test_path_variables_default_to_checkout_directories(tmp_path, monkeypatch) -> None:
    project_dir = tmp_path / "checkout"
    config_dir = project_dir / "config"
    package_dir = project_dir / "marsdog_vision_interaction"
    model_dir = project_dir / "models" / "vision"
    config_dir.mkdir(parents=True)
    package_dir.mkdir()
    model_dir.mkdir(parents=True)
    (project_dir / "pyproject.toml").write_text("[project]\nname='test'\n")
    config_path = config_dir / "vision.yaml"
    _write_config(config_path)
    for name in (
        "MARSDOG_VISION_PROJECT_DIR",
        "MARSDOG_VISION_MODEL_DIR",
        "MARSDOG_VISION_DATA_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    config = load_config(config_path)

    assert config["model"] == str(model_dir / "model.task")
    assert config["data"] == str(project_dir / "data" / "faces")
