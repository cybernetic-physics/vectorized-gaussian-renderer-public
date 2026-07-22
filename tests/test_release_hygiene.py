from __future__ import annotations

import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_distribution_metadata_and_license_payload_are_explicit() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    metadata = project["project"]

    assert metadata["license"] == "Apache-2.0"
    assert metadata["authors"] == [{"name": "Cybernetic Physics"}]
    assert metadata["readme"]["file"] == "README.md"
    assert metadata["urls"]["Repository"] == (
        "https://github.com/cybernetic-physics/vectorized-gaussian-renderer"
    )

    required_license_files = {
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "src/isaacsim_gaussian_renderer/native/flashgs/LICENSE.flashgs",
    }
    assert set(metadata["license-files"]) == required_license_files
    assert all((ROOT / path).is_file() for path in required_license_files)

    package_data = project["tool"]["setuptools"]["package-data"]
    assert "native/*.cu" in package_data["isaacsim_gaussian_renderer"]
    assert "native/flashgs/LICENSE.flashgs" in package_data["isaacsim_gaussian_renderer"]


def test_public_scene_manifests_have_complete_attribution() -> None:
    home = json.loads(
        (ROOT / "datasets/home-scan-lod0.manifest.json").read_text(encoding="utf-8")
    )["source"]
    garage = json.loads(
        (ROOT / "datasets/garage-high-detail.manifest.json").read_text(encoding="utf-8")
    )["attribution"]

    expected_home = {
        "asset_title": "Home Scan (Creation process in description)",
        "author": "Isaiah Sweeney",
        "source_url": "https://superspl.at/scene/3f89bbd3",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
    }
    assert {key: home[key] for key in expected_home} == expected_home
    assert garage == {
        "author": "Ethan",
        "author_url": "https://superspl.at/user?id=ethan3111",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_url": "https://superspl.at/scene/8f4d8e05",
        "title": "Garage - High Detail Test",
    }
    assert "local_path" not in home


def test_results_log_cannot_present_historical_ovrtx_as_current_headline() -> None:
    results = (ROOT / "RESULTS.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "PASS headline" not in results
    assert "Publication status (2026-07-21)" in results
    assert "same-source, same-GPU" in results
    assert "/Users/" not in results
    assert "[Apache License 2.0](LICENSE)" in readme
    assert "[the third-party notices](THIRD_PARTY_NOTICES.md)" in readme
