"""The package, the project metadata and the API must report one version."""

import tomllib

from evaluator import __version__
from evaluator.config import REPO_ROOT
from evaluator.serving.app import app


def test_package_and_project_metadata_agree():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    assert project["version"] == __version__


def test_api_reports_the_package_version():
    assert app.version == __version__
