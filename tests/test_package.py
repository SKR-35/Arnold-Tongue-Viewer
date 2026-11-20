"""Bootstrap smoke tests for the installed distribution."""

from importlib.metadata import version

import arnold_tongue_viewer


def test_package_import() -> None:
    assert arnold_tongue_viewer is not None


def test_package_version() -> None:
    assert version("arnold-tongue-viewer") == "0.1.0"
