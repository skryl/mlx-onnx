import re
from pathlib import Path

from invoke import task


REPO_ROOT = Path(__file__).resolve().parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
README_PATH = REPO_ROOT / "README.md"


def _bump_four_part_version(version: str) -> str:
    parts = version.split(".")
    if len(parts) != 4 or any(not part.isdigit() for part in parts):
        raise ValueError(
            f"expected four-part numeric version 'X.Y.Z.W', got '{version}'"
        )
    major, minor, patch, build = (int(part) for part in parts)
    return f"{major}.{minor}.{patch}.{build + 1}"


@task
def test(c):
    """Run the full Python test suite."""
    c.run(
        "python -m pytest python/tests -q --ignore=python/tests/test_examples.py",
        pty=True,
    )
    c.run(
        (
            "python -c \"import sys, pytest, mlx.core as mx; "
            "mx.set_default_device(mx.cpu); "
            "sys.exit(pytest.main(['python/tests/test_examples.py', '-q']))\""
        ),
        pty=True,
    )
    c.run(
        (
            "python -c \"import sys, pytest, mlx.core as mx; "
            "mx.set_default_device(mx.gpu); "
            "sys.exit(pytest.main(['python/tests/test_examples.py', '-q']))\""
        ),
        pty=True,
    )


@task
def bump_version(c):
    """Bump project version by 0.0.0.1 in pyproject.toml and README badge."""
    _ = c
    pyproject_text = PYPROJECT_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"(?m)^version\s*=\s*\"(?P<version>\d+\.\d+\.\d+\.\d+)\"\s*$",
        pyproject_text,
    )
    if match is None:
        raise RuntimeError("could not locate version in pyproject.toml")

    old_version = match.group("version")
    new_version = _bump_four_part_version(old_version)
    updated_pyproject = pyproject_text[: match.start("version")] + new_version + pyproject_text[match.end("version") :]
    PYPROJECT_PATH.write_text(updated_pyproject, encoding="utf-8")

    readme_text = README_PATH.read_text(encoding="utf-8")
    readme_updated = re.sub(
        r"(img\.shields\.io/badge/version-)\d+\.\d+\.\d+\.\d+(-blue)",
        rf"\g<1>{new_version}\2",
        readme_text,
        count=1,
    )
    if readme_text != readme_updated:
        README_PATH.write_text(readme_updated, encoding="utf-8")

    print(f"Bumped version: {old_version} -> {new_version}")
