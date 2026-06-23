"""Clean source-package export helpers for LakePINN v10."""

from __future__ import annotations

import fnmatch
import zipfile
from pathlib import Path


DEFAULT_ALLOWED_ROOTS = (
    'lake_pinn',
    'tests',
    'experiments/splits',
    'experiments/manifests_clean',
    'scripts',
)
DEFAULT_FILES = (
    '.gitignore',
    '.source_packageignore',
)
DEFAULT_EXCLUDES = (
    'results/',
    'checkpoints/',
    'raw_remote/',
    'remote_jobs/',
    '.git/',
    '.pytest_cache/',
    '.mypy_cache/',
    '.ruff_cache/',
    '__pycache__/',
    '*.pyc',
    '*.pyo',
    '*.pyd',
    '*.pt',
    '*.pth',
    '*.ckpt',
    '*.zip',
    '*.tar',
    '*.tar.gz',
)


def load_source_package_excludes(root: Path) -> tuple[str, ...]:
    """Read ``.source_packageignore`` or return conservative defaults."""

    root = Path(root)
    ignore_path = root / '.source_packageignore'
    if not ignore_path.exists():
        return DEFAULT_EXCLUDES
    patterns = []
    for raw_line in ignore_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        patterns.append(line.replace('\\', '/'))
    return tuple(patterns) or DEFAULT_EXCLUDES


def is_source_package_excluded(relative_path: Path, patterns: tuple[str, ...]) -> bool:
    """Return true when a relative path matches source-package exclusions."""

    relative_path = Path(relative_path)
    text = relative_path.as_posix()
    name = relative_path.name
    parts = set(relative_path.parts)
    for pattern in patterns:
        normalized = pattern.rstrip('/')
        if pattern.endswith('/'):
            if normalized in parts or text == normalized or text.startswith(f'{normalized}/'):
                return True
            continue
        if fnmatch.fnmatch(text, pattern) or fnmatch.fnmatch(name, pattern):
            return True
    return False


def iter_source_files(
    root: Path,
    *,
    allowed_roots: tuple[str, ...] = DEFAULT_ALLOWED_ROOTS,
    include_files: tuple[str, ...] = DEFAULT_FILES,
    excludes: tuple[str, ...] | None = None,
) -> list[Path]:
    """Return packageable source files relative to ``root``."""

    root = Path(root)
    patterns = load_source_package_excludes(root) if excludes is None else excludes
    files: list[Path] = []
    for root_name in allowed_roots:
        current = root / root_name
        if not current.exists():
            continue
        for path in current.rglob('*'):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if not is_source_package_excluded(relative, patterns):
                files.append(relative)
    for file_name in include_files:
        path = root / file_name
        if path.is_file():
            relative = path.relative_to(root)
            if not is_source_package_excluded(relative, patterns):
                files.append(relative)
    return sorted(set(files), key=lambda item: item.as_posix())


def export_source_package(root: Path, output_path: Path, *, dry_run: bool = False) -> list[Path]:
    """Write a clean LakePINN source zip and return included relative paths."""

    root = Path(root)
    output_path = Path(output_path)
    files = iter_source_files(root)
    if dry_run:
        return files
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            archive.write(root / relative, relative.as_posix())
    return files
