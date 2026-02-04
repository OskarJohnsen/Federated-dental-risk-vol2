from pathlib import Path
import os

def get_project_root() -> Path:
    """Resolve the repository root robustly.
    """
    env_root = os.getenv("FDRP_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if p.exists():
            return p

    cwd = Path.cwd().resolve()
    for up in [cwd, *cwd.parents]:
        if (up / "pyproject.toml").exists() and (up / "configs").is_dir():
            return up

    here = Path(__file__).resolve()
    for up in [*here.parents]:
        if (up / "configs").is_dir():
            return up

    return cwd

def root_path(*parts: str) -> Path:
    """Build a path relative to the project root."""
    return get_project_root().joinpath(*parts)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)