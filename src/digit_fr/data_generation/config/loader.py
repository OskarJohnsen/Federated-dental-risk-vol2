from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Tuple
from ...core.paths import root_path
from ..profiles.generator import generate_client_profiles

_CACHE: Dict[str, Any] = {}

def _read_json(path: Path) -> Any:
    with path.open("r") as f:
         return json.load(f)

def _validate_risk_stats(cfg: Dict[str, Any]) -> None:
    # basic check because risks got "lost" while restructuring code
    for key in ("AlveolarOsteitis", "SecondaryInfection", "NerveDysesthesia", "Bleeding"):
        if key not in cfg:
            raise ValueError(f"risk_stats missing section: {key}")

def load_all_configs(force_reload: bool = False) -> Dict[str, Any]:
    """Load and validate all configs; cached by default."""
    global _CACHE
    if _CACHE and not force_reload:
        return _CACHE

    cdir = root_path("configs")
    generation = _read_json(cdir / "generation_config.json")
    extraction_types = _read_json(cdir / "extraction_type_stats.json")
    extraction_binary = _read_json(cdir / "extraction_binary_stats.json")
    noise = _read_json(cdir / "noise_config.json")
    risks = _read_json(cdir / "risk_stats.json")

    _validate_risk_stats(risks)

    if "client_profiles" in generation:
        profile_config = generation["client_profiles"]
        n_profiles = profile_config.get("n_profiles", generation["dataset"]["n_clients"])
        ranges_config = profile_config.get("ranges", {})
        seed = profile_config.get("seed", generation["dataset"].get("random_seed", 42))
        
        client_profiles = generate_client_profiles(n_profiles, ranges_config, seed)
    else:
        print("Warning: Using default client profiles")
        client_profiles = _read_json(cdir / "client_profiles.json")

    _CACHE = {
        "generation": generation,
        "extraction_types": extraction_types,
        "extraction_binary": extraction_binary,
        "client_profiles": client_profiles,
        "noise": noise,
        "risks": risks,
    }
    return _CACHE