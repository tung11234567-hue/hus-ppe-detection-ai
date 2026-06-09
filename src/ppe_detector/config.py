from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .rules import RuleConfig


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_rule_config(config_path: str | Path | None = None) -> RuleConfig:
    if config_path is None:
        return RuleConfig()
    cfg = load_yaml(config_path)
    rules = cfg.get("rules", {})
    classes = cfg.get("classes", {})
    return RuleConfig(
        head_zone_ratio=float(rules.get("head_zone_ratio", 0.45)),
        torso_y1_ratio=float(rules.get("torso_y1_ratio", 0.25)),
        torso_y2_ratio=float(rules.get("torso_y2_ratio", 0.85)),
        min_overlap_ratio=float(rules.get("min_overlap_ratio", 0.05)),
        person_aliases=list(classes.get("person_aliases", ["person", "worker", "employee"])),
        helmet_aliases=list(classes.get("helmet_aliases", ["helmet", "hardhat", "hard_hat", "safety_helmet"])),
        vest_aliases=list(classes.get("vest_aliases", ["vest", "safety_vest", "reflective_vest", "high_vis_vest"])),
        no_helmet_aliases=list(classes.get("no_helmet_aliases", ["no_helmet", "no-helmet", "no_hardhat", "no-hardhat"])),
        no_vest_aliases=list(classes.get("no_vest_aliases", ["no_vest", "no-vest", "no_safety_vest", "no-safety-vest"])),
    )
