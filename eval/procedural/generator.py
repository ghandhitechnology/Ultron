from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class ProceduralTemplate:
    template_id: str
    family: str
    weight: float
    misconfig_ids: tuple[str, ...]


def load_template(path: Path) -> ProceduralTemplate:
    raw: Any = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: template must be an object")
    template_id = raw.get("id")
    family = raw.get("family")
    weight = raw.get("weight")
    ids = raw.get("misconfig_ids")
    if not isinstance(template_id, str) or not isinstance(family, str):
        raise ValueError(f"{path}: id and family must be strings")
    if not isinstance(weight, (int, float)) or not 0 < float(weight) <= 1:
        raise ValueError(f"{path}: weight must be in (0, 1]")
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError(f"{path}: misconfig_ids must be strings")
    return ProceduralTemplate(template_id, family, float(weight), tuple(ids))


def choose_templates(
    templates: Iterable[ProceduralTemplate], count: int, *, seed: int
) -> list[ProceduralTemplate]:
    candidates = list(templates)
    if count < 0 or count > len(candidates):
        raise ValueError("count must fit the template pool")
    rng = random.Random(seed)
    chosen: list[ProceduralTemplate] = []
    while len(chosen) < count:
        weights = [template.weight for template in candidates]
        selected = rng.choices(candidates, weights=weights, k=1)[0]
        chosen.append(selected)
        candidates.remove(selected)
    return chosen
