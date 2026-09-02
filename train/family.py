"""ultron.train.family — job-wide base-model family pin.

Public surface: FamilyName, resolve(), FamilyPack, the three errors.
Hugging Face ids live only in the pack YAMLs. This module stores paths.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn

import yaml


class FamilyName(str, Enum):
    QWEN_4B = "qwen-4b"
    QWEN_8B = "qwen-8b"
    GEMMA = "gemma"


class FamilyError(ValueError):
    """Boundary failure for family selection or pack integrity."""


class UnknownFamilyError(FamilyError):
    """Name is not one of the three closed options."""


class InconsistentFamilyPackError(FamilyError):
    """The three pack YAMLs do not share one Hugging Face id."""


class FamilyOverrideConflictError(FamilyError):
    """ULTRON_BASE_MODEL is set and disagrees with the selected pack."""


@dataclass(frozen=True)
class FamilyPack:
    """Everything one job needs after a family name is chosen.

    Invariants:
    - name is a FamilyName (already parsed).
    - base_model equals model.yaml base_model, train_grpo
      actor_rollout_ref.model.path, and train_dpo model.path.
    - model_config / grpo / dpo paths exist and share one key tree
      with the other shipped packs (enforced by tests, not by this type).
    - chat_template_kwargs is None ⇒ serve omits the vLLM flag.
    - For QWEN_4B, artifact roots are the historical paths. Other
      families are namespaced so two families can coexist on one tree.
    """

    name: FamilyName
    base_model: str
    model_config: Path
    grpo_config_dir: Path
    grpo_config_name: str
    dpo_config: Path
    checkpoint_root: Path
    archive_root: Path
    pfsp_manifest: Path
    max_model_len: int
    gpu_memory_utilization: float
    chat_template_kwargs: dict[str, Any] | None

    def vllm_chat_template_args(self) -> tuple[str, ...]:
        """Ready-to-splice vLLM argv. Empty when the pack has no kwargs."""
        if self.chat_template_kwargs is None:
            return ()
        return ("--chat-template-kwargs", json.dumps(self.chat_template_kwargs))

    def export_environ(self) -> dict[str, str]:
        """Shell-facing projection. Values are unquoted; main() quotes."""
        kwargs = (
            ""
            if self.chat_template_kwargs is None
            else json.dumps(self.chat_template_kwargs, separators=(",", ":"))
        )
        return {
            "ULTRON_MODEL_FAMILY": self.name.value,
            "ULTRON_BASE_MODEL": self.base_model,
            "ULTRON_MODEL_CONFIG": str(self.model_config),
            "ULTRON_GRPO_CONFIG_PATH": str(self.grpo_config_dir),
            "ULTRON_GRPO_CONFIG_NAME": self.grpo_config_name,
            "ULTRON_DPO_CONFIG": str(self.dpo_config),
            "ULTRON_CHECKPOINT_ROOT": str(self.checkpoint_root),
            "ULTRON_ARCHIVE_ROOT": str(self.archive_root),
            "ULTRON_PFSP_MANIFEST": str(self.pfsp_manifest),
            "ULTRON_VLLM_MAX_MODEL_LEN": str(self.max_model_len),
            "ULTRON_VLLM_GPU_MEMORY_UTILIZATION": str(self.gpu_memory_utilization),
            "ULTRON_VLLM_CHAT_TEMPLATE_KWARGS": kwargs,
        }


@dataclass(frozen=True)
class _PackLayout:
    """Private. Paths relative to repo_root. No Hugging Face ids."""

    model_config: Path
    grpo_dir: Path
    grpo_name: str
    dpo_config: Path
    artifact_namespace: str | None  # None ⇒ historical default roots


def parse_family_name(raw: str) -> FamilyName:
    """Boundary parse. Strip only. Case-sensitive. Unknown → UnknownFamilyError."""
    token = raw.strip()
    try:
        return FamilyName(token)
    except ValueError:
        values = [family.value for family in FamilyName]
        expected = f"{', '.join(values[:-1])}, or {values[-1]}"
        raise UnknownFamilyError(f"unknown model family {token!r}; expected {expected}") from None


def resolve(
    name: FamilyName | str | None = None,
    *,
    repo_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> FamilyPack:
    """Select, load, and validate one family pack.

    Precedence: explicit `name` > environ ULTRON_MODEL_FAMILY > qwen-4b.
    Empty / whitespace-only env is treated as unset.
    """
    root = Path(__file__).resolve().parent.parent if repo_root is None else Path(repo_root)
    env = os.environ if environ is None else environ
    selected = _select_name(name, env)
    layout = _layout(selected)
    model_path = root / layout.model_config
    grpo_path = root / layout.grpo_dir / f"{layout.grpo_name}.yaml"
    dpo_path = root / layout.dpo_config
    model = _read_yaml(model_path)
    grpo = _read_yaml(grpo_path)
    dpo = _read_yaml(dpo_path)
    model_id, grpo_id, dpo_id = _hf_ids(model, grpo, dpo)
    if model_id != grpo_id or model_id != dpo_id:
        raise InconsistentFamilyPackError(
            f"{selected.value} Hugging Face ids disagree: "
            f"model.yaml={model_id!r} train_grpo={grpo_id!r} train_dpo={dpo_id!r}"
        )
    override = env.get("ULTRON_BASE_MODEL")
    if override is not None and override.strip() and override.strip() != model_id:
        raise FamilyOverrideConflictError(
            f"ULTRON_BASE_MODEL={override.strip()!r} disagrees with pack base_model={model_id!r}"
        )
    declared = model.get("family")
    if declared is not None and declared != selected.value:
        raise InconsistentFamilyPackError(
            f"{model_path} family={declared!r} does not match {selected.value}"
        )
    serving = model.get("serving")
    if not isinstance(serving, dict):
        raise InconsistentFamilyPackError(f"{model_path} is missing serving")
    try:
        max_model_len = int(serving["max_model_len"])
        gpu_memory_utilization = float(serving["gpu_memory_utilization"])
        grpo_prompt = int(grpo["data"]["max_prompt_length"])
        grpo_response = int(grpo["data"]["max_response_length"])
        dpo_prompt = int(dpo["data"]["max_prompt_length"])
        dpo_response = int(dpo["data"]["max_response_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InconsistentFamilyPackError("pack is missing a context or serving field") from exc
    if grpo_prompt + grpo_response > max_model_len or dpo_prompt + dpo_response > max_model_len:
        raise InconsistentFamilyPackError(
            f"{selected.value} prompt+response exceeds serving.max_model_len={max_model_len}"
        )
    raw_kwargs = serving.get("chat_template_kwargs")
    if raw_kwargs is None:
        chat_kwargs: dict[str, Any] | None = None
    elif isinstance(raw_kwargs, dict):
        chat_kwargs = raw_kwargs
    else:
        raise InconsistentFamilyPackError(
            f"{model_path} serving.chat_template_kwargs must be a mapping or null"
        )
    checkpoint_root, archive_root, pfsp_manifest = _artifact_roots(root, layout.artifact_namespace)
    return FamilyPack(
        name=selected,
        base_model=model_id,
        model_config=model_path,
        grpo_config_dir=(root / layout.grpo_dir).resolve(),
        grpo_config_name=layout.grpo_name,
        dpo_config=dpo_path,
        checkpoint_root=checkpoint_root,
        archive_root=archive_root,
        pfsp_manifest=pfsp_manifest,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        chat_template_kwargs=chat_kwargs,
    )


def _select_name(name: FamilyName | str | None, environ: Mapping[str, str]) -> FamilyName:
    if name is not None:
        if isinstance(name, FamilyName):
            return name
        return parse_family_name(name)
    raw = environ.get("ULTRON_MODEL_FAMILY")
    if raw is None or not raw.strip():
        return FamilyName.QWEN_4B
    return parse_family_name(raw)


def _layout(name: FamilyName) -> _PackLayout:
    """Closed path table. Adding a FamilyName without a case is a type error."""
    match name:
        case FamilyName.QWEN_4B:
            return _PackLayout(
                model_config=Path("configs/model.yaml"),
                grpo_dir=Path("configs"),
                grpo_name="train_grpo",
                dpo_config=Path("configs/train_dpo.yaml"),
                artifact_namespace=None,
            )
        case FamilyName.QWEN_8B:
            return _PackLayout(
                model_config=Path("configs/families/qwen-8b/model.yaml"),
                grpo_dir=Path("configs/families/qwen-8b"),
                grpo_name="train_grpo",
                dpo_config=Path("configs/families/qwen-8b/train_dpo.yaml"),
                artifact_namespace="qwen-8b",
            )
        case FamilyName.GEMMA:
            return _PackLayout(
                model_config=Path("configs/families/gemma/model.yaml"),
                grpo_dir=Path("configs/families/gemma"),
                grpo_name="train_grpo",
                dpo_config=Path("configs/families/gemma/train_dpo.yaml"),
                artifact_namespace="gemma",
            )
    _unreachable(name)


def _unreachable(name: FamilyName) -> NoReturn:
    raise AssertionError(f"unhandled family {name!r}")


def _artifact_roots(repo_root: Path, namespace: str | None) -> tuple[Path, Path, Path]:
    # Default stays data/{checkpoints,archives}. Other families are isolated so
    # they cannot clobber 4B adapters or the historical PFSP file.
    if namespace is None:
        checkpoint_root = repo_root / "data" / "checkpoints"
        archive_root = repo_root / "data" / "archives"
        return checkpoint_root, archive_root, checkpoint_root / "pfsp_pool.json"
    family_root = repo_root / "data" / "families" / namespace
    checkpoint_root = family_root / "checkpoints"
    archive_root = family_root / "archives"
    return checkpoint_root, archive_root, checkpoint_root / "pfsp_pool.json"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise InconsistentFamilyPackError(f"missing pack file {path}")
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise InconsistentFamilyPackError(f"{path} is not a mapping")
    return payload


def _hf_ids(model: dict[str, Any], grpo: dict[str, Any], dpo: dict[str, Any]) -> tuple[str, str, str]:
    """Pull the three pin sites. Missing key → InconsistentFamilyPackError."""
    try:
        model_id = model["base_model"]
        grpo_id = grpo["actor_rollout_ref"]["model"]["path"]
        dpo_id = dpo["model"]["path"]
    except (KeyError, TypeError) as exc:
        raise InconsistentFamilyPackError("pack is missing a Hugging Face id field") from exc
    if not isinstance(model_id, str) or not isinstance(grpo_id, str) or not isinstance(dpo_id, str):
        raise InconsistentFamilyPackError("pack Hugging Face id fields must be strings")
    return model_id, grpo_id, dpo_id


def main(argv: list[str] | None = None) -> int:
    """`python -m ultron.train.family export [--family NAME]`.

    Prints `declare -x KEY=quoted` lines for eval. Exit 2 on FamilyError.
    No other subcommands.
    """
    parser = argparse.ArgumentParser(prog="ultron.train.family")
    parser.add_argument("command", choices=["export"])
    parser.add_argument("--family")
    args = parser.parse_args(argv)
    try:
        pack = resolve(args.family)
    except FamilyError as exc:
        print(exc, file=sys.stderr)
        return 2
    for key, value in pack.export_environ().items():
        print(f"declare -x {key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
