import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from ultron.train.archive import load_base_model
from ultron.train.family import (
    FamilyName,
    FamilyOverrideConflictError,
    FamilyPack,
    InconsistentFamilyPackError,
    UnknownFamilyError,
    main,
    parse_family_name,
    resolve,
)

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = {
    "model": ROOT / "configs" / "model.yaml",
    "grpo": ROOT / "configs" / "train_grpo.yaml",
    "dpo": ROOT / "configs" / "train_dpo.yaml",
}


def _is_additive_model_key(path: str) -> bool:
    return path == "family" or path == "serving.chat_template_kwargs" or path.startswith(
        "serving.chat_template_kwargs."
    )


def _key_tree(payload: object, prefix: str = "") -> set[str]:
    if not isinstance(payload, dict):
        return {prefix} if prefix else set()
    keys: set[str] = set()
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        keys.add(path)
        if isinstance(value, dict):
            keys |= _key_tree(value, path)
    return keys


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict)
    return payload


def _pack_files(name: FamilyName) -> dict[str, Path]:
    if name is FamilyName.QWEN_27B:
        return dict(ORIGINAL)
    folder = ROOT / "configs" / "families" / name.value
    return {
        "model": folder / "model.yaml",
        "grpo": folder / "train_grpo.yaml",
        "dpo": folder / "train_dpo.yaml",
    }


def test_default_is_qwen_27b_with_historical_roots() -> None:
    pack = resolve(environ={})
    assert pack.name is FamilyName.QWEN_27B
    assert pack.base_model == "orcarouter/Qwen3.8-27B-Uncensored-FP8"
    assert pack.model_config == ORIGINAL["model"]
    assert pack.checkpoint_root == ROOT / "data" / "checkpoints"
    assert pack.archive_root == ROOT / "data" / "archives"
    assert pack.pfsp_manifest == ROOT / "data" / "checkpoints" / "pfsp_pool.json"


def test_root_slot_belongs_to_the_default_family() -> None:
    default = resolve(environ={})
    assert default.name is FamilyName.QWEN_27B
    assert default.model_config == ORIGINAL["model"]
    others = [resolve(name, environ={}) for name in FamilyName if name is not default.name]
    assert all(pack.model_config != default.model_config for pack in others)
    assert all(pack.checkpoint_root != default.checkpoint_root for pack in others)
    assert all(pack.archive_root != default.archive_root for pack in others)


def test_empty_or_whitespace_env_is_unset() -> None:
    assert resolve(environ={"ULTRON_MODEL_FAMILY": ""}).name is FamilyName.QWEN_27B
    assert resolve(environ={"ULTRON_MODEL_FAMILY": "  \n"}).name is FamilyName.QWEN_27B


def test_env_selects_family() -> None:
    pack = resolve(environ={"ULTRON_MODEL_FAMILY": "qwen-8b"})
    assert pack.name is FamilyName.QWEN_8B
    assert pack.base_model == "Qwen/Qwen3-8B"


def test_explicit_name_beats_env() -> None:
    pack = resolve("gemma", environ={"ULTRON_MODEL_FAMILY": "qwen-8b"})
    assert pack.name is FamilyName.GEMMA


def test_unknown_name_fails() -> None:
    with pytest.raises(UnknownFamilyError, match="qwen-4b") as exc:
        resolve("llama-8b", environ={})
    assert str(exc.value).endswith("expected qwen-27b, qwen-4b, qwen-8b, or gemma")
    with pytest.raises(UnknownFamilyError):
        parse_family_name("qwen3.5-8b")


def test_each_pack_shares_original_key_tree() -> None:
    original_trees = {kind: _key_tree(_load(path)) for kind, path in ORIGINAL.items()}
    for name in FamilyName:
        files = _pack_files(name)
        for kind, path in files.items():
            tree = _key_tree(_load(path))
            extra = tree - original_trees[kind]
            missing = original_trees[kind] - tree
            if kind == "model":
                assert all(_is_additive_model_key(item) for item in extra | missing)
            else:
                assert extra == set()
                assert missing == set()


def test_three_hf_pins_agree_in_each_pack() -> None:
    for name in FamilyName:
        pack = resolve(name, environ={})
        files = _pack_files(name)
        model_id = _load(files["model"])["base_model"]
        grpo_id = _load(files["grpo"])["actor_rollout_ref"]["model"]["path"]
        dpo_id = _load(files["dpo"])["model"]["path"]
        assert model_id == grpo_id == dpo_id == pack.base_model


def test_qwen_packs_include_thinking_off_and_gemma_omits() -> None:
    qwen_27b = resolve(FamilyName.QWEN_27B, environ={})
    qwen_4b = resolve(FamilyName.QWEN_4B, environ={})
    qwen_8b = resolve(FamilyName.QWEN_8B, environ={})
    gemma = resolve(FamilyName.GEMMA, environ={})
    for pack in (qwen_27b, qwen_4b, qwen_8b):
        args = pack.vllm_chat_template_args()
        assert args[0] == "--chat-template-kwargs"
        assert json.loads(args[1]) == {"enable_thinking": False}
        exported = pack.export_environ()["ULTRON_VLLM_CHAT_TEMPLATE_KWARGS"]
        assert json.loads(exported) == {"enable_thinking": False}
    assert gemma.vllm_chat_template_args() == ()
    assert gemma.export_environ()["ULTRON_VLLM_CHAT_TEMPLATE_KWARGS"] == ""
    assert gemma.chat_template_kwargs is None
    assert gemma.max_model_len == 32768


def test_qwen_4b_is_namespaced_and_keeps_its_id() -> None:
    pack = resolve("qwen-4b", repo_root=ROOT, environ={})
    assert pack.base_model == "Qwen/Qwen3.5-4B"
    assert pack.model_config == ROOT / "configs" / "families" / "qwen-4b" / "model.yaml"
    assert pack.checkpoint_root == ROOT / "data" / "families" / "qwen-4b" / "checkpoints"
    assert pack.archive_root == ROOT / "data" / "families" / "qwen-4b" / "archives"
    assert pack.pfsp_manifest == ROOT / "data" / "families" / "qwen-4b" / "checkpoints" / "pfsp_pool.json"


def test_qwen_8b_id_and_namespaced_roots() -> None:
    pack = resolve("qwen-8b", repo_root=ROOT, environ={})
    assert pack.base_model == "Qwen/Qwen3-8B"
    assert pack.checkpoint_root == ROOT / "data" / "families" / "qwen-8b" / "checkpoints"
    assert pack.archive_root == ROOT / "data" / "families" / "qwen-8b" / "archives"
    assert pack.pfsp_manifest == ROOT / "data" / "families" / "qwen-8b" / "checkpoints" / "pfsp_pool.json"


def test_gemma_roots_are_namespaced() -> None:
    pack = resolve(FamilyName.GEMMA, repo_root=ROOT, environ={})
    assert pack.base_model == "google/gemma-4-12B-it"
    assert pack.checkpoint_root == ROOT / "data" / "families" / "gemma" / "checkpoints"
    assert pack.archive_root == ROOT / "data" / "families" / "gemma" / "archives"


def test_default_artifact_roots_unchanged() -> None:
    pack = resolve(environ={})
    assert pack.checkpoint_root == ROOT / "data" / "checkpoints"
    assert pack.archive_root == ROOT / "data" / "archives"
    assert pack.pfsp_manifest == ROOT / "data" / "checkpoints" / "pfsp_pool.json"
    other = resolve("qwen-8b", environ={})
    assert other.checkpoint_root != pack.checkpoint_root
    assert other.archive_root != pack.archive_root


def test_base_model_override_mismatch_raises() -> None:
    with pytest.raises(FamilyOverrideConflictError, match="Qwen/Qwen3-8B"):
        resolve("qwen-4b", environ={"ULTRON_BASE_MODEL": "Qwen/Qwen3-8B"})


def test_base_model_override_match_is_ok() -> None:
    pack = resolve("qwen-4b", environ={"ULTRON_BASE_MODEL": "Qwen/Qwen3.5-4B"})
    assert pack.base_model == "Qwen/Qwen3.5-4B"


def test_export_environ_keys_and_cli() -> None:
    pack = resolve(environ={})
    exported = pack.export_environ()
    assert exported["ULTRON_MODEL_FAMILY"] == "qwen-27b"
    assert exported["ULTRON_PACK_BASE_MODEL"] == "orcarouter/Qwen3.8-27B-Uncensored-FP8"
    assert exported["ULTRON_GRPO_CONFIG_NAME"] == "train_grpo"
    assert exported["ULTRON_VLLM_MAX_MODEL_LEN"] == "32768"
    assert set(exported) == {
        "ULTRON_MODEL_FAMILY",
        "ULTRON_PACK_BASE_MODEL",
        "ULTRON_MODEL_CONFIG",
        "ULTRON_GRPO_CONFIG_PATH",
        "ULTRON_GRPO_CONFIG_NAME",
        "ULTRON_DPO_CONFIG",
        "ULTRON_CHECKPOINT_ROOT",
        "ULTRON_ARCHIVE_ROOT",
        "ULTRON_PFSP_MANIFEST",
        "ULTRON_VLLM_MAX_MODEL_LEN",
        "ULTRON_VLLM_GPU_MEMORY_UTILIZATION",
        "ULTRON_VLLM_CHAT_TEMPLATE_KWARGS",
    }
    assert main(["export"]) == 0
    assert main(["export", "--family", "not-a-family"]) == 2
    env = os.environ.copy()
    env.pop("ULTRON_MODEL_FAMILY", None)
    env.pop("ULTRON_BASE_MODEL", None)
    result = subprocess.run(
        [sys.executable, "-m", "ultron.train.family", "export"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "declare -x ULTRON_MODEL_FAMILY=qwen-27b" in result.stdout
    assert "declare -x ULTRON_PACK_BASE_MODEL=orcarouter/Qwen3.8-27B-Uncensored-FP8" in result.stdout


def test_lib_family_exports_survive_the_function() -> None:
    script = """
    set -euo pipefail
    source scripts/lib_family.sh
    ultron_load_family
    printf '%s\\n' "${ULTRON_MODEL_FAMILY}" "${ULTRON_PACK_BASE_MODEL}" "${ULTRON_CHECKPOINT_ROOT}"
    """
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k not in {"ULTRON_MODEL_FAMILY", "ULTRON_BASE_MODEL"}},
    )
    assert result.returncode == 0, result.stderr
    family, base, checkpoints = result.stdout.splitlines()
    assert family == "qwen-27b"
    assert base == "orcarouter/Qwen3.8-27B-Uncensored-FP8"
    assert checkpoints.endswith("/data/checkpoints")


def test_lib_family_can_switch_and_rejects_unknown() -> None:
    script = """
    set -euo pipefail
    source scripts/lib_family.sh
    export ULTRON_MODEL_FAMILY=qwen-8b
    ultron_load_family
    printf '%s %s\\n' "${ULTRON_MODEL_FAMILY}" "${ULTRON_PACK_BASE_MODEL}"
    export ULTRON_MODEL_FAMILY=gemma
    ultron_load_family
    printf '%s %s\\n' "${ULTRON_MODEL_FAMILY}" "${ULTRON_PACK_BASE_MODEL}"
    export ULTRON_MODEL_FAMILY=qwen-4b
    ultron_load_family
    printf '%s %s %s\\n' "${ULTRON_MODEL_FAMILY}" "${ULTRON_PACK_BASE_MODEL}" "${ULTRON_CHECKPOINT_ROOT}"
    export ULTRON_MODEL_FAMILY=llama-8b
    if ultron_load_family; then
      echo load-should-fail
      exit 1
    fi
    """
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k not in {"ULTRON_MODEL_FAMILY", "ULTRON_BASE_MODEL"}},
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == "qwen-8b Qwen/Qwen3-8B"
    assert lines[1] == "gemma google/gemma-4-12B-it"
    assert lines[2].startswith("qwen-4b Qwen/Qwen3.5-4B ")
    assert lines[2].endswith("/data/families/qwen-4b/checkpoints")
    assert "unknown model family" in result.stderr


def test_load_base_model_fallback_matches_default_pack(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    assert load_base_model(missing) == "orcarouter/Qwen3.8-27B-Uncensored-FP8"
    assert load_base_model(missing) == resolve(environ={}).base_model
    assert load_base_model(missing) == _load(ORIGINAL["model"])["base_model"]


def test_context_budget_must_fit_window(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    model = _load(ORIGINAL["model"])
    grpo = _load(ORIGINAL["grpo"])
    dpo = _load(ORIGINAL["dpo"])
    model["serving"]["max_model_len"] = 100
    (configs / "model.yaml").write_text(yaml.safe_dump(model))
    (configs / "train_grpo.yaml").write_text(yaml.safe_dump(grpo))
    (configs / "train_dpo.yaml").write_text(yaml.safe_dump(dpo))
    with pytest.raises(InconsistentFamilyPackError, match="max_model_len"):
        resolve("qwen-27b", repo_root=tmp_path, environ={})


def test_shipped_packs_fit_context_window() -> None:
    for name in FamilyName:
        pack = resolve(name, environ={})
        files = _pack_files(name)
        grpo = _load(files["grpo"])
        dpo = _load(files["dpo"])
        assert (
            grpo["data"]["max_prompt_length"] + grpo["data"]["max_response_length"]
            <= pack.max_model_len
        )
        assert (
            dpo["data"]["max_prompt_length"] + dpo["data"]["max_response_length"]
            <= pack.max_model_len
        )


def test_pack_is_frozen() -> None:
    pack = resolve(environ={})
    assert isinstance(pack, FamilyPack)
    assert pack.__dataclass_params__.frozen is True
