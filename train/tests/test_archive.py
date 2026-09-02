import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from ultron.train.archive import (
    archive_all_generations,
    archive_generation,
    find_adapter_root,
    find_all_adapter_roots,
    select_adapter,
)
from ultron.train.family import resolve
from ultron.train.pfsp import load_pool
from ultron.train.schema_v1 import Role


def write_adapter(path: Path, marker: str) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(json.dumps({"task_type": "CAUSAL_LM", "r": 64}))
    (path / "adapter_model.safetensors").write_bytes(marker.encode())


def test_find_adapter_prefers_highest_global_step(tmp_path: Path) -> None:
    write_adapter(tmp_path / "global_step_3" / "actor", "old")
    write_adapter(tmp_path / "global_step_12" / "actor", "new")
    chosen = find_adapter_root(tmp_path)
    assert chosen is not None
    assert chosen.as_posix().endswith("global_step_12/actor")
    assert len(find_all_adapter_roots(tmp_path)) == 2


def test_select_adapter_prefers_dpo(tmp_path: Path) -> None:
    write_adapter(tmp_path / "gen2" / "attacker_lora" / "actor", "grpo")
    write_adapter(tmp_path / "gen2" / "attacker_lora_dpo", "dpo")
    selected = select_adapter(tmp_path, 2, Role.ATTACKER)
    assert selected.stage == "dpo"
    assert selected.root == tmp_path / "gen2" / "attacker_lora_dpo"


def test_archive_generation_copies_every_checkpoint_and_final_sh(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    archives = tmp_path / "archives"
    model_config = tmp_path / "model.yaml"
    pfsp = tmp_path / "pfsp.json"
    write_adapter(checkpoints / "gen1" / "attacker_lora" / "global_step_1" / "actor", "atk1")
    write_adapter(checkpoints / "gen1" / "attacker_lora" / "global_step_8" / "actor", "atk8")
    write_adapter(checkpoints / "gen1" / "attacker_lora_dpo", "dpo")
    write_adapter(checkpoints / "gen1" / "defender_lora" / "global_step_4" / "actor", "def")
    model_config.write_text("base_model: Qwen/Qwen3.5-4B\n")

    manifest = archive_generation(
        1,
        checkpoint_root=checkpoints,
        archive_root=archives,
        model_config=model_config,
        pfsp_manifest=pfsp,
    )

    assert (archives / "gen1" / "attacker_lora" / "adapter_model.safetensors").read_bytes() == b"dpo"
    assert (archives / "gen1" / "defender_lora" / "adapter_model.safetensors").read_bytes() == b"def"
    assert (
        archives
        / "gen1"
        / "checkpoints"
        / "attacker"
        / "grpo"
        / "global_step_1"
        / "actor"
        / "adapter_model.safetensors"
    ).read_bytes() == b"atk1"
    assert (
        archives
        / "gen1"
        / "checkpoints"
        / "attacker"
        / "grpo"
        / "global_step_8"
        / "actor"
        / "adapter_model.safetensors"
    ).read_bytes() == b"atk8"
    assert len(manifest["checkpoints"]) == 4
    assert manifest["roles"]["attacker"]["stage"] == "dpo"

    final = archives / "FINAL.sh"
    assert final.is_file()
    listed = subprocess.check_output([str(final)], text=True)
    assert "generation 1" in listed
    attacker = subprocess.check_output([str(final), "attacker"], text=True).strip()
    assert Path(attacker).joinpath("adapter_model.safetensors").read_bytes() == b"dpo"
    ckpts = subprocess.check_output([str(final), "checkpoints"], text=True)
    assert ckpts.count("adapter_config.json") >= 1

    with tarfile.open(archives / "ultron-gen1.tar") as archive:
        names = archive.getnames()
    assert "FINAL.sh" in names
    assert "ultron-gen1/FINAL.sh" in names

    pool = load_pool(pfsp)
    assert {entry.checkpoint_id for entry in pool} == {"attacker-gen1", "defender-gen1"}


def test_archive_all_generations_skips_incomplete(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    archives = tmp_path / "archives"
    write_adapter(checkpoints / "gen0" / "attacker_lora", "a0")
    write_adapter(checkpoints / "gen0" / "defender_lora", "d0")
    write_adapter(checkpoints / "gen1" / "attacker_lora", "a1")
    payload = archive_all_generations(
        checkpoint_root=checkpoints,
        archive_root=archives,
        model_config=tmp_path / "missing.yaml",
        pfsp_manifest=tmp_path / "pfsp.json",
        pack=False,
    )
    assert payload["archived"] == [0]
    assert payload["skipped"]
    assert (archives / "FINAL.sh").is_file()
    assert (archives / "final" / "attacker_lora" / "adapter_model.safetensors").read_bytes() == b"a0"


def test_archive_generation_requires_both_roles(tmp_path: Path) -> None:
    write_adapter(tmp_path / "gen0" / "attacker_lora", "atk")
    with pytest.raises(FileNotFoundError, match="defender"):
        archive_generation(
            0,
            checkpoint_root=tmp_path,
            archive_root=tmp_path / "archives",
            model_config=tmp_path / "missing.yaml",
            pfsp_manifest=tmp_path / "pfsp.json",
            pack=False,
        )


def test_default_resolve_reads_repo_model_yaml() -> None:
    pack = resolve(environ={})
    assert pack.model_config.name == "model.yaml"
    assert pack.model_config.parent.name == "configs"
    assert pack.base_model == "Qwen/Qwen3.5-4B"
