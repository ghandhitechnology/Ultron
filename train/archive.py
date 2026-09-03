from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .family import FamilyError, resolve
from .pfsp import PoolEntry, load_pool, save_pool, update_pool
from .schema_v1 import Role

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
DEFAULT_CHECKPOINT_ROOT = Path("data/checkpoints")
DEFAULT_ARCHIVE_ROOT = Path("data/archives")
DEFAULT_MODEL_CONFIG = Path("configs/model.yaml")
DEFAULT_PFSP_MANIFEST = Path("data/checkpoints/pfsp_pool.json")
FINAL_SCRIPT_NAME = "FINAL.sh"


@dataclass(frozen=True)
class SelectedAdapter:
    role: Role
    stage: str
    source: Path
    root: Path


@dataclass(frozen=True)
class CheckpointCopy:
    role: Role
    stage: str
    source: Path
    root: Path
    dest: Path
    step: int
    files: list[dict[str, Any]]


def is_adapter_dir(path: Path) -> bool:
    if not (path / ADAPTER_CONFIG).is_file():
        return False
    return any((path / name).is_file() for name in ADAPTER_WEIGHTS)


def _global_step(path: Path) -> int:
    step = -1
    for part in path.parts:
        if not part.startswith("global_step_"):
            continue
        suffix = part.removeprefix("global_step_")
        if suffix.isdigit():
            step = int(suffix)
    return step


def find_all_adapter_roots(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if is_adapter_dir(path):
        return [path]
    found = {
        candidate.parent
        for candidate in path.rglob(ADAPTER_CONFIG)
        if is_adapter_dir(candidate.parent)
    }
    return sorted(found, key=lambda item: (_global_step(item), item.as_posix()))


def find_adapter_root(path: Path) -> Path | None:
    found = find_all_adapter_roots(path)
    if not found:
        return None
    return max(found, key=lambda item: (_global_step(item), len(item.parts)))


def adapter_candidates(checkpoint_root: Path, generation: int, role: Role) -> list[tuple[str, Path]]:
    prefix = checkpoint_root / f"gen{generation}" / f"{role.value}_lora"
    return [("dpo", Path(f"{prefix}_dpo")), ("grpo", prefix)]


def select_adapter(checkpoint_root: Path, generation: int, role: Role) -> SelectedAdapter:
    for stage, source in adapter_candidates(checkpoint_root, generation, role):
        root = find_adapter_root(source)
        if root is not None:
            return SelectedAdapter(role=role, stage=stage, source=source, root=root)
    searched = ", ".join(str(source) for _, source in adapter_candidates(checkpoint_root, generation, role))
    raise FileNotFoundError(f"no {role.value} LoRA under {searched}")


def discover_generations(checkpoint_root: Path) -> list[int]:
    if not checkpoint_root.is_dir():
        return []
    generations: list[int] = []
    for child in checkpoint_root.iterdir():
        if child.is_dir() and child.name.startswith("gen") and child.name[3:].isdigit():
            generations.append(int(child.name[3:]))
    return sorted(set(generations))


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_adapter_files(source: Path, destination: Path) -> list[dict[str, Any]]:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for item in sorted(source.iterdir()):
        if not item.is_file():
            continue
        target = destination / item.name
        shutil.copy2(item, target)
        records.append({"name": item.name, "sha256": file_digest(target), "bytes": target.stat().st_size})
    if not any(record["name"] == ADAPTER_CONFIG for record in records):
        raise FileNotFoundError(f"{source} is missing {ADAPTER_CONFIG}")
    return records


def _checkpoint_dest(target: Path, role: Role, stage: str, source: Path, root: Path) -> Path:
    base = target / "checkpoints" / role.value / stage
    try:
        relative = root.relative_to(source)
    except ValueError:
        relative = Path(root.name)
    if relative == Path("."):
        return base
    return base / relative


def collect_generation_checkpoints(checkpoint_root: Path, generation: int, target: Path) -> list[CheckpointCopy]:
    copies: list[CheckpointCopy] = []
    for role in (Role.ATTACKER, Role.DEFENDER):
        for stage, source in adapter_candidates(checkpoint_root, generation, role):
            for root in find_all_adapter_roots(source):
                dest = _checkpoint_dest(target, role, stage, source, root)
                copies.append(
                    CheckpointCopy(
                        role=role,
                        stage=stage,
                        source=source,
                        root=root,
                        dest=dest,
                        step=_global_step(root),
                        files=copy_adapter_files(root, dest),
                    )
                )
    return copies


def load_base_model(config: Path = DEFAULT_MODEL_CONFIG) -> str:
    if not config.is_file():
        return "orcarouter/Qwen3.8-27B-Uncensored-FP8"
    payload = yaml.safe_load(config.read_text()) or {}
    return str(payload.get("base_model") or "orcarouter/Qwen3.8-27B-Uncensored-FP8")


def archive_dir(archive_root: Path, generation: int) -> Path:
    return archive_root / f"gen{generation}"


def write_index(archive_root: Path, generation: int) -> Path:
    generations: list[int] = []
    for child in archive_root.iterdir():
        if child.is_dir() and child.name.startswith("gen") and child.name[3:].isdigit():
            generations.append(int(child.name[3:]))
    generations = sorted(set(generations))
    index_path = archive_root / "index.json"
    index_path.write_text(
        json.dumps({"latest": generation, "generations": generations, "final": FINAL_SCRIPT_NAME}, indent=2)
        + "\n"
    )
    latest = archive_root / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(f"gen{generation}")
    return index_path


def write_final_script(path: Path, generation: int, checkpoint_count: int) -> None:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"# Ultron final adapters for generation {generation}.",
                'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
                f"GEN={generation}",
                f"CHECKPOINTS={checkpoint_count}",
                "if [[ -d \"$HERE/final/attacker_lora\" ]]; then",
                "  ROOT=\"$HERE/final\"",
                "elif [[ -d \"$HERE/attacker_lora\" ]]; then",
                "  ROOT=\"$HERE\"",
                "else",
                "  echo \"FINAL.sh: missing attacker_lora next to this script\" >&2",
                "  exit 2",
                "fi",
                "case \"${1:-}\" in",
                "  attacker|defender)",
                "    printf '%s\\n' \"$ROOT/${1}_lora\"",
                "    ;;",
                "  checkpoints)",
                "    for dir in \"$HERE/checkpoints\" \"$HERE/gen${GEN}/checkpoints\" \"$HERE/ultron-gen${GEN}/checkpoints\"; do",
                "      if [[ -d \"$dir\" ]]; then",
                "        find \"$dir\" -name adapter_config.json -print | sort",
                "        exit 0",
                "      fi",
                "    done",
                "    ;;",
                "  \"\")",
                "    printf 'generation %s\\n' \"$GEN\"",
                "    printf 'attacker %s\\n' \"$ROOT/attacker_lora\"",
                "    printf 'defender %s\\n' \"$ROOT/defender_lora\"",
                "    printf 'checkpoints %s\\n' \"$CHECKPOINTS\"",
                "    ;;",
                "  *)",
                "    echo \"usage: FINAL.sh [attacker|defender|checkpoints]\" >&2",
                "    exit 2",
                "    ;;",
                "esac",
                "",
            ]
        )
    )
    path.chmod(0o755)


def publish_final(archive_root: Path, generation: int, roles: dict[str, Any], checkpoint_count: int) -> Path:
    final_dir = archive_root / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True)
    source_gen = archive_dir(archive_root, generation)
    for role_name in roles:
        copy_adapter_files(source_gen / f"{role_name}_lora", final_dir / f"{role_name}_lora")
    write_final_script(archive_root / FINAL_SCRIPT_NAME, generation, checkpoint_count)
    write_final_script(source_gen / FINAL_SCRIPT_NAME, generation, checkpoint_count)
    return archive_root / FINAL_SCRIPT_NAME


def _family_label(model_config: Path, resolved_family: str | None) -> str | None:
    if resolved_family is not None:
        return resolved_family
    if not model_config.is_file():
        return None
    payload = yaml.safe_load(model_config.read_text()) or {}
    family = payload.get("family") if isinstance(payload, dict) else None
    if family is None:
        return None
    return str(family)


def archive_generation(
    generation: int,
    *,
    checkpoint_root: Path | None = None,
    archive_root: Path | None = None,
    model_config: Path | None = None,
    pfsp_manifest: Path | None = None,
    pack: bool = True,
    publish: bool = True,
) -> dict[str, Any]:
    if generation < 0:
        raise ValueError("generation must be >= 0")
    resolved_family: str | None = None
    if checkpoint_root is None or archive_root is None or model_config is None or pfsp_manifest is None:
        family_pack = resolve()
        resolved_family = family_pack.name.value
        if checkpoint_root is None:
            checkpoint_root = family_pack.checkpoint_root
        if archive_root is None:
            archive_root = family_pack.archive_root
        if model_config is None:
            model_config = family_pack.model_config
        if pfsp_manifest is None:
            pfsp_manifest = family_pack.pfsp_manifest
    selected = [select_adapter(checkpoint_root, generation, role) for role in (Role.ATTACKER, Role.DEFENDER)]
    target = archive_dir(archive_root, generation)
    target.mkdir(parents=True, exist_ok=True)
    checkpoints = collect_generation_checkpoints(checkpoint_root, generation, target)
    roles: dict[str, Any] = {}
    for adapter in selected:
        dest = target / f"{adapter.role.value}_lora"
        files = copy_adapter_files(adapter.root, dest)
        roles[adapter.role.value] = {
            "stage": adapter.stage,
            "source": str(adapter.source),
            "adapter_root": str(adapter.root),
            "path": str(dest),
            "files": files,
        }
    manifest = {
        "generation": generation,
        "base_model": load_base_model(model_config),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "final": FINAL_SCRIPT_NAME,
        "roles": roles,
        "checkpoints": [
            {
                "role": item.role.value,
                "stage": item.stage,
                "step": item.step,
                "source": str(item.root),
                "path": str(item.dest),
                "files": item.files,
            }
            for item in checkpoints
        ],
    }
    family = _family_label(model_config, resolved_family)
    if family is not None:
        manifest["family"] = family
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_final_script(target / FINAL_SCRIPT_NAME, generation, len(checkpoints))
    write_index(archive_root, generation)
    if publish:
        final_path = publish_final(archive_root, generation, roles, len(checkpoints))
        manifest["final_script"] = str(final_path)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if pack:
        tarball = archive_root / f"ultron-gen{generation}.tar"
        with tarfile.open(tarball, "w") as archive:
            archive.add(target, arcname=f"ultron-gen{generation}")
            final_script = archive_root / FINAL_SCRIPT_NAME
            if final_script.is_file():
                archive.add(final_script, arcname=FINAL_SCRIPT_NAME)
            final_dir = archive_root / "final"
            if final_dir.is_dir():
                archive.add(final_dir, arcname="final")
        manifest["tarball"] = str(tarball)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    _update_pfsp(pfsp_manifest, generation, roles)
    return manifest


def archive_all_generations(
    *,
    checkpoint_root: Path | None = None,
    archive_root: Path | None = None,
    model_config: Path | None = None,
    pfsp_manifest: Path | None = None,
    pack: bool = True,
) -> dict[str, Any]:
    if checkpoint_root is None or archive_root is None or model_config is None or pfsp_manifest is None:
        family_pack = resolve()
        if checkpoint_root is None:
            checkpoint_root = family_pack.checkpoint_root
        if archive_root is None:
            archive_root = family_pack.archive_root
        if model_config is None:
            model_config = family_pack.model_config
        if pfsp_manifest is None:
            pfsp_manifest = family_pack.pfsp_manifest
    generations = discover_generations(checkpoint_root)
    if not generations:
        raise FileNotFoundError(f"no genN directories under {checkpoint_root}")
    archived: list[int] = []
    last: dict[str, Any] | None = None
    errors: list[str] = []
    for generation in generations:
        try:
            last = archive_generation(
                generation,
                checkpoint_root=checkpoint_root,
                archive_root=archive_root,
                model_config=model_config,
                pfsp_manifest=pfsp_manifest,
                pack=pack,
                publish=False,
            )
            archived.append(generation)
        except FileNotFoundError as exc:
            errors.append(f"gen{generation}: {exc}")
    if last is None:
        raise FileNotFoundError("; ".join(errors) or f"no complete generations under {checkpoint_root}")
    publish_final(archive_root, archived[-1], last["roles"], len(last["checkpoints"]))
    write_index(archive_root, archived[-1])
    return {
        "archived": archived,
        "skipped": errors,
        "final": str(archive_root / FINAL_SCRIPT_NAME),
        "latest": last,
    }


def _update_pfsp(manifest_path: Path, generation: int, roles: dict[str, Any]) -> None:
    entries = load_pool(manifest_path)
    for role_name, info in roles.items():
        role = Role(role_name)
        entries = update_pool(
            entries,
            PoolEntry(
                checkpoint_id=f"{role.value}-gen{generation}",
                path=str(info["path"]),
                role=role,
                win_rate_vs_live=0.5,
            ),
        )
    save_pool(manifest_path, generation, entries)


def list_archives(archive_root: Path = DEFAULT_ARCHIVE_ROOT) -> dict[str, Any]:
    index_path = archive_root / "index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text())
        if isinstance(payload, dict):
            return payload
    return {"latest": None, "generations": [], "final": FINAL_SCRIPT_NAME}


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy every training LoRA checkpoint and publish FINAL.sh for the last adapters."
    )
    parser.add_argument("--generation", type=int)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Retrieve every genN directory under the training checkpoint root.",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--pfsp-manifest", type=Path)
    parser.add_argument("--no-pack", action="store_true")
    args = parser.parse_args()
    try:
        family_pack = resolve()
    except FamilyError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc
    checkpoint_root = args.checkpoint_root or family_pack.checkpoint_root
    archive_root = args.archive_root or family_pack.archive_root
    model_config = args.model_config or family_pack.model_config
    pfsp_manifest = args.pfsp_manifest or family_pack.pfsp_manifest
    if args.list:
        print(json.dumps(list_archives(archive_root), indent=2))
        return
    if args.all or args.generation is None:
        payload = archive_all_generations(
            checkpoint_root=checkpoint_root,
            archive_root=archive_root,
            model_config=model_config,
            pfsp_manifest=pfsp_manifest,
            pack=not args.no_pack,
        )
        print(json.dumps(payload, indent=2))
        return
    manifest = archive_generation(
        args.generation,
        checkpoint_root=checkpoint_root,
        archive_root=archive_root,
        model_config=model_config,
        pfsp_manifest=pfsp_manifest,
        pack=not args.no_pack,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    _main()
