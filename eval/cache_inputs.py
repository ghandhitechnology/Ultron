"""Resolve the local inputs actually selected by an evaluation command.

Cache declarations use ``local_inputs: {harness_root, dataset_root}``. Both roots
must be local, read-only inputs. The dataset must be the exact directory selected
by the CLI. A Python console entrypoint outside the harness root must resolve,
through its own interpreter, to installed harness source inside that root.
Unresolved inputs disable reuse; they do not prevent evaluation execution.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any, Mapping

_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_MAX_FILES = 20_000
_MAX_BYTES = 256 * 1024 * 1024


class UnresolvedInputs(ValueError):
    pass


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _tree_hash(root: Path) -> str:
    if not root.is_dir():
        raise UnresolvedInputs("input root is not a directory")
    digest = hashlib.sha256()
    size = count = 0
    for directory, folders, files in os.walk(root, followlinks=False):
        folders[:] = sorted(name for name in folders if name not in _SKIP_DIRS)
        if any((Path(directory) / name).is_symlink() for name in folders):
            raise UnresolvedInputs("input tree contains an unresolved directory symlink")
        for name in sorted(files):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = Path(directory) / name
            if not _inside(path.resolve(), root):
                raise UnresolvedInputs("input file points outside its declared root")
            before = path.stat()
            count += 1
            size += before.st_size
            if count > _MAX_FILES or size > _MAX_BYTES:
                raise UnresolvedInputs("input tree exceeds bounded cache fingerprint size")
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise UnresolvedInputs("input changed while being fingerprinted")
            digest.update(b"\0")
    if not count:
        raise UnresolvedInputs("input tree is empty")
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    if path.stat().st_size > _MAX_BYTES:
        raise UnresolvedInputs("executable is too large for bounded cache fingerprinting")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str, cwd: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else cwd / path).resolve()


def _dataset_argument(argv: tuple[str, ...]) -> str:
    selected = []
    flags = {"-p", "--path", "-d", "--dataset"}
    for index, value in enumerate(argv):
        if value in flags:
            if index + 1 >= len(argv):
                raise UnresolvedInputs("dataset flag has no argument")
            selected.append(argv[index + 1])
        elif value.startswith(("--path=", "--dataset=")):
            selected.append(value.split("=", 1)[1])
    if len(selected) != 1:
        raise UnresolvedInputs("command does not select one known local dataset input")
    return selected[0]


def _which(value: str, cwd: Path, env: Mapping[str, str]) -> Path:
    if "/" in value:
        given = Path(value).expanduser()
        path = (given if given.is_absolute() else cwd / given).absolute()
    else:
        resolved = shutil.which(value, path=env.get("PATH", os.defpath))
        if not resolved:
            raise UnresolvedInputs("evaluation executable is unavailable")
        path = Path(resolved).absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise UnresolvedInputs("evaluation executable is unavailable")
    # Preserve a virtualenv Python symlink for interpreter invocation.
    return path


_INSPECT = r'''
import importlib.metadata, importlib.util, json, sys
sys.path[0] = sys.argv[1]
requested = json.loads(sys.argv[3])
modules = {}
for name in requested:
    if name in sys.stdlib_module_names:
        continue
    spec = importlib.util.find_spec(name)
    modules[name] = None if spec is None else {
        "origin": spec.origin,
        "locations": list(spec.submodule_search_locations or []),
    }
entries = []
for entry in importlib.metadata.entry_points(group="console_scripts"):
    if entry.name == sys.argv[2]:
        entries.append({"value": entry.value, "distribution": entry.dist.metadata["Name"],
                        "version": entry.dist.version})
print(json.dumps({"modules": modules, "entries": entries, "python_version": sys.version,
                  "python_prefix": sys.prefix, "python_flags": list(sys.flags)}))
'''


def _python_source(executable: Path, root: Path, cwd: Path, env: Mapping[str, str]) -> dict | None:
    with executable.open("rb") as stream:
        head = stream.read(2)
    if head != b"#!":
        return None
    if executable.stat().st_size > 1024 * 1024:
        raise UnresolvedInputs("entrypoint is not a bounded Python console script")
    source = executable.read_text()
    parts = shlex.split(source.splitlines()[0][2:])
    if not parts:
        raise UnresolvedInputs("empty executable shebang")
    if Path(parts[0]).name == "env":
        parts = parts[1:]
        if parts and parts[0] == "-S":
            parts = parts[1:]
    if not parts or "python" not in Path(parts[0]).name.lower():
        raise UnresolvedInputs("wrapper executable does not resolve to a Python harness")
    flags = parts[1:]
    if any(flag not in {"-s", "-E", "-I", "-O", "-OO"} for flag in flags):
        raise UnresolvedInputs("unsupported Python entrypoint flags")
    interpreter = _which(parts[0], cwd, env)
    tree = ast.parse(source)
    imports = set()
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported_modules.add(node.module)
            imports.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval", "__import__"}:
            raise UnresolvedInputs("dynamic entrypoint cannot establish harness source linkage")
    result = subprocess.run([str(interpreter), *flags, "-c", _INSPECT, str(executable.resolve().parent), executable.name,
                             json.dumps(sorted(imports))], cwd=str(cwd) if cwd.is_dir() else None,
                            env=dict(env), capture_output=True, text=True, check=True, timeout=10)
    observed = json.loads(result.stdout)
    for module in observed["modules"].values():
        if module is None:
            raise UnresolvedInputs("entrypoint import cannot be resolved")
        paths = [*module["locations"]]
        if module["origin"] not in {None, "built-in", "frozen"}:
            paths.append(module["origin"])
        if not paths or any(not _inside(Path(path).resolve(), root) for path in paths):
            raise UnresolvedInputs("entrypoint imports source outside the declared harness root")
    if not _inside(executable.resolve(), root):
        matching = [entry for entry in observed["entries"] if entry["value"].split(":", 1)[0] in imported_modules]
        if len(matching) != 1 or not observed["modules"]:
            raise UnresolvedInputs("installed console entrypoint is not linked to declared harness source")
        observed["entries"] = matching
    else:
        observed["entries"] = []
    observed.update(interpreter=str(interpreter), interpreter_sha256=_file_hash(interpreter.resolve()))
    return observed


def actual_cache_inputs(job: Any) -> dict[str, Any] | None:
    """Return observed dataset/harness fingerprints, or disable reuse on ambiguity."""
    try:
        declared = dict(job.cache_context).get("local_inputs")
        if not isinstance(declared, Mapping):
            return None
        cwd = Path(job.cwd).resolve() if job.cwd else Path.cwd()
        harness = _resolve_path(str(declared["harness_root"]), cwd)
        dataset = _resolve_path(str(declared["dataset_root"]), cwd)
        argument = _dataset_argument(tuple(job.argv))
        if "@" in argument and not Path(argument).is_absolute() and not argument.startswith("."):
            return None
        actual_dataset = _resolve_path(argument, cwd)
        # Registry references, including @latest and unverifiable remote hashes,
        # never become local data merely because a revision was declared beside them.
        if actual_dataset != dataset or not actual_dataset.is_dir():
            return None
        output = Path(job.workdir).resolve()
        if any(_inside(output, root) or _inside(root, output) for root in (harness, dataset)):
            return None
        env = {**os.environ, **dict(job.env)}
        executable = _which(job.argv[0], cwd, env)
        python = _python_source(executable, harness, cwd, env)
        if python is None:
            if not _inside(executable.resolve(), harness):
                return None
            with executable.open("rb") as stream:
                magic = stream.read(4)
            if magic not in {b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe"}:
                return None
        return {"version": 1, "dataset_root": str(dataset), "dataset_sha256": _tree_hash(dataset),
                "harness_root": str(harness), "harness_sha256": _tree_hash(harness),
                "executable": str(executable), "executable_sha256": _file_hash(executable.resolve()),
                "python": python}
    except (KeyError, IndexError, OSError, ValueError, SyntaxError, subprocess.SubprocessError):
        return None
