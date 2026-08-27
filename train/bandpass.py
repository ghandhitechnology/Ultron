from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_profiles(
    win_rates: dict[str, float],
    *,
    generation: int,
    minimum_profiles: tuple[str, ...] = ("workstation", "web"),
    lower: float = 0.30,
    upper: float = 0.70,
) -> list[str]:
    if generation == 0:
        return list(dict.fromkeys((*minimum_profiles, *win_rates.keys())))
    selected = [name for name, rate in win_rates.items() if lower <= rate <= upper]
    for profile in minimum_profiles:
        if profile in win_rates and len(selected) < len(minimum_profiles) and profile not in selected:
            selected.append(profile)
    if len(selected) < len(minimum_profiles):
        nearest = sorted(win_rates, key=lambda name: abs(win_rates[name] - 0.5))
        for profile in nearest:
            if profile not in selected:
                selected.append(profile)
            if len(selected) >= min(len(minimum_profiles), len(win_rates)):
                break
    return selected


def kill_switch_reason(asr: float, generation: int) -> str | None:
    if generation > 0 and asr in (0.0, 1.0):
        return f"ASR stuck at {asr:.1f} after generation 0"
    return None


def _main() -> None:
    parser = argparse.ArgumentParser(description="Apply Ultron curriculum gates.")
    parser.add_argument("--check-kill-switch", action="store_true")
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    if not args.check_kill_switch:
        parser.error("--check-kill-switch is required")
    if args.metrics is None or not args.metrics.exists():
        print("No metrics file supplied; kill-switch check skipped.")
        return
    asr = float(json.loads(args.metrics.read_text())["asr"])
    reason = kill_switch_reason(asr, args.generation)
    if reason:
        raise SystemExit(reason)


if __name__ == "__main__":
    _main()
