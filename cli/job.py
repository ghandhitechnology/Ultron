from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from ultron.train.episode_runner import EpisodeConfig, EpisodeRunner, GuestVm

from .ascii import render
from .board import Board
from .demo import default_spec, play_script, scripted_guest, scripted_seams
from .fold import BoardFold
from .spec import JobSpec, Seams


def instrument(seams: Seams, fold: BoardFold) -> Seams:
    def restore(vm: GuestVm, sha256: str) -> None:
        fold.apply_restore_start(vm, sha256)
        try:
            seams.restore(vm, sha256)
        except BaseException as exc:
            fold.apply_fail("restore", exc)
            raise
        fold.apply_restore_done(vm)

    def run_turn(vm: GuestVm, role, profile, turn):
        fold.apply_turn_start(role, turn)
        try:
            steps = seams.run_turn(vm, role, profile, turn)
        except BaseException as exc:
            fold.apply_fail("turn", exc)
            raise
        fold.apply_turn_done(role, turn, steps)
        return steps

    def final_probe(vm: GuestVm, profile):
        fold.apply_probe_start()
        try:
            probe = seams.final_probe(vm, profile)
        except BaseException as exc:
            fold.apply_fail("probe", exc)
            raise
        fold.apply_probe_done(probe)
        return probe

    return Seams(restore, run_turn, final_probe, seams.load_profile)


@dataclass(frozen=True)
class LiveJob:
    spec: JobSpec
    seams: Seams
    guests: Iterable[GuestVm]
    config_for: Callable[[GuestVm], EpisodeConfig]
    quarantine_of: Callable[[], Mapping[str, str]] | None = None
    script: str | None = None

    def board_at_end(self) -> Board:
        fold = BoardFold(self.spec, clock=time.monotonic)
        if self.script is not None:
            play_script(fold, self.script, scripted_guest(), self.spec)
            return fold.snapshot()
        wrapped = instrument(self.seams, fold)
        runner = EpisodeRunner(
            snapshot_sha256=self.spec.snapshot_sha256,
            load_profile=wrapped.load_profile,
            run_turn=wrapped.run_turn,
            final_probe=wrapped.final_probe,
            restore=wrapped.restore,
            turns_per_side=self.spec.turns_per_side,
        )
        count = 0
        for guest in self.guests:
            if self.quarantine_of is not None:
                fold.poll_quarantine(self.quarantine_of())
            runner.run(self.config_for(guest), guest)
            fold.apply_episode_done()
            count += 1
            if count >= self.spec.episode_count:
                break
        fold.apply_job_done()
        return fold.snapshot()

    def ascii(self) -> str:
        return render(self.board_at_end())

    def run(self) -> Board:
        try:
            from ultron.cli.tui.app import run_theater
        except ImportError as exc:
            raise SystemExit(
                "install the TUI extra: pip install 'ultron[tui]'"
            ) from exc
        fold = BoardFold(self.spec, clock=time.monotonic)
        if self.script is not None:
            play_script(fold, self.script, scripted_guest(), self.spec)
            return run_theater(fold)
        wrapped = instrument(self.seams, fold)
        runner = EpisodeRunner(
            snapshot_sha256=self.spec.snapshot_sha256,
            load_profile=wrapped.load_profile,
            run_turn=wrapped.run_turn,
            final_probe=wrapped.final_probe,
            restore=wrapped.restore,
            turns_per_side=self.spec.turns_per_side,
        )

        def work() -> None:
            count = 0
            try:
                for guest in self.guests:
                    if self.quarantine_of is not None:
                        fold.poll_quarantine(self.quarantine_of())
                    runner.run(self.config_for(guest), guest)
                    fold.apply_episode_done()
                    count += 1
                    if count >= self.spec.episode_count:
                        break
                fold.apply_job_done()
            except BaseException as exc:
                fold.apply_fail("loop", exc)

        return run_theater(fold, worker=work)

    @classmethod
    def demo(cls, script: str = "full-16") -> LiveJob:
        spec = default_spec()
        guest = scripted_guest()
        seams = scripted_seams(script)
        return cls(
            spec=spec,
            seams=seams,
            guests=(guest,),
            config_for=lambda vm: EpisodeConfig(
                profile_id=spec.profile_id,
                generation=spec.generation,
                group_id=spec.group_id,
                opponent_checkpoint_id=spec.opponent_checkpoint_id,
                attacker_ckpt="unused",
                defender_ckpt="unused",
            ),
            script=script,
        )
