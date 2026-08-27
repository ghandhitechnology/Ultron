from typing import Protocol


class UserExecutor(Protocol):
    def exec_as_user(self, username: str, cmd: str) -> tuple[str, int]: ...


def host_confirm_root(guest_agent: UserExecutor, attacker_username: str) -> bool:
    stdout, exit_code = guest_agent.exec_as_user(attacker_username, "id -u")
    return exit_code == 0 and stdout.strip() == "0"
