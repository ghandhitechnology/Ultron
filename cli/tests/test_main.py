from ultron.cli.main import main


def test_demo_rejects_negative_generation() -> None:
    assert main(["demo", "--generation", "-1"]) == 2


def test_demo_rejects_zero_episodes() -> None:
    assert main(["demo", "--episodes", "0"]) == 2
