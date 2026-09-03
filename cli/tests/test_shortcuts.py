from ultron.cli.shortcuts import (
    ConsoleFocus,
    ConsoleScreen,
    GymFocus,
    Shortcut,
    console_shortcuts,
    gym_shortcuts,
)


def test_catalog_lists_run_and_navigation() -> None:
    keys = [item.key for item in console_shortcuts(ConsoleScreen.CATALOG, ConsoleFocus.CATALOG)]
    assert keys == ["enter", "m", "j", "r", "t", "q"]
    assert all(isinstance(item, Shortcut) for item in console_shortcuts(ConsoleScreen.CATALOG, ConsoleFocus.CATALOG))


def test_form_focus_hides_letter_bindings_and_orders_by_input() -> None:
    empty = console_shortcuts(ConsoleScreen.CATALOG, ConsoleFocus.FORM, typing=True, filled=False)
    assert [item.key for item in empty] == ["tab", "enter", "esc", "click"]
    assert [item.action for item in empty] == ["focus_next", "confirm", "show_catalog", "quit"]
    filled = console_shortcuts(ConsoleScreen.CATALOG, ConsoleFocus.FORM, typing=True, filled=True)
    assert [item.key for item in filled] == ["enter", "tab", "esc", "click"]
    assert "j" not in {item.key for item in filled}
    assert "q" not in {item.key for item in filled}


def test_family_focus_offers_pick_and_return() -> None:
    keys = [item.key for item in console_shortcuts(ConsoleScreen.CATALOG, ConsoleFocus.FAMILY)]
    assert keys == ["enter", "esc", "a", "q"]


def test_jobs_shortcuts_depend_on_selection_and_focus() -> None:
    empty = console_shortcuts(ConsoleScreen.JOBS, ConsoleFocus.JOBS, has_job=False)
    assert [item.action for item in empty] == ["refresh", "back", "quit"]
    selected = console_shortcuts(ConsoleScreen.JOBS, ConsoleFocus.JOBS, has_job=True)
    assert [item.action for item in selected] == ["confirm", "stop_job", "refresh", "back", "quit"]
    log = console_shortcuts(ConsoleScreen.JOBS, ConsoleFocus.JOB_LOG, has_job=True)
    assert [item.action for item in log] == ["back", "refresh", "stop_job", "quit"]


def test_results_and_run_shortcuts_follow_position() -> None:
    none = console_shortcuts(ConsoleScreen.RESULTS, ConsoleFocus.RESULTS, has_generation=False)
    assert [item.action for item in none] == ["refresh", "back", "quit"]
    rows = console_shortcuts(ConsoleScreen.RESULTS, ConsoleFocus.RESULTS, has_generation=True)
    assert rows[0] == Shortcut("enter", "review", "confirm")
    review = console_shortcuts(ConsoleScreen.RESULTS, ConsoleFocus.REVIEW, has_generation=True)
    assert [item.action for item in review] == ["back", "refresh", "quit"]
    idle = console_shortcuts(ConsoleScreen.RUN, ConsoleFocus.RUN, can_stop=False)
    assert [item.action for item in idle] == ["back", "quit"]
    live = console_shortcuts(ConsoleScreen.RUN, ConsoleFocus.RUN, can_stop=True)
    assert [item.action for item in live] == ["stop_job", "back", "quit"]


def test_gym_shortcuts_follow_expanded_pane_and_focus() -> None:
    arena = gym_shortcuts(expanded=None, focus=GymFocus.ARENA)
    assert [item.key for item in arena] == ["a", "s", "d", "t", "q"]
    detail = gym_shortcuts(expanded="sandbox", focus=GymFocus.DETAIL)
    assert detail[0] == Shortcut("esc", "fold", "collapse")
    assert "expand:sandbox" in {item.action for item in detail}
    log = gym_shortcuts(expanded="tool", focus=GymFocus.LOG)
    assert [item.action for item in log] == ["expand:tool", "collapse", "quit"]
    log_arena = gym_shortcuts(expanded=None, focus=GymFocus.LOG)
    assert [item.action for item in log_arena] == ["expand:tool", "expand:attacker", "quit"]
