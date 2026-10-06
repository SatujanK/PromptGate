"""Tests for the promptgate command, with fake clients and a temporary log."""

import json

import pytest

from promptgate.cli import main
from promptgate.config import load_config
from promptgate.log import read_records
from tests.fakes import FakeClaude, FakeJev, ScriptedUI, make_clients
from tests.fixtures import SAMPLES

QUICK = SAMPLES[0]


@pytest.fixture
def config(tmp_path):
    config = load_config()
    config.paths.log_file = tmp_path / "logs" / "runs.jsonl"
    return config


def run_cli(config, argv, jev=None, claude=None, ui_answers=()):
    clients = make_clients(jev=jev or FakeJev(QUICK.verdict), claude=claude)
    return main(argv, config=config, clients=clients, ui=ScriptedUI(list(ui_answers))), clients


# ---------------------------------------------------------------------------
# Running a prompt
# ---------------------------------------------------------------------------


def test_prompt_text_runs_and_exits_zero(config):
    code, clients = run_cli(config, [QUICK.prompt])
    assert code == 0
    assert len(clients.claude.execute_calls) == 1
    assert len(read_records(config.paths.log_file)) == 1


def test_dry_run_flag_never_calls_claude(config):
    code, clients = run_cli(config, ["--dry-run", QUICK.prompt])
    assert code == 0
    assert clients.claude.plan_calls == []
    assert clients.claude.execute_calls == []
    assert read_records(config.paths.log_file)[0].dry_run is True


def test_prompt_from_a_file_is_read_word_for_word(config, tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("What does\ndf.shape return?\n", encoding="utf-8")
    code, clients = run_cli(config, ["--file", str(prompt_file)])
    assert code == 0
    assert clients.jev.calls == ["What does\ndf.shape return?\n"]


def test_input_files_are_attached_and_hidden_from_jev(config, tmp_path):
    first = tmp_path / "article.txt"
    second = tmp_path / "notes.md"
    first.write_text("Article body", encoding="utf-8")
    second.write_text("My notes", encoding="utf-8")
    code, clients = run_cli(config, ["--input", str(first), "--input", str(second), QUICK.prompt])
    assert code == 0
    assert clients.jev.calls == [QUICK.prompt]
    rendered = clients.claude.execute_calls[0][0]
    assert "<source>article.txt</source>" in rendered.user
    assert "<source>notes.md</source>" in rendered.user
    assert read_records(config.paths.log_file)[0].input_sources == ["article.txt", "notes.md"]


@pytest.mark.parametrize(
    "argv",
    [[], ["--file", "prompt.txt", "and text"], ["   "]],
    ids=["nothing", "both", "blank"],
)
def test_bad_prompt_arguments_exit_with_code_2(config, argv, tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        run_cli(config, argv)
    assert exit_info.value.code == 2


def test_missing_input_file_exits_with_code_2(config, tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        run_cli(config, ["--input", str(tmp_path / "nope.txt"), QUICK.prompt])
    assert exit_info.value.code == 2


def test_failed_execution_exits_with_code_1(config):
    code, _ = run_cli(config, [QUICK.prompt], claude=FakeClaude(fail_execute=True))
    assert code == 1


def test_ctrl_c_exits_with_code_130(config):
    class InterruptingUI(ScriptedUI):
        def stream(self, text):
            raise KeyboardInterrupt

    clients = make_clients(jev=FakeJev(QUICK.verdict))
    assert main([QUICK.prompt], config=config, clients=clients, ui=InterruptingUI()) == 130


def test_missing_api_keys_stop_with_a_clear_message(config, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="Set ANTHROPIC_API_KEY and TYPESAFE_API_KEY"):
        main([QUICK.prompt], config=config, ui=ScriptedUI())
    assert not config.paths.log_file.exists()  # nothing ran, so nothing was logged


def test_one_missing_key_is_named(config, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="Set TYPESAFE_API_KEY in"):
        main([QUICK.prompt], config=config, ui=ScriptedUI())


# ---------------------------------------------------------------------------
# promptgate mark <run_id> --corrected
# ---------------------------------------------------------------------------


def test_mark_sets_corrected_on_a_real_run(config, capsys):
    run_cli(config, [QUICK.prompt])
    run_id = read_records(config.paths.log_file)[0].run_id

    assert main(["mark", run_id, "--corrected"], config=config) == 0
    assert read_records(config.paths.log_file)[0].corrected is True
    assert f"Marked {run_id} as corrected." in capsys.readouterr().out


def test_mark_unknown_run_exits_with_code_1(config, capsys):
    run_cli(config, [QUICK.prompt])
    assert main(["mark", "nope", "--corrected"], config=config) == 1
    assert "no run with id 'nope'" in capsys.readouterr().err


def test_mark_without_corrected_flag_exits_with_code_2(config):
    with pytest.raises(SystemExit) as exit_info:
        main(["mark", "abc"], config=config)
    assert exit_info.value.code == 2


def test_a_prompt_that_starts_with_the_word_mark_is_still_a_prompt(config):
    code, clients = run_cli(config, ["mark my essay out of 10"])
    assert code == 0
    assert clients.jev.calls == ["mark my essay out of 10"]
    assert json.loads(config.paths.log_file.read_text(encoding="utf-8"))["prompt"] == "mark my essay out of 10"
