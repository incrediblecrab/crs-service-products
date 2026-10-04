"""The CLI's contract with the workflows: exit codes, $GITHUB_OUTPUT keys, Trusted Publishing, each dataset's routing, and workflows that call only commands, options and outputs the CLI has. Also pipeline.yml's inactivity job."""

import json
import os
import re
import subprocess
import time
from pathlib import Path

import httpx
import huggingface_hub
import pytest
import yaml
from huggingface_hub.errors import HfHubHTTPError

from crs_products import cli, constitution, summaries
from crs_products.http import Unavailable
from crs_products.pipeline import Context
from crs_products.store import CARD, LocalStore
from crs_products.store import partition_of as products_partition_of
from conftest import ScriptedSource, local_store, run_once, scripted

WORKFLOWS = Path(__file__).parent.parent / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "pipeline.yml"
# Each workflow, the dataset its commands name, and the commands it runs.
DATASETS = {"pipeline.yml": "products", "summaries.yml": "summaries", "constitution.yml": "constitution"}
COMMANDS = {"pipeline.yml": {"run", "probe", "verify", "squash"}, "summaries.yml": {"run", "verify", "squash"}, "constitution.yml": {"run", "verify", "squash"}}
# What each command writes to $GITHUB_OUTPUT. The probe and run tests check the commands against this, and the workflow test checks the workflow's if: expressions against it.
OUTPUTS = {"probe": {"needed"}, "run": {"commits", "more"}}


@pytest.fixture
def actions(tmp_path, monkeypatch):
    """A GitHub Actions environment with $GITHUB_OUTPUT as a file. HF_OIDC_RESOURCE starts unset and is restored afterwards, since the CLI sets it in os.environ."""
    output = tmp_path / "github_output"
    output.touch()
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("HF_OIDC_RESOURCE", "placeholder")
    monkeypatch.delenv("HF_OIDC_RESOURCE")
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    return output


def outputs(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


def hub_error(status, message):
    return HfHubHTTPError(message, response=httpx.Response(status, request=httpx.Request("POST", "https://huggingface.co/oauth/token")))


def raising(error):
    def fail(*args, **kwargs):
        raise error
    return fail


def test_trusted_publishing_is_requested_only_in_actions_and_only_for_the_hub(actions, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(cli, "cmd_probe", lambda args: seen.append(os.environ.get("HF_OIDC_RESOURCE")) or 0)
    assert cli.main(["probe", "--repo", "someone/some-dataset"]) == 0
    monkeypatch.delenv("HF_OIDC_RESOURCE")
    cli.main(["probe", "--local", str(tmp_path)])
    monkeypatch.delenv("GITHUB_ACTIONS")
    cli.main(["probe", "--repo", "someone/some-dataset"])
    assert seen == ["datasets/someone/some-dataset", None, None]


def test_run_without_a_trusted_publisher_fails_so_github_notifies_the_owner(actions, monkeypatch, capsys):
    monkeypatch.setattr(cli, "open_store", raising(hub_error(400, f"400 Client Error: Bad Request for url: https://huggingface.co/oauth/token ({cli.NO_PUBLISHER} for datasets/x/y)")))
    assert cli.main(["run"]) == 1
    assert outputs(actions) == {"commits": "0", "more": "false"}
    assert capsys.readouterr().out.startswith(f"::error::{cli.NO_PUBLISHER} for {cli.DEFAULT_REPO}, so nothing was written.")


def test_run_with_any_other_hub_refusal_fails(actions, monkeypatch):
    monkeypatch.setattr(cli, "open_store", raising(hub_error(401, "401 Client Error: Unauthorized")))
    with pytest.raises(HfHubHTTPError):
        cli.main(["run"])


@pytest.mark.parametrize("dataset", ["products", "constitution"])
def test_run_refuses_to_start_without_pdftotext(monkeypatch, tmp_path, dataset):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="pdftotext is missing"):
        cli.main(["run", "--dataset", dataset, "--local", str(tmp_path)])


def capture(calls):
    def runner(ctx, source):
        calls.append((ctx, source))
        return {"stopped": None, "finished": True, "commits": 0, "fetched": 0}
    return runner


def test_each_dataset_runs_its_own_source_and_partitions(actions, monkeypatch, tmp_path):
    """Summaries have a sync of their own and need no pdftotext; the Constitution Annotated goes through the products' loop with its own partitions and comparison."""
    calls = []
    monkeypatch.setattr(cli.summaries, "sync", capture(calls))
    monkeypatch.setattr(cli, "sync", capture(calls))
    assert cli.main(["run", "--dataset", "summaries", "--local", str(tmp_path / "s")]) == 0
    assert cli.main(["run", "--dataset", "constitution", "--local", str(tmp_path / "c")]) == 0
    assert cli.main(["run", "--local", str(tmp_path / "p")]) == 0
    (s_ctx, s_source), (c_ctx, c_source), (p_ctx, p_source) = calls
    assert isinstance(s_source, summaries.SummariesSource) and s_ctx.partition_of is summaries.partition_of and s_ctx.comparable is summaries.comparable
    assert isinstance(c_source, constitution.ConanSource) and c_ctx.partition_of is constitution.partition_of and c_ctx.comparable is constitution.comparable
    assert isinstance(p_source, cli.CrsSource) and p_ctx.partition_of is products_partition_of
    assert (s_ctx.source_url, c_ctx.source_url) == (summaries.SOURCE_URL, constitution.SOURCE_URL)
    calls.clear()
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli.main(["run", "--dataset", "summaries", "--local", str(tmp_path / "s")]) == 0 and len(calls) == 1


@pytest.mark.parametrize("dataset, tally, live", [("products", "text_source", None), ("summaries", "bill_type", summaries.live_counts), ("constitution", "kind", None)])
def test_verify_checks_each_dataset_with_its_own_partitions_and_tally(monkeypatch, tmp_path, dataset, tally, live):
    import crs_products.verify

    seen = {}
    monkeypatch.setattr(crs_products.verify, "verify", lambda store, source=None, **kwargs: seen.update(kwargs, source=source) or {"problems": []})
    assert cli.main(["verify", "--dataset", dataset, "--local", str(tmp_path)]) == 0
    expected = {"products": {}, "summaries": {"partition_of": summaries.partition_of, "tally": tally, "live": live}, "constitution": {"partition_of": constitution.partition_of, "tally": tally}}[dataset]
    assert seen == dict(expected, source=None)


@pytest.mark.parametrize("problems, code", [([], 0), (["planted integrity failure"], 1)])
def test_degraded_verification_warns_but_integrity_failures_still_exit_one(actions, monkeypatch, tmp_path, capsys, problems, code):
    import crs_products.verify

    monkeypatch.setattr(crs_products.verify, "verify", lambda *args, **kwargs: {"problems": problems, "warnings": ["source listing incomplete"]})
    assert cli.main(["verify", "--local", str(tmp_path)]) == code
    assert "::warning::source listing incomplete" in capsys.readouterr().out


def test_an_incomplete_sync_warns_without_starting_a_continuation(actions, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "sync", lambda ctx, source: {"finished": False, "stopped": "incomplete listing", "warnings": ["source listing incomplete"], "fetched": 0, "commits": 1})
    assert cli.main(["run", "--local", str(tmp_path)]) == 0
    assert "::warning::source listing incomplete" in capsys.readouterr().out
    assert outputs(actions) == {"commits": "1", "more": "false"}


@pytest.mark.parametrize("dataset, repo", sorted(cli.REPOS.items()))
def test_trusted_publishing_asks_for_the_dataset_s_own_repo(actions, monkeypatch, dataset, repo):
    seen = []
    monkeypatch.setattr(cli, "cmd_squash", lambda args: seen.append((args.repo, os.environ.get("HF_OIDC_RESOURCE"))) or 0)
    assert cli.main(["squash", "--dataset", dataset]) == 0
    assert seen == [(repo, f"datasets/{repo}")] and repo.startswith("incrediblecrab/crs-")


@pytest.mark.parametrize("argv, message", [(["probe", "--dataset", "summaries"], "only the products dataset has a probe"), (["probe", "--dataset", "constitution"], "only the products dataset has a probe"),
                                           (["run", "--dataset", "summaries", "--max-units", "2"], "--max-units does not apply to summaries")])
def test_options_a_dataset_does_not_have_are_refused(argv, message, capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(argv)
    assert stopped.value.code == 2 and message in capsys.readouterr().err


@pytest.mark.parametrize("stopped, code", [(None, 0), ("budget", 0), ("deferred", 0), ("superseded", 0), ("Blocked: bot challenge at www.congress.gov/x.pdf", 1), ("RuntimeError: boom", 1)])
def test_run_exit_codes(actions, monkeypatch, tmp_path, stopped, code):
    monkeypatch.setattr(cli, "sync", lambda ctx, source: {"stopped": stopped, "finished": stopped is None, "commits": 3, "fetched": 1})
    assert cli.main(["run", "--local", str(tmp_path / "hub")]) == code
    assert outputs(actions)["commits"] == "3" and set(outputs(actions)) == OUTPUTS["run"]


# sync() leaves fetched out of a deferred run's record.
@pytest.mark.parametrize("stopped, fetched, more", [("budget", 4, "true"), ("budget", 0, "false"), (None, 4, "false"), ("superseded", 4, "false"), ("RuntimeError: boom", 4, "false"), ("deferred", None, "false")])
def test_only_a_run_that_ran_out_of_budget_while_fetching_asks_for_the_next_run(actions, monkeypatch, tmp_path, stopped, fetched, more):
    run = {"stopped": stopped, "finished": stopped is None, "commits": 1} | ({} if fetched is None else {"fetched": fetched})
    monkeypatch.setattr(cli, "sync", lambda ctx, source: run)
    cli.main(["run", "--local", str(tmp_path / "hub")])
    assert outputs(actions)["more"] == more


def test_more_follows_the_record_the_real_sync_returns(actions, monkeypatch, tmp_path):
    state = scripted(units={f"R40{n:03d}": "2026-09-01T10:00:00Z" for n in range(5)}, stop_after=2)
    monkeypatch.setattr(cli, "CrsSource", lambda fetcher: ScriptedSource(state))
    monkeypatch.setattr(cli, "Context", lambda **kwargs: setattr(state, "ctx", Context(**kwargs)) or state.ctx)
    argv = ["run", "--local", str(tmp_path / "hub"), "--workdir", str(tmp_path)]
    assert cli.main(argv) == 0 and outputs(actions)["more"] == "true"
    state.stop_after = None
    actions.write_text("")
    assert cli.main(argv) == 0 and outputs(actions)["more"] == "false" and len(state.fetched) == 5


class FakeSource:
    """Stands in for CrsSource(fetcher): head() answers the given head or raises the given error."""

    def __init__(self, head=None, error=None):
        self.value, self.error = head, error

    def __call__(self, fetcher):
        return self

    def head(self):
        if self.error:
            raise self.error
        return self.value


def test_probe_asks_for_a_sync_when_the_hub_has_no_manifest(actions, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "CrsSource", FakeSource(head={"count": 5, "newest": "R40001@x"}))
    assert cli.main(["probe", "--local", str(tmp_path)]) == 0
    assert outputs(actions) == {"needed": "true"} and set(outputs(actions)) == OUTPUTS["probe"]
    assert '"reason": "no manifest yet"' in capsys.readouterr().out


def test_probe_decides_from_the_card_without_reading_the_manifest(actions, monkeypatch, tmp_path, capsys):
    state = scripted(units={"R40001": "2026-09-01T10:00:00Z", "IN12001": "2026-09-02T10:00:00Z"})
    run_once(local_store(tmp_path), state, writer="github-actions")
    monkeypatch.setattr(cli, "CrsSource", FakeSource(head=ScriptedSource(state).head()))
    monkeypatch.setattr(LocalStore, "read_manifest", lambda self: pytest.fail("planted: the probe read the manifest"))
    assert cli.main(["probe", "--local", str(tmp_path / "hub")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["needed"], out["reason"], out["state_from"]) == (False, "up to date", "card") and outputs(actions) == {"needed": "false"}


@pytest.mark.parametrize("card", ["---\nlicense: other\n---\n# a card from before the probe state\n", None])
def test_probe_falls_back_to_the_manifest_when_the_card_has_no_state(actions, monkeypatch, tmp_path, capsys, card):
    state = scripted(units={"R40001": "2026-09-01T10:00:00Z"})
    run_once(local_store(tmp_path), state, writer="github-actions")
    readme = tmp_path / "hub" / CARD
    readme.write_text(card) if card else readme.unlink()
    monkeypatch.setattr(cli, "CrsSource", FakeSource(head=dict(ScriptedSource(state).head(), count=2)))
    assert cli.main(["probe", "--local", str(tmp_path / "hub")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["needed"], out["reason"], out["state_from"]) == (True, "count 1 -> 2", "manifest")


@pytest.mark.parametrize("error", [Unavailable("HTTP 503 from api.congress.gov/v3/crsreport"), hub_error(503, "503 Server Error"), hub_error(429, "429 Too Many Requests"), httpx.ConnectError("refused")])
def test_a_transient_outage_skips_the_probe_quietly(actions, monkeypatch, tmp_path, error):
    monkeypatch.setattr(cli, "CrsSource", FakeSource(error=error))
    assert cli.main(["probe", "--local", str(tmp_path)]) == 0
    assert outputs(actions) == {"needed": "false"}


@pytest.mark.parametrize("error", [RuntimeError("a bug"), hub_error(404, "404 Client Error: Repository Not Found")])
def test_any_other_probe_error_fails_the_job(actions, monkeypatch, tmp_path, error):
    monkeypatch.setattr(cli, "CrsSource", FakeSource(error=error))
    with pytest.raises(type(error)):
        cli.main(["probe", "--local", str(tmp_path)])
    assert outputs(actions) == {}


@pytest.mark.parametrize("commits, squashed", [(cli.SQUASH_AFTER_COMMITS, False), (cli.SQUASH_AFTER_COMMITS + 1, True)])
def test_squash_only_past_the_threshold(monkeypatch, commits, squashed):
    calls = []

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def list_repo_commits(self, repo_id, repo_type):
            return [None] * commits

        def super_squash_history(self, repo_id, repo_type, commit_message):
            calls.append((repo_id, commit_message))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    assert cli.main(["squash", "--repo", "x/y"]) == 0
    assert bool(calls) == squashed


def shell_argv(line, env):
    """The argv bash builds for one `python -m crs_products ...` line of a workflow step, with the step's env."""
    command = line.split("|")[0].strip().replace("python -m crs_products", "printf '%s\\0'", 1)
    done = subprocess.run(["bash", "-c", command], capture_output=True, text=True, env={"PATH": os.environ["PATH"], **env}, check=True)
    return done.stdout.split("\0")[:-1]


def test_every_workflow_is_checked_here():
    assert sorted(path.name for path in WORKFLOWS.iterdir() if path.suffix in (".yml", ".yaml")) == sorted(DATASETS)


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_the_workflow_calls_only_commands_options_and_outputs_the_cli_has(monkeypatch, name):
    path = WORKFLOWS / name
    workflow = yaml.safe_load(path.read_text())
    triggers = workflow.get("on", workflow.get(True))  # PyYAML reads the key `on` as True
    jobs = workflow["jobs"]
    dataset = DATASETS[name]

    example = re.search(r"e\.g\. (.+?) \(", triggers["workflow_dispatch"]["inputs"]["args"]["description"]).group(1)
    parsed = []
    for command in ("run", "probe", "verify", "squash"):
        monkeypatch.setattr(cli, f"cmd_{command}", lambda args: parsed.append(args) or 0)
    command_of = {}
    for step in jobs["sync"]["steps"]:
        for line in (step.get("run") or "").splitlines():
            if "python -m crs_products" not in line:
                continue
            # The products' workflow predates --dataset and relies on its default; every other workflow names its dataset, so no command can fall back to the products.
            assert dataset == "products" or f"--dataset {dataset} " in line + " ", f"{step.get('name')}: {line}"
            envs = [{}]
            if "$EXTRA_ARGS" in line or "$BUDGET" in line:
                envs = [{}, {"BUDGET": "30", "EXTRA_ARGS": example}]
            for env in envs:
                argv = shell_argv(line, env)
                assert cli.main(argv) == 0, f"{step.get('name')}: {argv}"
                command_of[step.get("id")] = argv[0]
    assert {args.command for args in parsed} == COMMANDS[name]
    assert {(args.dataset, args.repo) for args in parsed if not args.local} == {(dataset, cli.REPOS[dataset])}
    smoke = next(args for args in parsed if args.command == "run" and args.local)
    assert smoke.budget_minutes == 30 and smoke.partitions and (smoke.max_units or dataset == "summaries")
    assert all(args.budget_minutes < jobs["sync"]["timeout-minutes"] for args in parsed if args.command == "run"), "the budget must end the run before GitHub does"

    expressions = " ".join([str(step.get("if", "")) for step in jobs["sync"]["steps"]] + list(jobs["sync"].get("outputs", {}).values()))
    referenced = re.findall(r"steps\.(\w+)\.outputs\.(\w+)", expressions)
    assert referenced
    for step_id, key in referenced:
        assert key in OUTPUTS[command_of[step_id]], f"steps.{step_id}.outputs.{key}: `{command_of[step_id]}` does not write {key}"

    needed = [(job_name, key) for job in jobs.values() for job_name, key in re.findall(r"needs\.(\w+)\.outputs\.(\w+)", str(job.get("if", "")))]
    for job_name, key in needed:
        assert key in jobs[job_name].get("outputs", {}), f"needs.{job_name}.outputs.{key}: job {job_name} declares no output {key}"
    # A job that starts runs starts this workflow, never after a bounded test, and holds no permission to touch the dataset; the job that parses downloads cannot start runs.
    starters = [job for job in jobs.values() if any("gh workflow run" in (step.get("run") or "") for step in job["steps"])]
    assert bool(starters) == bool(needed) == (name != "constitution.yml")
    for job in starters:
        assert all(f"gh workflow run {name} " in step["run"] for step in job["steps"] if "gh workflow run" in (step.get("run") or ""))
        assert "!inputs.args" in job["if"] and job["permissions"] == {"actions": "write"}
    assert "actions" not in jobs["sync"]["permissions"]
    # No job may commit: GitHub Trust & Safety called commits that keep a schedule enabled a violation of its Terms.
    assert workflow["permissions"] == {} and workflow["concurrency"]["group"] == path.stem
    for job_name, job in jobs.items():
        assert job.get("permissions", {}).get("contents") != "write", f"job {job_name} can push"
        assert not any(re.search(r"\bgit\s+(commit|push)\b", step.get("run") or "") for step in job["steps"]), f"job {job_name} commits"


def crons(name):
    workflow = yaml.safe_load((WORKFLOWS / name).read_text())
    return [entry["cron"] for entry in workflow.get("on", workflow.get(True))["schedule"]]


def test_every_schedule_is_what_the_code_and_cards_say():
    # The owner asked on September 26, 2026 for every dataset to update at 00:00 and 12:00 UTC.
    for name in DATASETS:
        assert crons(name) == ["0 0,12 * * *"], name
    assert summaries.SCHEDULE_HOURS == 12
    # The daily check keeps to one slot: the run a day later is due, the one before it is not.
    assert 24 - summaries.SCHEDULE_HOURS < summaries.RECONCILE_HOURS < 24
    for dataset in DATASETS.values():
        text = cli.card_of(dataset)({})
        assert "scheduled at 00:00 and 12:00 UTC" in text or f"every {summaries.SCHEDULE_HOURS} hours, at 00:00 and 12:00 UTC" in text, dataset
        assert not re.search(r"every 5 minutes|every 6 hours|once a week|weekly", text), dataset


@pytest.mark.parametrize("idle_days, fails", [(49, False), (50, True)])
def test_inactivity_runs_on_every_schedule_and_fails_from_50_idle_days_without_committing(tmp_path, idle_days, fails):
    """The step run the way GitHub runs a step (bash -e), in a checkout made the way actions/checkout makes one, against a local origin. That no job commits is checked for every workflow above."""
    workflow = yaml.safe_load(WORKFLOW.read_text())
    job = workflow["jobs"]["inactivity"]
    assert job["if"] == "github.event_name == 'schedule'"
    (step,) = [step for step in job["steps"] if "run" in step]
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}

    def git(*args, cwd, **extra):
        return subprocess.run(["git", *args], cwd=cwd, env={**env, **extra}, capture_output=True, text=True, check=True).stdout

    origin, seed, work = tmp_path / "origin.git", tmp_path / "seed", tmp_path / "work"
    for path in (origin, seed, work):
        path.mkdir()
    git("init", "-q", "--bare", "-b", "main", cwd=origin)
    git("init", "-q", "-b", "main", cwd=seed)
    stamp = f"@{int(time.time()) - idle_days * 86400} +0000"
    who = {"GIT_AUTHOR_NAME": "a", "GIT_AUTHOR_EMAIL": "a@example.com", "GIT_COMMITTER_NAME": "a", "GIT_COMMITTER_EMAIL": "a@example.com"}
    for title in ("first change", "last change"):
        git("commit", "-q", "--allow-empty", "-m", title, cwd=seed, GIT_AUTHOR_DATE=stamp, GIT_COMMITTER_DATE=stamp, **who)
    git("push", "-q", origin.as_uri(), "main", cwd=seed)
    # actions/checkout on a scheduled run: a shallow fetch, then a local main that tracks origin's.
    git("init", "-q", cwd=work)
    git("remote", "add", "origin", origin.as_uri(), cwd=work)
    git("fetch", "-q", "--depth=1", "origin", "+refs/heads/main:refs/remotes/origin/main", cwd=work)
    git("checkout", "-q", "--force", "-B", "main", "refs/remotes/origin/main", cwd=work)
    assert git("rev-parse", "--is-shallow-repository", cwd=work).strip() == "true"

    done = subprocess.run(["bash", "-e", "-c", step["run"]], cwd=work, env={**env, "SCHEDULE": "0 0,12 * * *"}, capture_output=True, text=True)
    assert done.returncode == (1 if fails else 0), done.stdout + done.stderr
    assert "schedule: 0 0,12 * * *" in done.stdout
    assert f"last commit {idle_days} days ago" in done.stdout
    errors = [line for line in done.stdout.splitlines() if line.startswith("::error::")]
    assert bool(errors) == fails
    assert all(f"No commit in {idle_days} days" in line and "60 days" in line for line in errors)
    assert git("log", "--format=%s", "main", cwd=origin).splitlines() == ["last change", "first change"]
