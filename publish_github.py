from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{result.stderr.strip()}")


def run_capture(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/push GitHub repository for this project")
    parser.add_argument("--repo", default="Rengmitca-IPA", help="Repository name (GitHub slug)")
    parser.add_argument("--visibility", choices=["public", "private"], default="public")
    parser.add_argument("--branch", default="main")
    args = parser.parse_args()

    gh_token = os.getenv("GH_TOKEN")
    if not gh_token:
        print("GH_TOKEN is required.", file=sys.stderr)
        sys.exit(1)

    repo_dir = Path(".").resolve()
    if not (repo_dir / ".git").exists():
        run(["git", "init"], cwd=repo_dir)
        run(["git", "branch", "-M", args.branch], cwd=repo_dir)

    # If GH_TOKEN is present, gh uses it automatically; explicit login can fail by design.
    auth_status = subprocess.run(
        ["gh", "auth", "status"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    if auth_status.returncode != 0 and "GH_TOKEN" not in (auth_status.stdout + auth_status.stderr):
        login = subprocess.run(
            ["gh", "auth", "login", "--with-token"],
            input=gh_token,
            text=True,
            capture_output=True,
            cwd=str(repo_dir),
        )
        if login.returncode != 0 and "already logged in" not in login.stderr.lower():
            raise RuntimeError(f"gh auth login failed:\n{login.stderr.strip()}")

    run(["git", "add", "-A"], cwd=repo_dir)
    commit = run_capture(
        ["git", "commit", "-m", "Initial Rengmitca IPA pipeline\n\nCo-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"],
        cwd=repo_dir,
    )
    if commit.returncode != 0:
        commit_text = (commit.stdout + "\n" + commit.stderr).lower()
        if "nothing to commit" not in commit_text:
            raise RuntimeError(f"git commit failed:\n{commit.stderr.strip()}")

    repo_slug = args.repo.replace(" ", "-")
    visibility_flag = "--public" if args.visibility == "public" else "--private"
    remote = run_capture(["git", "remote", "get-url", "origin"], cwd=repo_dir)
    if remote.returncode != 0:
        create = run_capture(
            ["gh", "repo", "create", repo_slug, visibility_flag, "--source", ".", "--remote", "origin"],
            cwd=repo_dir,
        )
        if create.returncode != 0 and "already exists" not in (create.stdout + create.stderr).lower():
            raise RuntimeError(f"gh repo create failed:\n{create.stderr.strip()}")

        # If repo already existed and origin was not set, add it manually.
        remote = run_capture(["git", "remote", "get-url", "origin"], cwd=repo_dir)
        if remote.returncode != 0:
            owner = run_capture(["gh", "api", "user", "-q", ".login"], cwd=repo_dir)
            if owner.returncode != 0:
                raise RuntimeError(f"Could not determine GitHub username:\n{owner.stderr.strip()}")
            repo_url = f"https://github.com/{owner.stdout.strip()}/{repo_slug}.git"
            run(["git", "remote", "add", "origin", repo_url], cwd=repo_dir)

    run(["git", "push", "-u", "origin", args.branch], cwd=repo_dir)
    print(f"Pushed to GitHub repo: {repo_slug}")


if __name__ == "__main__":
    main()
