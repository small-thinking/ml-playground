---
name: branch-first-uv-workflow
description: Enforce a safe git-and-environment workflow before making code changes. Use when Codex is asked to implement, edit, refactor, fix, review-follow-up, or otherwise modify a codebase that should be isolated on a feature branch, based on the latest remote default branch, and executed in the repo's uv-managed Python environment when one exists.
---

# Branch First Uv Workflow

Inspect git state before editing. Keep code changes off the default branch, base work on the latest remote default branch, and prefer the repo's existing `uv` workflow for Python commands when `pyproject.toml` or `uv.lock` is present.

Follow this workflow before substantial code edits unless the user explicitly asks for a different git flow.

## Workflow

1. Inspect the repository before editing.
2. Identify the remote default branch from `origin/HEAD` when possible instead of assuming `main` or `master`.
3. Make sure the work happens on a feature branch, not the default branch.
4. Update the default branch from the remote before branching.
5. Re-apply any local work onto the feature branch if the tree started dirty.
6. Use `uv` for Python environment sync, execution, and lockfile updates when the repo is `uv`-based.
7. Run relevant checks in that same environment.
8. Commit, push, and auto-create a PR when the task calls for publication.

## Branching Rules

- Never start implementation directly on the local default branch.
- If the user already gave a branch name, use it unless it would overwrite existing work.
- If there are uncommitted changes on the default branch, preserve them first, update the default branch, create the feature branch from the updated base, then restore the work on that branch.
- Prefer branch names with a clear task summary, such as `codex/fix-foo` or `codex/add-bar`.

## Update Sequence

Use a sequence equivalent to this:

```bash
git status --short --branch
git fetch origin
git checkout <default-branch>
git pull --ff-only
git checkout -b <feature-branch>
```

If the tree started dirty, stash or otherwise preserve local edits before updating the default branch, then restore them after switching to the feature branch.

## Uv Rules

- If `uv.lock` exists, or `pyproject.toml` clearly describes a `uv` workflow, treat `uv` as the source of truth for Python dependencies.
- Prefer `uv sync`, `uv run`, and `uv lock` over ad hoc `pip install`, `python`, or `pytest` commands.
- Run project checks through `uv run`, for example `uv run pytest`.
- If the repo is not `uv`-based, use the repo's existing environment manager instead of forcing `uv`.
- If legacy files like `requirements.txt` exist but are not the intended source of truth, consolidate carefully and update docs so only one Python environment workflow remains.

## Dirty Tree Handling

- Do not discard user changes.
- If moving work off the default branch, preserve the local state with a safe mechanism such as `git stash push -u` before updating the base branch.
- Restore the saved work only after the feature branch is created from the updated default branch.
- If stash application fails, recover carefully and avoid destructive resets unless the user explicitly asks for them.

## PR Finish

- Prepare a focused commit on the feature branch.
- Push that branch, not the default branch.
- Auto-attempt PR creation with the best available GitHub path in the environment, such as the GitHub app integration or `gh`.
- If automated PR creation is blocked by permissions, still push the branch and provide the direct PR creation URL plus the ready-to-use PR title and body.
- Open the PR with a clear title and a structured body.
- Always include a `Summary` section in the PR body that explains what changed and why.
- Always include a `Test Plan` section in the PR body with exact commands that were run and their outcomes.
- Call out any residual risks, skipped checks, or follow-up work.
