---
name: paperview-uv-environment
description: Standardize all Python environment and dependency management on uv for this repo. Use when installing new packages, running Python scripts, starting local services, or updating lockfiles to ensure uv is used consistently.
---

# Paperview Uv Environment

## Overview

Enforce uv as the single entry point for Python execution and dependency management in this repository.

## Rules

- Use `uv run` to execute Python scripts and start services.
- Use `uv add` for new dependencies; avoid `pip install`.
- Keep `pyproject.toml` and `uv.lock` as the source of truth.
- Do not create ad-hoc virtualenvs; use uv-managed environments only.

## Common Operations

- Run a script: `uv run python <script.py> [args]`
- Start a local service: `uv run python local_service.py --port <port>`
- Add a dependency: `uv add <package>`
- Sync environment from lock: `uv sync`

## Notes

- If a user requests a new package or tool, confirm it will be added via `uv add`.
- If a user requests a Python server, launch it with `uv run` and avoid raw `python` or `pip`.
