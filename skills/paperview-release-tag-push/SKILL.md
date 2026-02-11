---
name: paperview-release-tag-push
description: Handle release requests like 提交新vX.X.X版本 / release vX.X.X by completing the full Git release flow: verify clean state, ensure version/changelog are updated, commit release changes, create an annotated tag matching the version, and push both branch and tag to origin.
---

# Paperview Release Tag Push

## Required Trigger

- Use this skill when the user asks for a versioned release commit such as:
  - `提交新v0.4.0版本`
  - `release v0.4.0`
  - `发版 vX.X.X`

## Required Workflow

1. Parse target version from user message (must be `vX.Y.Z`).
2. Validate local repository state:
   - `git status --short` should include only intended release changes.
   - If missing release version updates, stop and report what is missing.
3. Ensure release metadata is consistent:
   - `manifest.json` version should be `X.Y.Z` (without `v` prefix).
   - `CHANGELOG.md` should include a top section for `vX.Y.Z` with current date.
4. Run validations/build before release commit:
   - Run relevant tests if available.
   - Run `bash scripts/build_xpi.sh`.
5. Create release commit:
   - `git add -A`
   - `git commit -m "chore: release vX.Y.Z"`
6. Create annotated tag:
   - `git tag -a vX.Y.Z -m "vX.Y.Z"`
7. Push release to remote:
   - `git push origin <current-branch>`
   - `git push origin vX.Y.Z`
8. Report completion with:
   - commit hash
   - pushed branch
   - pushed tag

## Safety Rules

- Never use force push for release flow.
- Never overwrite an existing tag:
  - If `vX.Y.Z` already exists locally or remotely, stop and ask user whether to bump version.
- Do not amend release commit unless user explicitly requests it.
