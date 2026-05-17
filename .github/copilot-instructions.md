---
applyTo: '**/*.{py,md,yaml,json,ps1,txt}'
description: "SHARP-LLM repository conventions: keep README.md synchronized with plan and implementation changes."
---
# SHARP-LLM Repository Instructions

## README Synchronization Rule

Whenever you make changes to any of the following, you MUST also update `README.md` to reflect the new state:

### Triggers for README Update

1. **Plan changes** — Any modification to `context/journal-paper/the-plan.md` (new phases, timeline shifts, status changes)
2. **New source files** — Creating files in `src/` (update Project Structure section)
3. **New scripts** — Creating files in `scripts/` (update Project Structure section)
4. **Status changes** — When a component moves from planned → in-progress → complete (update Current Status table)
5. **New dependencies** — Changes to `requirements.txt` (verify Setup section is still accurate)
6. **New experiments** — Adding experiment configurations or results (update Experiments table)
7. **Model changes** — Retraining models or adding new ones (update HVSS ML Models table or Supported Models table)
8. **Configuration changes** — Changes to `config.yaml` that affect usage instructions

### What to Update in README

- **Current Status table** — Mark items ✅, 🔜, or 🚧 as appropriate
- **Project Structure tree** — Add/remove files to match actual layout
- **HSVSS section** — Update dimension details, R² scores, or scoring formula changes
- **Roadmap table** — Reflect actual progress against timeline
- **Setup section** — If Python version or install steps change
- **Commands/examples** — If CLI interfaces change

### How to Update

- Keep changes minimal and factual — don't rewrite sections unnecessarily
- Match the existing README style (tables, code blocks, section headers)
- If a new major section is needed, add it in logical order relative to existing sections
- Update the plan file (`context/journal-paper/the-plan.md`) status tracking tables too

## Plan File Rule

All architectural decisions, phase transitions, and implementation milestones must be recorded in `context/journal-paper/the-plan.md`. When completing a fix or implementing a new module:
- Add a row to the relevant status table
- Note the date of completion
- Record any key metrics (R² scores, F1, latency, etc.)

## Git Commit Rule

When the user says **"commit"**, **"check in"**, or **"save changes"**, stage all modified/new files and commit with a concise message. Use this format:

```powershell
git add -A; git commit -m "<type>: <short description>"
```

Commit types:
- `feat` — New feature or module
- `fix` — Bug fix or correction
- `refactor` — Code restructuring without behavior change
- `docs` — Documentation-only changes (README, plan, comments)
- `chore` — Dependency updates, config changes, tooling
- `data` — New mappings, datasets, or model artifacts

Rules:
- Keep the message under 72 characters
- Do NOT push unless the user explicitly says "push"
- If changes span multiple concerns, use the dominant type
- Always confirm the commit message with the user before running
