---
name: hemoc-branches
description: Audit HEMOC branches and PRs. Use when managing the 21+ open PRs, checking branch freshness, or planning merges across the HEMOC repository.
allowed-tools: Read, Grep, Glob, Bash(git *), Bash(cd * && git *), Bash(gh *)
---

# HEMOC Branch Auditor

Audit the HEMOC repository's branch structure, open PRs, and merge state.
The HEMOC repo has 21+ open PRs across branching chains — this skill helps manage them.

## HEMOC Repository

```
/home/user/HEMOC-Stain-Glass-Flower
```

## Branch Lineage (Known Chain)

The primary development chain is linear:

```
main
  └── claude/atom-of-thoughts-repo-tD7FD
        └── claude/review-testing-progress-F6a3C
              └── claude/restore-linear-path-decoding-EELHW
                    └── hemoc-visual-system-init-4592349023335104422  (most complete)
```

Side branches fork off `restore-linear-path`:
- 6+ `codex/2026-02-06/conduct-cross-repository-analysis-*` branches
- `codex/2026-01-29/create-catalog-index-*`
- `codex/2026-02-12/review-repo-and-create-ontology*`
- `codex/2026-02-14/review-repo-and-create-ontology*`

Other branches:
- `claude/project-digest-*` — cross-repo synthesis work
- `claude/organize-webgpu-webgl-shaders-*`
- `claude/kirigami-quaternion-framework-*`
- `revert-*` branches for undone merges

## Audit Steps

### 1. List All Branches with Freshness

```bash
cd /home/user/HEMOC-Stain-Glass-Flower
git fetch --all -q 2>/dev/null
git for-each-ref --sort=-committerdate refs/remotes/origin/ \
  --format='%(committerdate:relative)|%(refname:short)|%(subject)' | head -30
```

For each branch, report:
- Last commit date (relative)
- Last commit message
- Whether it has a corresponding open PR

### 2. Stale Branch Detection

Flag branches where:
- Last commit > 14 days ago AND no open PR
- Branch has 0 unique commits vs its parent (empty branch)
- Branch is a `codex/` auto-generated branch with no meaningful diff

### 3. PR Status (if gh CLI available)

```bash
cd /home/user/HEMOC-Stain-Glass-Flower
gh pr list --state open --limit 30 --json number,title,headRefName,baseRefName,createdAt,isDraft
```

For each PR:
- Number, title, head → base
- Age (created date)
- Draft status
- Merge conflict status: `gh pr view <N> --json mergeable`

### 4. Unique Commit Analysis

For the main chain branches, count unique commits:
```bash
git log --oneline origin/main..origin/claude/atom-of-thoughts-repo-tD7FD | wc -l
git log --oneline origin/claude/atom-of-thoughts-repo-tD7FD..origin/claude/review-testing-progress-F6a3C | wc -l
# etc.
```

### 5. Merge Recommendations

Based on the audit, recommend:

**Merge candidates** — branches with:
- All tests passing
- Meaningful unique commits
- No merge conflicts with their target
- Clear value-add to the target branch

**Close candidates** — branches with:
- 0 unique commits (duplicate of parent)
- Codex auto-generated branches that added only docs/reorganization
- Superseded by later work on a different branch

**Keep candidates** — branches with:
- Active work in progress
- Branch-specific files not available elsewhere (benchmark_contract.md, visual system)
- Open PRs with review comments

## Output Format

```
## HEMOC Branch Audit

### Branch Freshness (most recent first)
| Branch | Last Commit | Unique Commits | PR | Status |
|--------|-------------|----------------|-----|--------|
| hemoc-visual-system-init-* | 3 days ago | 12 | #N | Active |
| claude/restore-linear-path-* | 5 days ago | 8 | #N | Active |
| codex/2026-02-06/conduct-*-yoawus | 9 days ago | 2 | #N | Stale |
| ... | ... | ... | ... | ... |

### Recommendations
**Merge**: [branches ready to merge]
**Close**: [stale/empty branches]
**Keep**: [branches with unique value]

### Branch-Specific Assets
| Asset | Only on Branch |
|-------|---------------|
| docs/benchmark_contract.md | hemoc-visual-system-init-* |
| scripts/validate_results_schema.py | hemoc-visual-system-init-* |
| docs/HEMOC_ARCHITECTURE_DETAILED.md | hemoc-visual-system-init-* |
| hemoc-visual-system/ scaffold | hemoc-visual-system-init-* |
```

## Critical Context

- The `hemoc-visual-system-init-4592349023335104422` branch has the most complete state
  for Recovery Plan work (benchmark contract, schema validator, recovery audit).
- Branch from this one for new Recovery Plan phases.
- The `revert-*` branches exist because some codex merges were undone. Don't re-merge them.
