# Code Health and Maintainability Log

This document tracks significant code health improvements, refactoring efforts, and architectural changes aimed at improving the maintainability and readability of the Polytopal Projection Processing (PPP) platform.

## 2026-02-19: Script Deduplication via Symlinks

### 🎯 Issue
Identified significant code duplication between the root `scripts/` directory and the `_SYNERGIZED_SYSTEM/frontend/platform/scripts/` directory. 33 files were found to be bit-for-bit identical.

### 💡 Rationale
Duplicate files increase technical debt and the risk of divergence. Maintaining a single source of truth is critical for an academic project where consistency across experimental platforms is paramount.

### ✅ Action Taken
- Removed all 33 identical duplicate files from `_SYNERGIZED_SYSTEM/frontend/platform/scripts/`.
- Replaced them with relative symlinks pointing to the root `scripts/` directory (e.g., `ln -s ../../../../scripts/app.js app.js`).
- Verified that all symlinks resolve correctly and that the core test suite (`npm test`) remains passing.

### ✨ Result
- Reduced codebase size and eliminated redundant files.
- Ensured that any future changes to core logic are automatically reflected across all sub-systems.
- Preserved the existing directory structure required by frontend entry points while centralizing the implementation.

---

## 2026-02-19: Test Suite Deduplication via Symlinks

### 🎯 Issue
The `tests/` directory in the project root and `_SYNERGIZED_SYSTEM/frontend/platform/tests/` were found to be bit-for-bit identical, including the `fixtures/` subdirectory.

### 💡 Rationale
As with the core scripts, duplicating the test suite leads to maintenance overhead. Any new tests added to the root would need to be manually copied to the synergized sub-system.

### ✅ Action Taken
- Replaced all files and the `fixtures/` directory in `_SYNERGIZED_SYSTEM/frontend/platform/tests/` with relative symlinks to the root `tests/` directory.
- Verified that `npm test` still executes correctly (noting that some tests may require root-level context to find sample data).

## 2026-02-19: HTML Entry Point Deduplication

### 🎯 Issue
`phase-lock-live.html` was identical in both the root and `_SYNERGIZED_SYSTEM/frontend/platform/`.

### ✅ Action Taken
- Symlinked `_SYNERGIZED_SYSTEM/frontend/platform/phase-lock-live.html` to the root version.
- Note: `index.html` remains duplicated due to minor functional differences in navigation links, but is a candidate for future template-based unification.

---

## Future Recommendations

### 1. Unified Source (`src/`) Management
Many files in `src/lib/` are currently identical between root and `_SYNERGIZED_SYSTEM`. A strategy should be developed to share these core TypeScript components while allowing for the specific extensions (like `HemocPythonBridge.ts`) required by the synergized system.

### 2. Centralized Asset Management
Move shared images and styles into a common `assets/` directory at the root and link to them, rather than duplicating them in sub-system folders.
