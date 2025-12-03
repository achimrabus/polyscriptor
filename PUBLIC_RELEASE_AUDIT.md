# Public Release Audit (Work in Progress)

Branch: `work` (cleanup branch)

## Actions taken in this iteration
- Created a dedicated cleanup branch to prepare a safe public release.
- Removed bundled zip archives (`readme_draft.zip`, `edit_catmos.zip`) to avoid leaking embedded assets or metadata. These are now blocked by existing `.gitignore` patterns for compressed files.
- Added `docs/PUBLICATION_CHECKLIST.md` to track the specific sanitation steps (secret scans, doc classification, configs/notebooks/assets/tests, history rewrite).
- Added `examples/example_training_config.yaml` as a sanitized template that uses environment variables instead of hardcoded dataset paths or output locations.
- Updated `.gitignore` to keep any internal/private notes out of the public tree (`internal/`, `private_docs/`).
- Documented public-release hygiene in `README.md` so contributors avoid committing private configs, outputs, or non-redistributable assets.
- Ran an initial regex-based sweep (`rg -n "(api[_-]?key|token|secret)"`) to spot obvious hardcoded credentials; no concrete keys were observed, but a full history scan with a dedicated tool is still required.
- Inventoried high-risk Markdown files and captured proposed dispositions in `docs/DOCUMENT_CLASSIFICATION_STATUS.md` to guide what should be public, redacted, or kept private.
- Cataloged notebooks and GUI/API scripts that need output stripping or credential-handling verification in `docs/DOCUMENT_CLASSIFICATION_STATUS.md` and `docs/NOTEBOOK_CLEANUP_PLAN.md`.
- **New:** Removed vendor-facing API guides (`COMMERCIAL_API_VALIDATION_GUIDE.md`, `GEMINI_*` quickstarts/enhancement docs, `QWEN_AND_COMMERCIAL_API_GUIDE.md`, `QWEN3_TROUBLESHOOTING.md`) from the public tree and logged them in `docs/REMOVED_INTERNAL_DOCS.md` for private archival prior to history rewrite.
- **New:** Excised infrastructure and dataset investigation reports (`V4_DATASET_REPORT.md`, `SEGMENTATION_ANALYSIS_REPORT.md`, `SEGMENTATION_INVESTIGATION_SUMMARY.md`, `INVESTIGATION_SUMMARY.md`, `LINUX_SERVER_MIGRATION.md`, `REFERENCE_MOBAXTERM.md`) and recorded them in `docs/REMOVED_INTERNAL_DOCS.md` to keep environment-specific details private.

## High-risk areas identified for further sanitization
- **Root-level operational guides** (multiple `GEMINI_*`, `PYLAIA_*`, and other implementation reports) likely contain internal procedures and environment-specific details; review for public suitability or relocate to a private archive.
- **Training and preprocessing scripts** reference dataset-specific assumptions and local paths; provide redacted/public-safe examples under a `configs/` or `examples/` directory and move private variants out of the repo.
- **Notebooks** (`Fine_Tune_TrOCR.ipynb`, `inspect_*.ipynb`, etc.) may embed outputs or sample data; strip outputs or keep private.
- **Assets and logos** under `assets/` may include third-party or internal branding; verify licensing and replace with neutral placeholders if necessary.
- **GUI and API client scripts** accept API keys; confirm no keys are hardcoded and document environment-variable usage.
- **Git history** still contains previous versions of removed or sensitive files; plan a history rewrite (`git filter-repo`) before publishing.

## Next steps
- Run a full secret scan across the repository and history (e.g., `gitleaks`, `trufflehog`).
- Classify documentation into public vs. private; move public docs into `docs/` and excise internal notes from history.
- Replace private configs with parameterized examples; ensure tests/demos run on synthetic sample data (using `examples/example_training_config.yaml` as the pattern).
- Strip notebook outputs or remove them from the public set if they contain non-redistributable content.
- Validate assets/licensing and replace branding where needed.
- Add contributor guidance after the tree is sanitized.
