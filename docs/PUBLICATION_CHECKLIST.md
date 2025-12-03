# Public Release Checklist

This checklist tracks the concrete steps required before publishing the repository. Use it alongside `PUBLIC_RELEASE_AUDIT.md` to record decisions and findings.

## 1) Secrets and credentials
- [ ] Run a full secret scan on the working tree and git history (e.g., `gitleaks detect --source .` and `gitleaks detect --source . --log-opts="--all"`).
- [ ] Record manual sweeps (e.g., regex scans) and findings; current status: initial `rg` sweep found no hardcoded keys, full scan pending.
- [ ] Rotate and remove any discovered keys; replace inline values with environment variables.
- [ ] Add pre-commit hooks for secret scanning before pushing.

## 2) Documentation hygiene
- [ ] Classify documentation into **public** vs **internal**; move public docs to `docs/` and keep internal notes out of the public tree.
- [ ] Redact environment-specific details (hosts, buckets, dataset paths) from any documents that must stay public.
- [ ] Add a short public README section describing the safe entry points and examples.
- [x] Remove vendor-facing API playbooks from the public tree and log them for private archival (`docs/REMOVED_INTERNAL_DOCS.md`).
- [x] Remove infrastructure and dataset investigation reports (e.g., segmentation analyses, server migration notes) from the public tree and track them in `docs/REMOVED_INTERNAL_DOCS.md`.

## 3) Configs and scripts
- [ ] Replace dataset-specific configs with sanitized examples under `examples/` and point scripts to use environment variables.
- [ ] Confirm no scripts bake in private paths or API endpoints; refactor to accept parameters.
- [ ] Document required environment variables in the README and example configs.

## 4) Notebooks
- [ ] Strip outputs/metadata from notebooks; remove any embedded samples that are not redistributable.
- [ ] Keep heavy experimental notebooks private or summarize them as markdown guides.

## 5) Assets and branding
- [ ] Inventory images/fonts; confirm licenses and remove non-redistributable branding.
- [ ] Provide neutral placeholders for demos and note attribution requirements.

## 6) Tests and demos
- [ ] Ensure tests run on synthetic/sample data; skip GPU-only or private-data tests in the public suite.
- [ ] Add a minimal example dataset (or generator) for public CI validation.

## 7) History hygiene
- [ ] After sanitizing files, rewrite history (`git filter-repo`) to purge removed sensitive content.
- [ ] Tag a release candidate from the cleaned branch and archive a record of the checks performed.
