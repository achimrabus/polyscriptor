# Documentation Classification Status (public-cleanup)

This log tracks which files should remain public, require redaction, or should be kept private before release. It focuses on high-risk Markdown reports, configs, and scripts surfaced during the audit.

## Markdown / Reports
| Path | Proposed disposition | Notes |
| --- | --- | --- |
| `README.md` | Public (keep) | Public entry point; keep sanitized guidance and links only. |
| `docs/PUBLICATION_CHECKLIST.md` | Public (keep) | Core release checklist. |
| `PUBLIC_RELEASE_AUDIT.md` | Public (keep) | Tracks decisions; safe once sensitive findings are omitted. |
| `COMMERCIAL_API_VALIDATION_GUIDE.md` | Private or redact | Contains internal validation flow; likely references credentials/endpoints. |
| `GEMINI_*` guides (e.g., `GEMINI_3_ENHANCEMENTS.md`, `GEMINI_QUICK_START.md`, `GEMINI_FREE_TIER_GUIDE.md`) | Private or heavily redacted | Operational playbooks and tuning specifics; likely internal. |
| `PYLAIA_*` reports and plans | Private or summarized | Training notes with dataset/environment details. |
| `QWEN3_TROUBLESHOOTING.md`, `QWEN_AND_COMMERCIAL_API_GUIDE.md` | Private or redact | API usage and troubleshooting likely include internal procedures. |
| `LOGO_GUIDE.md`, `LOGO_IMPLEMENTATION_SUMMARY.md` | Review licensing | Ensure no proprietary branding; replace with neutral assets. |
| `TRAIN_CER_LOGGING_EXPLANATION.md`, `INFERENCE_OPTIMIZATION_GUIDE.md` | Public with redaction | Keep technical guidance but scrub environment-specific paths. |
| `PREPROCESSING_CHECKLIST.md`, `DATASET_FORMAT_FIX.md` | Private or redact | May reference dataset structure; ensure no private samples. |
| `V4_DATASET_REPORT.md`, `SEGMENTATION_*` reports, `INVESTIGATION_SUMMARY.md` | Private archive | Likely contain internal dataset analytics. |
| `REFERENCE_MOBAXTERM.md`, `LINUX_SERVER_MIGRATION.md` | Private (removed) | Environment-specific server notes. |
| `README_POLYSCRIPTOR_BATCH_GUI.md`, `POLYSCRIPTOR_BATCH_GUI_PLAN.md` | Public with review | Keep if free of private endpoints/assets. |

## Scripts / configs needing attention
| Path(s) | Proposed disposition | Notes |
| --- | --- | --- |
| `diagnose_gemini_api.py`, `inference_commercial_api.py`, `list_gemini_models.py` | Public with verification | Confirm no hardcoded keys; require env vars. |
| `transcription_gui_qt.py`, `engines/qwen3_engine.py` | Public with credential handling review | GUI stores API keys locally; ensure instructions emphasize env vars and no commits of saved key files. |
| `start_*training.*`, `train_*.py`, `config_*.yaml` | Replace with examples | Move private configs out; rely on sanitized templates under `examples/`. |
| `create_demo_logo.py`, assets under `assets/` | Licensing review | Confirm redistributable assets or swap placeholders. |

## Removed/archived (private)
The following documents were removed from the public tree and logged in `docs/REMOVED_INTERNAL_DOCS.md` pending history rewrite:
- `COMMERCIAL_API_VALIDATION_GUIDE.md`
- `GEMINI_3_ENHANCEMENTS.md`
- `GEMINI_DOCS_INDEX.md`
- `GEMINI_FREE_TIER_GUIDE.md`
- `GEMINI_IMPLEMENTATION_SUMMARY.md`
- `GEMINI_QUICK_START.md`
- `GEMINI_UI_LAYOUT.md`
- `QWEN_AND_COMMERCIAL_API_GUIDE.md`
- `QWEN3_TROUBLESHOOTING.md`
- `V4_DATASET_REPORT.md`
- `SEGMENTATION_ANALYSIS_REPORT.md`
- `SEGMENTATION_INVESTIGATION_SUMMARY.md`
- `INVESTIGATION_SUMMARY.md`
- `LINUX_SERVER_MIGRATION.md`
- `REFERENCE_MOBAXTERM.md`

## Next classification actions
- Move approved public docs into `docs/` and exclude private/internal notes from the public tree.
- Redact or remove internal metrics, dataset paths, and server references.
- Pair each published script with an example config that uses environment variables and synthetic data.
