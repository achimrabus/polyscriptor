# Internal Documents Removed from Public Tree

The following files were removed during public-release cleanup because they contain vendor-specific operational guidance, pricing notes, or troubleshooting details that should stay private. Keep these files in a private archive and ensure they are purged from git history before publishing.

| Removed file | Reason for removal | Replacement/next step |
| --- | --- | --- |
| `COMMERCIAL_API_VALIDATION_GUIDE.md` | Contains API validation flow and usage notes that may reference internal environments. | Keep privately; replace with a short public README entry that points to environment variables and safe usage patterns. |
| `GEMINI_3_ENHANCEMENTS.md` | Vendor-specific tuning and operational notes. | Omit from public repo; summarize only high-level learnings if needed. |
| `GEMINI_DOCS_INDEX.md` | Internal documentation index for Gemini playbooks. | Do not republish; restructure public docs to exclude vendor-specific indexes. |
| `GEMINI_FREE_TIER_GUIDE.md` | Pricing/usage details for a specific vendor tier. | Remove; public docs should reference official vendor pages instead. |
| `GEMINI_IMPLEMENTATION_SUMMARY.md` | Implementation notes tied to internal workflows. | Keep private; create a neutral public quickstart if necessary. |
| `GEMINI_QUICK_START.md` | Internal quickstart that may assume private assets or keys. | Replace with a sanitized public quickstart after secret scan. |
| `GEMINI_UI_LAYOUT.md` | Screens/layouts potentially tied to internal branding. | Archive privately; rebuild public UI docs with neutral assets. |
| `QWEN_AND_COMMERCIAL_API_GUIDE.md` | Combined guide with API usage details and key handling workflows. | Remove from public; public docs should only reference environment variables and generic instructions. |
| `QWEN3_TROUBLESHOOTING.md` | Troubleshooting notes that reference specific accounts/configurations. | Keep private; provide generic troubleshooting tips in public docs. |
| `V4_DATASET_REPORT.md` | Dataset analysis with potentially sensitive metrics and data references. | Keep in private archive; summarize sanitized findings if needed. |
| `SEGMENTATION_ANALYSIS_REPORT.md` | Internal segmentation evaluation that may include dataset specifics. | Retain privately; produce a redacted public summary only if necessary. |
| `SEGMENTATION_INVESTIGATION_SUMMARY.md` | Investigation notes with environment and data details. | Archive privately; remove from public history during rewrite. |
| `INVESTIGATION_SUMMARY.md` | Broad internal investigation notes with environment-specific context. | Keep private; capture only high-level learnings in public docs. |
| `LINUX_SERVER_MIGRATION.md` | Server migration steps tied to private infrastructure. | Do not republish; ensure history rewrite removes it. |
| `REFERENCE_MOBAXTERM.md` | Host-specific remote access instructions. | Keep private; no public replacement required. |

## Next actions
- Purge these files from git history (`git filter-repo`) before creating the public release.
- Double-check other Markdown reports for similar vendor-specific content and move them here if removed.
- Update public documentation to rely on environment variables and neutral guidance instead of internal playbooks.
