# Notebook Cleanup Plan

Strip outputs and review each notebook for private paths, sample data, or credentials before publication.

| Notebook | Action | Notes |
| --- | --- | --- |
| `Fine_Tune_TrOCR.ipynb` | Clear outputs and scrub dataset paths | Contains tokenizer installs and training snippets referencing local file systems. |
| `inspect_prosta_mova_v3.ipynb` | Clear outputs or keep private | Likely includes dataset inspection outputs. |
| `inspect_prosta_mova_v4.ipynb` | Clear outputs or keep private | Same as above; may contain sample images/paths. |
| `inspect_ukrainian_v2.ipynb` | Clear outputs or keep private | Verify no proprietary samples are embedded. |
| `lineSegmentation.ipynb` | Clear outputs and confirm sample data is synthetic | Ensure no GT annotations from private datasets. |

## Cleanup steps
- Run `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace <notebook>` for each kept notebook.
- Replace dataset-specific paths with placeholders or environment variables.
- Move notebooks that cannot be fully sanitized into a private archive excluded from the public release.
