"""Polyscriptor command line interface.

All subcommands except ``serve`` are thin pass-through wrappers around the
underlying module CLIs, so ``polyscriptor batch --help`` shows that module's
own flags. This keeps one canonical entry point without duplicating the
flag definitions.
"""

import sys
from importlib import import_module

import click

_CONTEXT = dict(ignore_unknown_options=True, allow_extra_args=True)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="polyscriptor", prog_name="polyscriptor")
def main():
    """Polyscriptor — multi-engine HTR toolkit for historical manuscripts."""


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address.")
@click.option("--port", default=8765, show_default=True, type=int, help="Port.")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (development).")
def serve(host, port, reload):
    """Start the Polyscriptor web interface."""
    import uvicorn

    uvicorn.run(
        "polyscriptor.web.polyscriptor_server:app",
        host=host,
        port=port,
        reload=reload,
    )


def _passthrough(name, module_path, short_help):
    """Subcommand forwarding all args to ``module_path.main()`` (argparse-based)."""

    @main.command(name=name, context_settings=_CONTEXT, short_help=short_help)
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    def cmd(args):
        mod = import_module(module_path)
        sys.argv = [name] + list(args)
        sys.exit(mod.main())


_passthrough(
    "batch", "polyscriptor.cli.batch",
    "Batch-transcribe a folder of images (flags: `polyscriptor batch -h`).",
)
_passthrough(
    "transkribus", "polyscriptor.data.transkribus_parser",
    "Prepare a dataset from Transkribus PAGE XML exports.",
)
_passthrough(
    "alto", "polyscriptor.data.alto_parser",
    "Prepare a dataset from ALTO XML exports.",
)
_passthrough(
    "pylaia-convert", "polyscriptor.data.convert_to_pylaia",
    "Convert a CSV dataset to CRNN-CTC (PyLaia) format.",
)
_passthrough(
    "train-crnn", "polyscriptor.training.crnn_train",
    "Train a CRNN-CTC model.",
)
_passthrough(
    "train-trocr", "polyscriptor.training.trocr_train",
    "Fine-tune a TrOCR model from a YAML config.",
)
_passthrough(
    "infer-page", "polyscriptor.inference.page",
    "Transcribe a single page image (segmentation + recognition).",
)
_passthrough(
    "segment", "polyscriptor.cli.pagexml_segment",
    "Batch line segmentation to PAGE XML (Kraken classical/neural).",
)


if __name__ == "__main__":
    main()
