"""Rich-based progress bars with a tqdm-compatible call signature.

Usage: ``pbar = tqdm(iterable, desc="...", unit="image", disable=False)``
supports the subset of the tqdm API the CLIs use: iteration,
``set_postfix(dict)`` and ``close()``.
"""

from rich.progress import (
    Progress,
    BarColumn,
    MofNCompleteColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class _RichPBar:
    """Thin tqdm-compatible wrapper around rich.Progress."""

    def __init__(self, iterable, desc="", unit="it", disable=False, **_ignored):
        self._iterable = iterable
        self._disable = disable
        self._desc = desc
        self._unit = unit
        self._postfix = {}
        self._progress = Progress(
            TextColumn("[bold blue]{task.fields[desc]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("• {task.fields[postfix]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
            disable=disable,
        )
        self._task = None

    def _postfix_text(self):
        return " ".join(f"{k}={v}" for k, v in self._postfix.items()) or "-"

    def __iter__(self):
        with self._progress:
            self._task = self._progress.add_task(
                self._desc, total=len(self._iterable),
                desc=self._desc, postfix="-",
            )
            for item in self._iterable:
                yield item
                self._progress.update(self._task, advance=1, postfix=self._postfix_text())

    def set_postfix(self, d, **_kw):
        self._postfix.update(d)

    def set_description(self, desc):
        self._desc = desc
        if self._task is not None:
            self._progress.update(self._task, description=desc, desc=desc)

    def close(self):
        try:
            self._progress.stop()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def tqdm(iterable, desc="", unit="it", disable=False, **kwargs):
    """tqdm-compatible factory backed by rich."""
    return _RichPBar(iterable, desc=desc, unit=unit, disable=disable, **kwargs)
