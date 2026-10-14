#!/usr/bin/env python3
"""Compile and run the runnable examples embedded in a Markdown document.

``spy`` compiles Python functions from their *source* (``inspect.getsource``),
so an example cannot be ``exec``-ed straight from a string: each example is
materialized as a real ``.py`` file (under the temporary directory) and then
executed, which lets ``spy`` find its source.

Conventions:

* A fenced code block whose info string is exactly ``python`` is **runnable**;
  its body is a self-contained module (definitions plus ``assert``-style
  checks that force compilation).
* Any other info string (``python,norun``, ``text``, ``mermaid``, ...) is
  ignored, so fragments and non-compiling illustrations can be shown too.
* The **first** runnable block is the shared *preamble*: it is prepended to
  every other runnable block (and also run once on its own), so examples can
  omit the imports.

Usage::

    python3 tools/run_doc_examples.py [doc/features.md ...]
"""

import re
import sys
import tempfile
import traceback
from pathlib import Path

# the ``spy`` package lives at the repository root (the parent of ``tools``);
# running a script puts its own directory on ``sys.path``, not the cwd, so add
# the root explicitly to make ``import spy.compiler`` resolve.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FENCE = re.compile(r'^\s*```\s*(\S*)\s*$')


def iter_blocks(text: str):
    """Yield ``(info, start_line, code)`` for every fenced code block."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if m is None:
            i += 1
            continue
        info = m.group(1)
        start = i + 1
        j = start
        while j < len(lines) and FENCE.match(lines[j]) is None:
            j += 1
        yield info, start, '\n'.join(lines[start:j])
        i = j + 1


def runnable_blocks(text: str) -> list[tuple[int, str]]:
    return [(start, code) for info, start, code in iter_blocks(text) if info == 'python']


def run(path: Path) -> list[str]:
    blocks = runnable_blocks(path.read_text())
    if not blocks:
        return [f'{path}: no runnable ```python blocks']

    _, preamble = blocks[0]
    failures: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix='spy-doc-examples-'))

    for index, (start, code) in enumerate(blocks):
        # the preamble is prepended to every block but the preamble itself
        source = code if index == 0 else preamble + '\n\n' + code
        offset = 0 if index == 0 else preamble.count('\n') + 2
        py = tmp / f'block_{index:03d}.py'
        py.write_text(source)
        try:
            namespace = {'__name__': f'md_block_{index}', '__file__': str(py)}
            exec(compile(source, str(py), 'exec'), namespace)  # noqa: S102
        except BaseException:  # noqa: BLE001
            tb = traceback.format_exc()
            failures.append(
                f'{path}:{start} (block #{index}, doc line {start + offset}):'
                f'\n{tb}'
            )
    return failures


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in (argv[1:] or ['doc/features.md'])]
    all_failures: list[str] = []
    for path in paths:
        if not path.exists():
            print(f'{path}: not found', file=sys.stderr)
            return 2
        all_failures.extend(run(path))
    if all_failures:
        print('\n'.join(all_failures))
        print(f'\n{len(all_failures)} example(s) FAILED')
        return 1
    print(f'OK: all examples in {len(paths)} file(s) ran')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
