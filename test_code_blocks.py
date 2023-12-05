"""A simple test runner for Python code blocks in Markdown files.

This script does not follow any formal Markdown format: it simply reads the input file
line by line, and lines starting with "<!-- test " are treated as test declarations.
For details, see the source code.
"""

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any, Dict, List

import tomli


def main(argv: List[str]) -> None:
    assert Path(__file__).absolute() in Path.cwd().iterdir()

    parser = argparse.ArgumentParser(
        description='Test Python code blocks in a Markdown file.'
    )
    parser.add_argument('path', metavar='PATH', help='The markdown file.')
    args = parser.parse_args(argv[1:])

    shared_globals: Dict[str, Dict[str, Any]] = {}
    try:
        exec(
            tomli.loads(Path('pyproject.toml').read_text())
            .get('tool', {})
            .get('test_code_blocks', {})
            .get('setup', 'pass'),
            {},
            shared_globals,
        )
    except Exception as err:
        raise RuntimeError(
            'Failed to execute the test setup from pyproject.toml'
        ) from err
    named_globals: Dict[str, Dict[str, Any]] = {}

    lines = enumerate(Path(args.path).read_text().splitlines())
    n_tests = 0
    for index, line in lines:
        # >>> If the line is not a start of test, go the next one.
        if not line.startswith('<!-- test '):
            continue

        test_header = line
        if not test_header.endswith(' -->'):
            raise RuntimeError(f'Invalid test header: {test_header}')
        if not next(lines)[1].startswith('```python'):
            raise RuntimeError(
                f'The test header "{test_header}"'
                ' is not follwed by a Python code blocks'
            )

        # >>> Assemble and print the code block.
        print('-' * 80)
        print('\033[4m' + f'# Test {n_tests + 1} (line {index})' + '\033[0m')
        print()
        testsource = '\n'.join(
            itertools.takewhile(
                lambda line_: line_ != '```',  # Take until the end of the code block.
                map(lambda index_line: index_line[1], lines),  # Drop line indices.
            ),
        )
        print(testsource)
        print()

        # >>> Parse the test header and prepare globals.
        # (A) <!-- test -->
        #     A simple test that relies only on the global setup from pyproject.toml.
        # (B) <!-- test name -->
        #     A named test that preserves its globals between code blocks.
        # (C) <!-- test name new_name -->
        #     A fork of one named test (name) into a new named test (new_name).
        #     The globals of the original test (name) remain unchanged.

        # (A)
        if test_header == '<!-- test -->':
            test_globals = shared_globals.copy()

        else:
            # ['<!--', 'test', name[, new_name], '-->']
            test_header_parts = test_header.strip().split()

            # (B)
            if len(test_header_parts) == 4:
                name = test_header_parts[2]
                test_globals = named_globals.get(name)
                if test_globals is None:
                    test_globals = shared_globals.copy()
                    named_globals[name] = test_globals

            # (C)
            elif len(test_header_parts) == 5:
                name = test_header_parts[2]
                new_name = test_header_parts[3]
                # '_' is a special value of new_name:
                # The header "<!-- test A _ -->" means that the test A is forked
                # (i.e. its globals are protected from changes), but the newly created
                # globals are not stored anywhere.

                if new_name != '_' and new_name in named_globals:
                    raise RuntimeError(
                        f'Invalid test header: {test_header}'
                        f' (test "{new_name}" already exists)'
                    )
                if name == new_name:
                    raise RuntimeError(
                        f'Invalid test header: {test_header}'
                        ' (test names must be different)'
                    )
                if name not in named_globals:
                    raise RuntimeError(
                        f'Invalid test header: {test_header} (the test "{name}" does'
                        ' not have any state yet, so it cannot be forked)'
                    )

                test_globals = shared_globals.copy()
                test_globals.update(named_globals[name])
                if new_name != '_':
                    named_globals[new_name] = test_globals

            else:
                raise RuntimeError(f'Invalid test header: {test_header}')

        exec(testsource, test_globals)
        del test_globals
        n_tests += 1

    print('-' * 80)
    print('\033[92m' + f'SUCCESS ({n_tests}/{n_tests})' + '\033[0m')


if __name__ == '__main__':
    main(sys.argv)
