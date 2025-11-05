# Np Dist2

[![PyPI](https://img.shields.io/pypi/v/np-dist2.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/np-dist2.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/np-dist2)][pypi status]
[![License](https://img.shields.io/pypi/l/np-dist2)][license]

[![Read the documentation at https://np-dist2.readthedocs.io/](https://img.shields.io/readthedocs/np-dist2/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/kostyfisik/np-dist2/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/kostyfisik/np-dist2/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Ruff codestyle][ruff badge]][ruff project]

[pypi status]: https://pypi.org/project/np-dist2/
[read the docs]: https://np-dist2.readthedocs.io/
[tests]: https://github.com/kostyfisik/np-dist2/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/kostyfisik/np-dist2
[pre-commit]: https://github.com/pre-commit/pre-commit
[ruff badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff project]: https://github.com/charliermarsh/ruff

## Features

- 3D grid generation for scientific computing
- Lattice analysis tools for crystal structure analysis
- 3D rotation functions for vector transformations
- Command-line interface for scientific data analysis
- Performance-optimized code with optional mypyc compilation

## Requirements

- Python 3.9 or later
- NumPy
- SciPy
- Matplotlib
- Click

## Installation

You can install _Np Dist2_ via [pip] from [PyPI]. The package is distributed as a pure Python package, but also with pre-compiled wheels for major platforms, which include performance optimizations.

```console
$ pip install np-dist2
```

The pre-compiled wheels are built using `mypyc` and will be used automatically if your platform is supported. You can check the files on PyPI to see the list of available wheels.

## Usage

### As a Library

```python
import numpy as np
from np_dist2.grid_generator import gen_unit_grid
from np_dist2.lattice import get_lat_by_id

# Generate a 3D grid
grid = gen_unit_grid(2)

# Analyze lattice properties
r, lat, num = get_lat_by_id(0, grid)
```

### Command-line Interface

Np Dist2 provides a command-line interface with several scientific analysis tools:

```console
$ np-dist2 --help
```

Available commands:
- `time-avg`: Average trajectory data over time steps
- `plot-dist`: Plot distance distribution
- `plot-ion`: Plot ion distribution
- `plot-lat`: Plot lattice structure
- `plot-rdf`: Plot radial distribution function

## Development

To contribute to this project, please see the [Contributor Guide].

### Mypyc Compilation

This project can be compiled with `mypyc` to produce a high-performance version of the package. The compilation is optional and is controlled by an environment variable.

To build and install the compiled version locally, you can use the `tests_compiled` nox session:

```console
$ nox -s tests_compiled
```

This will set the `NP_DIST2_COMPILE_MYPYC=1` environment variable, which triggers the compilation logic in `setup.py`. The compiled package will be installed in editable mode in a new virtual environment.

You can also build the compiled wheels for distribution using the `cibuildwheel` workflow, which is configured to run on releases. If you want to build the wheels locally, you can use `cibuildwheel` directly:

```console
$ pip install cibuildwheel
$ export NP_DIST2_COMPILE_MYPYC=1
$ cibuildwheel --output-dir wheelhouse
```

This will create the compiled wheels in the `wheelhouse` directory.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_Np Dist2_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [uv hypermodern python cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[uv hypermodern python cookiecutter]: https://github.com/bosd/cookiecutter-uv-hypermodern-python
[file an issue]: https://github.com/kostyfisik/np-dist2/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/kostyfisik/np-dist2/blob/main/LICENSE
[contributor guide]: https://github.com/kostyfisik/np-dist2/blob/main/CONTRIBUTING.md
[command-line reference]: https://np-dist2.readthedocs.io/en/latest/usage.html