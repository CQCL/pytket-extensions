# Pytket Extensions

This repository gives an overview of Python extension modules for 
[pytket](https://cqcl.github.io/tket/pytket/api/index.html), a quantum computing toolkit and optimising compiler developed by Quantinuum.

Each of these modules provides one or several _backends_ (interfaces to devices
or simulators), _frontends_ (interfaces to other high-level quantum languages),
or both.

All the extensions are written in pure Python, and depend on the `pytket`
module.

The extensions can be found in other repos:

* [pytket-aqt](https://github.com/CQCL/pytket-aqt)
* [pytket-braket](https://github.com/CQCL/pytket-braket)
* [pytket-cirq](https://github.com/CQCL/pytket-cirq)
* [pytket-ionq](https://github.com/CQCL/pytket-ionq)
* [pytket-iqm](https://github.com/CQCL/pytket-iqm)
* [pytket-pennylane](https://github.com/CQCL/pytket-pennylane)
* [pytket-projectq](https://github.com/CQCL/pytket-projectq)
* [pytket-pyquil](https://github.com/CQCL/pytket-pyquil)
* [pytket-pysimplex](https://github.com/CQCL/pytket-pysimplex)
* [pytket-pyzx](https://github.com/CQCL/pytket-pyzx)
* [pytket-qiskit](https://github.com/CQCL/pytket-qiskit)
* [pytket-qsharp](https://github.com/CQCL/pytket-qsharp)
* [pytket-quantinuum](https://github.com/CQCL/pytket-quantinuum)
* [pytket-qujax](https://github.com/CQCL/pytket-qujax)
* [pytket-qulacs](https://github.com/CQCL/pytket-qulacs)
* [pytket-stim](https://github.com/CQCL/pytket-stim)

Other extensions can be found in other repos:

* [pytket-braket](https://github.com/CQCL/pytket-braket)
* [pytket-iqm](https://github.com/CQCL/pytket-iqm)
* [pytket-quantinuum](https://github.com/CQCL/pytket-quantinuum)
* [pytket-qujax](https://github.com/CQCL/pytket-qujax)

Note that most backend providers require you to set up credentials in order to
submit jobs over the internet. These should be obtained directly from the
providers.

## Installing and using an extension

Each of the extensions can be installed using `pip`. For example:

```shell
pip install pytket-qiskit
```

This will install `pytket` if it isn't already installed, and add new classes
and methods into the `pytket.extensions` namespace.

Full documentation for all these extension module is available
[here](https://cqcl.github.io/pytket-extensions/api/index.html).

## Bugs, support and feature requests

Please file bugs and feature requests on the Github
[issue tracker](https://github.com/CQCL/pytket-extensions/issues).

There is also a Slack channel for discussion and support. Click [here](https://tketusers.slack.com/join/shared_invite/zt-18qmsamj9-UqQFVdkRzxnXCcKtcarLRA#/shared-invite/email) to join.

## Supported Backends

For a list of accessible backends refer to the [documentation](https://cqcl.github.io/pytket-extensions/api/index.html).

## Development

To install an extension in editable mode, simply change to its subdirectory
within the `modules` directory, and run:

```shell
pip install -e .
```

If you wish to write your own backend extension for `pytket`, we recommend
looking at the example notebook
[here](https://github.com/CQCL/pytket/blob/main/examples/creating_backends.ipynb)
which explains how to do so.

If you would like to add it to this repo, please follow the existing code and
naming convetions, and make a PR as described below with your module as a new
subdirectory in `modules`.

## Contributing

Pull requests are welcome. To make a PR, first fork the repo, make your proposed
changes on the `develop` branch, and open a PR from your fork. If it passes
tests and is accepted after review, it will be merged in.

### Code style

#### Formatting

All code should be formatted using
[black](https://black.readthedocs.io/en/stable/), with default options. This is
checked on the CI. The CI is currently using version 20.8b1.

#### Type annotation

On the CI, [mypy](https://mypy.readthedocs.io/en/stable/) is used as a static
type checker and all submissions must pass its checks. You should therefore run
`mypy` locally on any changed files before submitting a PR. Because of the way
extension modules embed themselves into the `pytket` namespace this is a little
complicated, but it should be sufficient to run the script `modules/mypy-check`
(passing as a single argument the root directory of the module to test). The
script requires `mypy` 0.800 or above.

#### Linting

We use [pylint](https://pypi.org/project/pylint/) on the CI to check compliance
with a set of style requirements (listed in `modules/.pylintrc`). You should run
`pylint` over any changed files from the `modules` directory before submitting a
PR, to catch any issues.

### Tests

To run the tests for a module:

1. `cd` into that module's `tests` directory;
2. ensure you have installed `pytest`, `hypothesis`, and any modules listed in
the `test-requirements.txt` file (all via `pip`);
3. run `pytest`.

When adding a new feature, please add a test for it. When fixing a bug, please
add a test that demonstrates the fix.
