# Pytket Extensions

This repository contains a collection of Python extension modules for CQC's
[pytket](https://cqcl.github.io/tket/pytket/api/index.html) quantum SDK.

Each of these modules provides one or several _backends_ (interfaces to devices
or simulators), _frontends_ (interfaces to other high-level quantum languages),
or both.

All the extensions are written in pure Python, and depend on the `pytket`
module.

Code for the following extensions is included here, each within its own
subdirectory of the `modules` directory:

* `pytket-aqt`
* `pytket-braket`
* `pytket-cirq`
* `pytket-ionq`
* `pytket-iqm`
* `pytket-projectq`
* `pytket-pyquil`
* `pytket-pysimplex`
* `pytket-pyzx`
* `pytket-qiskit`
* `pytket-qsharp`
* `pytket-quantinuum`
* `pytket-qulacs`
* `pytket-stim`

See the individual `README` files for descriptions.

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

## Bugs and feature requests

Please file bugs and feature requests on the Github
[issue tracker](https://github.com/CQCL/pytket-extensions/issues).

## Supported Backends

Here you can see a list of supported backends sorted by category.

### QPUs

[IBMQBackend](https://cqcl.github.io/pytket-extensions/api/qiskit/api.html#pytket.extensions.qiskit.IBMQBackend)
A backend for running circuits on remote IBMQ devices.

[IonQBackend](https://cqcl.github.io/pytket-extensions/api/ionq/api.html#pytket.extensions.ionq.IonQBackend)
Interface to an IonQ device.

[ForestBackend](https://cqcl.github.io/pytket-extensions/api/pyquil/api.html#pytket.extensions.pyquil.ForestBackend)
Interface to an Rigetti device.

[ForestStateBackend](https://cqcl.github.io/pytket-extensions/api/pyquil/api.html#pytket.extensions.pyquil.ForestStateBackend)
State based interface to an Rigetti device.

[AQTBackend](https://cqcl.github.io/pytket-extensions/api/aqt/api.html#pytket.extensions.aqt.AQTBackend)
Interface to an AQT device or simulator.

[QuantinuumBackend](https://cqcl.github.io/pytket-extensions/api/quantinuum/api.html#pytket.extensions.quantinuum.QuantinuumBackend)
Interface to a Quantinuum device.

### Cloud access

[AzureBackend](https://cqcl.github.io/pytket-extensions/api/qsharp/api.html#pytket.extensions.qsharp.AzureBackend)
Backend for running circuits remotely using Azure Quantum devices and simulators.

[BraketBackend](https://cqcl.github.io/pytket-extensions/api/braket/api.html#pytket.extensions.braket.BraketBackend)
Interface to Amazon Braket service.

### Emulators

[IBMQEmulatorBackend](https://cqcl.github.io/pytket-extensions/api/qiskit/api.html#pytket.extensions.qiskit.IBMQEmulatorBackend)
A backend which uses the AerBackend to emulate the behaviour of IBMQBackend.

### State vector simulators

[CirqStateSampleBackend](https://cqcl.github.io/pytket-extensions/api/cirq/api.html#pytket.extensions.cirq.CirqStateSampleBackend)
Backend for Cirq statevector simulator sampling.

[CirqStateSimBackend](https://cqcl.github.io/pytket-extensions/api/cirq/api.html#pytket.extensions.cirq.CirqStateSimBackend)
Backend for Cirq statevector simulator state return.

[AerStateBackend](https://cqcl.github.io/pytket-extensions/api/qiskit/api.html#pytket.extensions.qiskit.AerStateBackend)
Backend for running simulations on the Qiskit Aer Statevector simulator.

[ProjectQBackend](https://cqcl.github.io/pytket-extensions/api/projectq/api.html#pytket.extensions.projectq.ProjectQBackend)
Backend for running statevector simulations on the ProjectQ simulator.

### Unitary simulators

[AerUnitaryBackend](https://cqcl.github.io/pytket-extensions/api/qiskit/api.html#pytket.extensions.qiskit.AerUnitaryBackend)
Backend for running simulations on the Qiskit Aer Unitary simulator.

### Density matrix simulator

[CirqDensityMatrixSampleBackend](https://cqcl.github.io/pytket-extensions/api/cirq/api.html#pytket.extensions.cirq.CirqDensityMatrixSampleBackend)
Backend for Cirq density matrix simulator sampling.

[CirqDensityMatrixSimBackend](https://cqcl.github.io/pytket-extensions/api/cirq/api.html#pytket.extensions.cirq.CirqDensityMatrixSimBackend)
Backend for Cirq density matrix simulator density_matrix return.

### Clifford simulator

[CirqCliffordSampleBackend](https://cqcl.github.io/pytket-extensions/api/cirq/api.html#pytket.extensions.cirq.CirqCliffordSampleBackend)
Backend for Cirq Clifford simulator sampling.

[CirqCliffordSimBackend](https://cqcl.github.io/pytket-extensions/api/cirq/api.html#pytket.extensions.cirq.CirqCliffordSimBackend)
Backend for Cirq Clifford simulator state return.

[SimplexBackend](https://cqcl.github.io/pytket-extensions/api/pysimplex/api.html#pytket.extensions.pysimplex.SimplexBackend)
Backend for simulating Clifford circuits using pysimplex.

[StimBackend](https://cqcl.github.io/pytket-extensions/api/stim/api.html#pytket.extensions.stim.StimBackend)
Backend for simulating Clifford circuits using Stim.

### Other

[AerBackend](https://cqcl.github.io/pytket-extensions/api/qiskit/api.html#pytket.extensions.qiskit.AerBackend)
Backend for running simulations on the Qiskit Aer QASM simulator.

[tketBackendEngine](https://cqcl.github.io/pytket-extensions/api/projectq/api.html#pytket.extensions.projectq.tketBackendEngine)
A projectq backend designed to translate from projectq commands to tket Circuits.

[QulacsBackend](https://cqcl.github.io/pytket-extensions/api/qulacs/api.html#pytket.extensions.qulacs.QulacsBackend)
Backend for running simulations on the Qulacs simulator.

[QsharpSimulatorBackend](https://cqcl.github.io/pytket-extensions/api/qsharp/api.html#pytket.extensions.qsharp.QsharpSimulatorBackend)
Backend for simulating a circuit using the QDK.

[QsharpToffoliSimulatorBackend](https://cqcl.github.io/pytket-extensions/api/qsharp/api.html#pytket.extensions.qsharp.QsharpToffoliSimulatorBackend)
Backend for simulating a Toffoli circuit using the QDK.

[QsharpEstimatorBackend](https://cqcl.github.io/pytket-extensions/api/qsharp/api.html#pytket.extensions.qsharp.QsharpEstimatorBackend)
Backend for estimating resources of a circuit using the QDK.

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
