# pytket-qsharp

[Pytket](https://cqcl.github.io/pytket) is a Python module for interfacing
with CQC tket, a set of quantum programming tools.

[Azure Quantum](https://azure.microsoft.com/en-gb/services/quantum/) is a portal for accessing
quantum computers via Microsoft Azure.

Microsoft's [QDK](https://docs.microsoft.com/en-us/quantum/install-guide) is a
language and associated toolkit for quantum programming.

`pytket-qsharp` is an extension to `pytket` that allows `pytket` circuits to be
executed on remote devices and simulators via Azure Quantum,
as well as local simulators and resource estimators from the Microsoft QDK.

## Getting started

`pytket-qsharp` is available for Python 3.7, 3.8 and 3.9, on Linux, MacOS and Windows. To
install, run:

```pip install pytket-qsharp```

In order to use `pytket-qsharp` you will first need to install the `dotnet` SDK
(3.1) and the `iqsharp` tool. On some Linux systems it is also necessary to
modify your `PATH`:

1. See [this page](https://dotnet.microsoft.com/download/dotnet-core/3.1) for
instructions on installing the SDK on your operating system.

2. On Linux, ensure that the `dotnet` tools directory is on your path. Typically
this will be `~/.dotnet/tools`.

3. Run `dotnet tool install -g Microsoft.Quantum.IQSharp`.

4. Run `dotnet iqsharp install --user`.


Alternatively, you can set up an environment with all the required packages using conda:

```
conda create -n qsharp-env -c quantum-engineering qsharp notebook

conda activate qsharp-env
```
## Backends provided in this module

This module provides four
[backends](https://cqcl.github.io/pytket/build/html/backends.html), all deriving
from the `pytket` `Backend` class:

* `AzureBackend`, for executing pytket circuits on targets the user has access to on Azure Quantum;

* `QsharpSimulatorBackend`, for simulating a general pure-quantum circuit using
the QDK;

* `QsharpToffoliSimulatorBackend`, for simulating a Toffoli circuit using the
QDK;

* `QsharpEstimatorBackend`, for estimating various quantum resources of a
circuit using the QDK. This provides a `get_resources` method, which returns a
dictionary.
