Changelog
~~~~~~~~~

0.26.0 (unreleased)
-------------------

* Updated cirq version requirement to 1.x.

0.25.0 (July 2022)
------------------

* Updated pytket version requirement to 1.4.
* Updated cirq version requirement to 0.15.
* Changed `process_characterisation()` so that it expects a `GridDevice` instead
  of a `SerializableDevice`.

0.24.0 (June 2022)
------------------

* `cirq_to_tk()` and `tk_to_cirq()` now properly handle circuits with `NamedQubit`.
* Updated pytket version requirement to 1.3.

0.23.0 (May 2022)
-----------------

* Updated pytket version requirement to 1.2.

0.22.0 (April 2022)
-------------------

* Updated cirq version requirement to 0.14.
* Changed `process_characterisation()` so that it expects a `SerializableDevice`
  instead of an `XmonDevice`.
* Updated pytket version requirement to 1.1.

0.21.0 (March 2022)
-------------------

* Updated pytket version requirement to 1.0.

0.20.0 (February 2022)
----------------------

* Updated pytket version requirement to 0.19.
* Drop support for Python 3.7; add support for 3.10.

0.19.0 (January 2022)
---------------------

* Updated pytket version requirement to 0.18.

0.18.0 (November 2021)
----------------------

* Updated pytket version requirement to 0.17.

0.17.0 (October 2021)
---------------------

* Updated cirq version requirement to 0.13.
* Updated pytket version requirement to 0.16.

0.16.0 (September 2021)
-----------------------

* Updated cirq version requirement to 0.12.
* Updated pytket version requirement to 0.15.

0.15.0 (September 2021)
-----------------------

* Updated pytket version requirement to 0.14.

0.14.0 (July 2021)
------------------

* Updated pytket version requirement to 0.13.

0.13.0 (June 2021)
------------------

* Updated pytket version requirement to 0.12.

0.12.0 (May 2021)
-----------------

* Updated pytket version requirement to 0.11.

0.11.0 (unreleased)
-------------------

* Required cirq version updated to 0.11.
* Add CH gate to allowed gates for cirq to pytket conversion methods

0.10.1 (May 2021)
-----------------

* Pinning cirq version to 0.10.

0.10.0 (April 2021)
-------------------

* Improved error handling.
* Inclusion of unused qubits in state and density-matrix simulators.
