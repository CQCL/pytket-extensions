Changelog
~~~~~~~~~

0.19.1 (June 2022)
------------------

* Fixes to qubit index handling.

0.19.0 (May 2022)
-----------------

* Add optional "aws_session" parameter to the available_devices class method of BracketBackend.
* Respect measurement operations in submitted circuits. (Previously these were
  not allowed and measurements were automatically added to all qubits.)
* Updated pytket version requirement to 1.2.

0.18.0 (April 2022)
-------------------

* Allow user to specify different local backends.
* Fix dimension mismatch for measurements for Braket QPU results.
* Updated pytket version requirement to 1.1.

0.17.0 (March 2022)
-------------------

* Add optional "region" argument for initializing BraketBackend.
* Update requirements for amazon-braket-sdk and amazon-braket-schemas.
* Add support for oqc provider.
* Correct readout errors reported for Rigetti devices.
* Add support for density-matrix simulators.
* Updated pytket version requirement to 1.0.

0.16.0 (February 2022)
----------------------

* Updated pytket version requirement to 0.19.
* Drop support for Python 3.7; add support for 3.10.
* Add support for CV gates.

0.15.0 (January 2022)
---------------------

* Updated pytket version requirement to 0.18.

0.14.0 (November 2021)
----------------------

* Updated pytket version requirement to 0.17.

0.13.0 (October 2021)
---------------------

* Updated pytket version requirement to 0.16.

0.12.0 (September 2021)
-----------------------

* Updated pytket version requirement to 0.15.

0.11.0 (September 2021)
-----------------------

* Updated pytket version requirement to 0.14.

0.10.0 (July 2021)
------------------

* Updated pytket version requirement to 0.13.

0.9.0 (June 2021)
-----------------

* Updated pytket version requirement to 0.12.

0.8.2 (June 2021)
-----------------

* Make cache handling more robust.

0.8.1 (June 2021)
-----------------

* Make BracketBackend.device a property, in accordance with definition in base class.

0.8.0 (May 2021)
----------------

* Updated pytket version requirement to 0.11.

0.7.0 (May 2021)
----------------

* Contextual optimisation added to default compilation passes (except at optimisation level 0).
