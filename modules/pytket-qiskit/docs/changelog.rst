Changelog
~~~~~~~~~

0.28.1 (August 2022)
--------------------

* Fix error in aer backend creation

0.28.0 (August 2022)
--------------------

* Improve result retrieval speed of ``AerUnitaryBackend`` and ``AerStateBackend``.
* Update qiskit version to 0.37.
* Updated pytket version requirement to 1.5.

0.27.0 (July 2022)
------------------

* Updated pytket version requirement to 1.4.

0.26.0 (June 2022)
------------------

* Updated pytket version requirement to 1.3.

0.25.0 (May 2022)
-----------------

* Updated pytket version requirement to 1.2.

0.24.0 (April 2022)
-------------------

* Fix two-qubit unitary conversions.
* Update qiskit version to 0.36.
* Updated pytket version requirement to 1.1.

0.23.0 (March 2022)
-------------------

* Removed ``characterisation`` property of backends. (Use `backend_info`
  instead.)
* Updated pytket version requirement to 1.0.

0.22.2 (February 2022)
----------------------

* Fixed :py:meth:`IBMQEmulatorBackend.rebase_pass`.

0.22.1 (February 2022)
----------------------

* Added :py:meth:`IBMQEmulatorBackend.rebase_pass`.

0.22.0 (February 2022)
----------------------

* Qiskit version updated to 0.34.
* Updated pytket version requirement to 0.19.
* Drop support for Python 3.7; add support for 3.10.

0.21.0 (January 2022)
---------------------

* Qiskit version updated to 0.33.
* Updated pytket version requirement to 0.18.

0.20.0 (November 2021)
----------------------

* Qiskit version updated to 0.32.
* Updated pytket version requirement to 0.17.

0.19.0 (October 2021)
---------------------

* Qiskit version updated to 0.31.
* Removed deprecated :py:meth:`AerUnitaryBackend.get_unitary`. Use
  :py:meth:`AerUnitaryBackend.run_circuit` and
  :py:meth:`pytket.backends.backendresult.BackendResult.get_unitary` instead.
* Updated pytket version requirement to 0.16.

0.18.0 (September 2021)
-----------------------

* Qiskit version updated to 0.30.
* Updated pytket version requirement to 0.15.

0.17.0 (September 2021)
-----------------------

* Updated pytket version requirement to 0.14.

0.16.1 (July 2021)
------------------

* Fix slow/high memory use :py:meth:`AerBackend.get_operator_expectation_value`

0.16.0 (July 2021)
------------------

* Qiskit version updated to 0.28.
* Use provider API client to check job status without retrieving job in IBMQBackend.
* Updated pytket version requirement to 0.13.

0.15.1 (July 2021)
------------------

* Fixed bug in backends when n_shots argument was passed as list.

0.15.0 (June 2021)
------------------

* Updated pytket version requirement to 0.12.

0.14.0 (unreleased)
-------------------

* Qiskit version updated to 0.27.

0.13.0 (May 2021)
-----------------

* Updated pytket version requirement to 0.11.

0.12.0 (unreleased)
-------------------

* Qiskit version updated to 0.26.
* Code rewrites to avoid use of deprecated qiskit methods.
* Restriction to hermitian operators for expectation values in `AerBackend`.

0.11.0 (May 2021)
-----------------

* Contextual optimisation added to default compilation passes (except at optimisation level 0).
* Support for symbolic parameters in rebase pass.
* Correct phase when rebasing.
* Ability to preserve UUIDs of qiskit symbolic parameters when converting.
* Correction to error message.

0.10.0 (April 2021)
-------------------

* Support for symbolic phase in converters.
