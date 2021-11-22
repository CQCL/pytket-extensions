Changelog
~~~~~~~~~

0.18.0 (November 2021)
----------------------

* Updated pytket version requirement to 0.17.

0.17.0 (October 2021)
---------------------

* Updated pytket version requirement to 0.16.
* Renamed ``HoneywellBackend.available_devices`` to ``_available_devices`` so as
  not to conflict with abstract ``Backend`` method.

0.16.0 (September 2021)
-----------------------

* Updated pytket version requirement to 0.15.

0.15.0 (September 2021)
-----------------------

* Updated pytket version requirement to 0.14.

0.14.0 (August 2021)
--------------------

* Support new Honeywell simulator options in :py:class:`HoneywellBackend`:
  "simulator" for simulator type, and "noisy_simulation" to toggle simulations
  with and without error models.
* Device name no longer optional on :py:class:`HoneywellBackend` construction.

0.13.0 (July 2021)
------------------

* Updated pytket version requirement to 0.13.

0.12.0 (June 2021)
------------------

* Updated pytket version requirement to 0.12.

0.11.0 (May 2021)
-----------------

* Updated pytket version requirement to 0.11.

0.10.0 (May 2021)
-----------------

* Contextual optimisation added to default compilation passes (except at optimisation level 0).
