Changelog
=========

v0.3
----

**Unified RobotDynamics API**

* Added ``RobotDynamics`` class as the primary interface for all kinematics and
  dynamics computations. A single ``update_kinematics()`` call computes shared
  quantities (transforms, spatial adjoints, joint subspace, velocities) once,
  eliminating redundant tree traversals when multiple algorithms are used in
  the same control step.

* Added ``KinematicsState`` dataclass for holding cached kinematic quantities.

* Deprecated ``ForwardKinematics``, ``SpatialAcceleration``, ``Jacobian``,
  ``RNEA``, and ``CRBA`` classes. These are retained as thin wrappers around
  ``RobotDynamics`` for backward compatibility.

* Removed Inverse Kinematics from the roadmap.

v0.2
----

* Benchmarks and performance documentation.
* Fixed ``lxml`` dependency version.
* Removed Python 3.14 support from CI.

v0.1
----

* Initial release with batched Forward Kinematics, Jacobian, RNEA, CRBA,
  and Spatial Acceleration.
* URDF parsing support.
* ``torch.compile`` compatibility.
* Fixed-base and floating-base robot support.
