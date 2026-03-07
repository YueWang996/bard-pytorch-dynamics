API Reference
=============

This page provides the auto-generated API documentation for the core classes in ``bard``.

Robot Dynamics
--------------

.. automodule:: bard.core.robot_dynamics
   :members: RobotDynamics

Kinematics State
----------------

.. automodule:: bard.core.state
   :members: KinematicsState

Robot Chain
-----------

.. automodule:: bard.core.chain
   :members: Chain
   :exclude-members: nq, nv

URDF Parsing
------------

.. automodule:: bard.parsers.urdf
   :members:

Data Structures
---------------

.. automodule:: bard.structures.frame
   :members: Frame

.. automodule:: bard.structures.joint
   :members: Joint

.. automodule:: bard.structures.link
   :members: Link

Deprecated Classes
------------------

The following classes are deprecated. Use :class:`~bard.core.robot_dynamics.RobotDynamics` instead.

.. automodule:: bard.core.kinematics
   :members: ForwardKinematics, SpatialAcceleration

.. automodule:: bard.core.jacobian
   :members: Jacobian

.. automodule:: bard.core.dynamics
   :members: RNEA, CRBA