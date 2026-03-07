Quick Start
===========

This guide provides examples to get you started with ``bard``, from basic setup to performance optimization.

A Simple Example
----------------

Here is a complete example of performing batched forward kinematics, Jacobian, and inverse dynamics for a simple 2-link robot defined in a string.

.. code-block:: python

   import torch
   from bard import build_chain_from_urdf, RobotDynamics

   # A simple 2-link robot URDF
   urdf_string = """
   <robot name="simple_robot">
       <link name="link1"/>
       <link name="link2"/>
       <link name="link3"/>
       <joint name="joint1" type="revolute">
           <parent link="link1"/>
           <child link="link2"/>
           <origin xyz="0 0 1"/>
           <axis xyz="0 0 1"/>
       </joint>
       <joint name="joint2" type="revolute">
           <parent link="link2"/>
           <child link="link3"/>
           <origin xyz="0 0 1"/>
           <axis xyz="0 1 0"/>
       </joint>
   </robot>
   """

   # 1. Build the kinematic chain and create the dynamics interface
   chain = build_chain_from_urdf(urdf_string).to(dtype=torch.float64)
   rd = RobotDynamics(chain, max_batch_size=100)

   # 2. Define a batch of joint configurations (N=100)
   N = 100
   q = torch.rand(N, chain.n_joints, dtype=torch.float64) * torch.pi

   # 3. Get the end-effector frame index
   ee_frame_id = chain.get_frame_id("link3")

   # 4. Standalone forward kinematics (path-only traversal)
   T_world_to_ee = rd.fk(q, ee_frame_id)
   ee_positions = T_world_to_ee[:, :3, 3]

   print(f"Batch size: {N}")
   print(f"End-effector position shape: {ee_positions.shape}")

Cached Workflow
---------------

When you need multiple algorithms in a single control step (e.g., FK + Jacobian + RNEA), use ``update_kinematics()`` to traverse the kinematic tree once, then reuse the cached state for all subsequent calls.

.. code-block:: python

   import torch
   from bard import build_chain_from_urdf, RobotDynamics

   chain = build_chain_from_urdf("path/to/robot.urdf")
   chain.to(dtype=torch.float32, device="cuda")  # Move to GPU
   rd = RobotDynamics(chain, max_batch_size=4096)

   ee_id = chain.get_frame_id("end_effector_link")

   # In your control / RL training loop:
   q = ...    # (4096, nq)
   qd = ...   # (4096, nv)
   qdd = ...  # (4096, nv)

   # 1. Single tree traversal -- caches transforms, velocities, etc.
   state = rd.update_kinematics(q, qd)

   # 2. All algorithms reuse cached state (no redundant computation)
   T_eef = rd.forward_kinematics(ee_id, state)     # O(1) lookup
   J = rd.jacobian(ee_id, state, reference_frame="world")
   tau = rd.rnea(qdd, state)
   M = rd.crba(state)

.. note::

   If you only need position-level quantities (FK, Jacobian, CRBA), you can omit
   ``qd`` when calling ``update_kinematics(q)``. Velocity-level quantities are
   required for RNEA and spatial acceleration.

Loading a URDF from a File
--------------------------

Instead of defining a URDF in a string, you can load it directly from a file path.

.. code-block:: python

   from pathlib import Path
   import torch
   from bard import build_chain_from_urdf, RobotDynamics

   urdf_path = Path("path/to/your/robot.urdf")

   # Build the chain by passing the file path
   chain = build_chain_from_urdf(urdf_path)
   rd = RobotDynamics(chain, max_batch_size=1024)

   print(f"Loaded robot with {chain.n_joints} joint(s) from '{urdf_path}'.")

Floating-Base Robots
--------------------

For floating-base robots (e.g., quadrupeds, humanoids), pass ``floating_base=True``
when building the chain. The generalized coordinates include a 7-element base pose
``[tx, ty, tz, qw, qx, qy, qz]`` prepended to the joint angles.

.. code-block:: python

   chain = build_chain_from_urdf("go2.urdf", floating_base=True)
   rd = RobotDynamics(chain, max_batch_size=4096)

   # q: (B, 7 + n_joints) -- base pose + joint angles
   # qd: (B, 6 + n_joints) -- base spatial velocity + joint velocities

Enabling Compilation for Maximum Performance
---------------------------------------------

For performance-critical applications like reinforcement learning or trajectory optimization, you can JIT-compile the core computations using ``torch.compile``. This can significantly speed up execution after an initial warm-up phase.

.. code-block:: python

   rd = RobotDynamics(chain, max_batch_size=4096, compile_enabled=True)

   # Or enable compilation on an existing instance
   rd.enable_compilation(True)

   # The first call will have some overhead for compilation...
   state = rd.update_kinematics(q, qd)
   tau = rd.rnea(qdd, state)

   # ...subsequent calls will be much faster.
   state = rd.update_kinematics(q2, qd2)
   tau = rd.rnea(qdd2, state)
