Quick Start
===========

This guide provides examples to get you started with ``bard``, from basic setup to performance optimization.

A Simple Example
----------------

Here is a complete example of performing batched forward kinematics, Jacobian, and inverse dynamics for a simple 2-link robot defined in a string.

.. code-block:: python

   import torch
   import bard

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

   # 1. Build the model and create a data workspace
   model = bard.build_model_from_urdf(urdf_string).to(dtype=torch.float64)
   data = bard.create_data(model, max_batch_size=100)

   # 2. Define a batch of joint configurations (N=100)
   N = 100
   q = torch.rand(N, model.n_joints, dtype=torch.float64) * torch.pi

   # 3. Get the end-effector frame index
   ee_frame_id = model.get_frame_id("link3")

   # 4. Standalone forward kinematics (path-only traversal)
   T_world_to_ee = bard.forward_kinematics(model, data, ee_frame_id, q=q)
   ee_positions = T_world_to_ee[:, :3, 3]

   print(f"Batch size: {N}")
   print(f"End-effector position shape: {ee_positions.shape}")

Cached Workflow
---------------

When you need multiple algorithms in a single control step (e.g., FK + Jacobian + RNEA), use ``update_kinematics()`` to traverse the kinematic tree once, then reuse the cached data for all subsequent calls.

.. code-block:: python

   import torch
   import bard

   model = bard.build_model_from_urdf("path/to/robot.urdf")
   model.to(dtype=torch.float32, device="cuda")  # Move to GPU
   data = bard.create_data(model, max_batch_size=4096)

   ee_id = model.get_frame_id("end_effector_link")

   # In your control / RL training loop:
   q = ...    # (4096, nq)
   qd = ...   # (4096, nv)
   qdd = ...  # (4096, nv)

   # 1. Single tree traversal -- caches transforms, velocities, etc.
   bard.update_kinematics(model, data, q, qd)

   # 2. All algorithms reuse cached data (no redundant computation)
   T_eef = bard.forward_kinematics(model, data, ee_id)     # O(1) lookup
   J = bard.jacobian(model, data, ee_id, reference_frame="world")
   tau = bard.rnea(model, data, qdd)
   M = bard.crba(model, data)

.. note::

   If you only need position-level quantities (FK, Jacobian, CRBA), you can omit
   ``qd`` when calling ``update_kinematics()``. Velocity-level quantities are
   required for RNEA and spatial acceleration.

Loading a URDF from a File
--------------------------

Instead of defining a URDF in a string, you can load it directly from a file path.

.. code-block:: python

   from pathlib import Path
   import torch
   import bard

   urdf_path = Path("path/to/your/robot.urdf")

   # Build the model by passing the file path
   model = bard.build_model_from_urdf(urdf_path)
   data = bard.create_data(model, max_batch_size=1024)

   print(f"Loaded robot with {model.n_joints} joint(s) from '{urdf_path}'.")

Floating-Base Robots
--------------------

For floating-base robots (e.g., quadrupeds, humanoids), pass ``floating_base=True``
when building the model. The generalized coordinates include a 7-element base pose
``[tx, ty, tz, qw, qx, qy, qz]`` prepended to the joint angles.

.. code-block:: python

   model = bard.build_model_from_urdf("go2.urdf", floating_base=True)
   data = bard.create_data(model, max_batch_size=4096)

   # q: (B, 7 + n_joints) -- base pose + joint angles
   # qd: (B, 6 + n_joints) -- base spatial velocity + joint velocities

Autograd Differentiation
------------------------

bard supports PyTorch autograd for key gradient computations. Simply pass tensors
with ``requires_grad=True`` and bard automatically switches to a functional code
path that is compatible with the gradient tape.

.. code-block:: python

   import torch
   import bard

   model = bard.build_model_from_urdf("go2.urdf", floating_base=True)
   model.to(dtype=torch.float64)
   data = bard.create_data(model, max_batch_size=4)

   # d(M)/d(q): gradient of mass matrix w.r.t. joint configuration
   q = torch.randn(1, model.nq, dtype=torch.float64, requires_grad=True)
   bard.update_kinematics(model, data, q)
   M = bard.crba(model, data)
   M.sum().backward()
   print(f"dM/dq shape: {q.grad.shape}")  # (1, nq)

   # d(qdd)/d(tau): gradient of forward dynamics w.r.t. applied torques
   q2 = torch.randn(1, model.nq, dtype=torch.float64)
   qd = torch.randn(1, model.nv, dtype=torch.float64)
   tau = torch.randn(1, model.nv, dtype=torch.float64, requires_grad=True)
   bard.update_kinematics(model, data, q2, qd)
   qdd = bard.aba(model, data, tau)
   qdd.sum().backward()
   print(f"d(qdd)/d(tau) shape: {tau.grad.shape}")  # (1, nv)

Supported autograd paths:

- **CRBA:** ``d(M)/d(q)`` via ``update_kinematics(q) -> crba()``
- **ABA:** ``d(qdd)/d(tau)`` via ``aba(tau)``
- **RNEA:** ``d(tau)/d(qdd)`` via ``rnea(qdd)``

.. note::

   Gradients through ABA/RNEA with respect to ``q`` are not yet supported.
   When autograd is active, algorithms use fresh tensor allocations instead of
   pre-allocated buffers, which is slower but necessary for correct gradient
   computation.

Enabling Compilation for Maximum Performance
---------------------------------------------

For performance-critical applications like reinforcement learning or trajectory optimization, you can JIT-compile the core computations using ``torch.compile``. This can significantly speed up execution after an initial warm-up phase.

.. code-block:: python

   model = bard.build_model_from_urdf("robot.urdf")
   model.enable_compilation(True)
   data = bard.create_data(model, max_batch_size=4096)

   # The first call will have some overhead for compilation...
   bard.update_kinematics(model, data, q, qd)
   tau = bard.rnea(model, data, qdd)

   # ...subsequent calls will be much faster.
   bard.update_kinematics(model, data, q2, qd2)
   tau = bard.rnea(model, data, qdd2)
