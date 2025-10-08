Quick Start
===========

This guide provides examples to get you started with `bard`, from basic setup to performance optimization.

A Simple Example
----------------

Here is a complete example of performing batched forward kinematics and Jacobian calculations for a simple 2-link robot defined in a string.

.. code-block:: python

   import torch
   from bard.parsers.urdf import build_chain_from_urdf
   from bard.core.kinematics import ForwardKinematics
   from bard.core.jacobian import Jacobian

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

   # 1. Build the kinematic chain from the string
   chain = build_chain_from_urdf(urdf_string).to(dtype=torch.float64)
   
   # 2. Instantiate the algorithm classes once
   fk = ForwardKinematics(chain, max_batch_size=100)
   jac = Jacobian(chain, max_batch_size=100)

   # 3. Define a batch of joint configurations (N=100)
   N = 100
   q = torch.rand(N, chain.n_joints, dtype=torch.float64) * torch.pi

   # 4. Get the index of the end-effector frame
   ee_frame_name = "link3"
   ee_frame_id = chain.get_frame_indices(ee_frame_name).item()

   # 5. Perform batched forward kinematics
   T_world_to_ee = fk.calc(q, ee_frame_id)
   ee_positions = T_world_to_ee[:, :3, 3]

   # 6. Perform batched Jacobian calculation
   J = jac.calc(q, ee_frame_id, reference_frame="world")

   print(f"Batch size: {N}")
   print(f"End-effector position shape: {ee_positions.shape}")
   print(f"Jacobian shape: {J.shape}")

Loading a URDF from a File
--------------------------

Instead of defining a URDF in a string, you can load it directly from a file path.

.. code-block:: python

   from pathlib import Path
   import torch
   from bard.parsers.urdf import build_chain_from_urdf
   from bard.core.jacobian import Jacobian

   script_dir = Path(__file__).parent
   urdf_path = script_dir / "path/to/your/robot.urdf"

   # Build the chain by passing the file path
   chain_from_file = build_chain_from_urdf(urdf_path)

   print(f"Loaded robot with {chain_from_file.n_joints} joint(s) from '{urdf_path}'.")

Enabling Compilation for Maximum Performance
---------------------------------------------

For performance-critical applications like reinforcement learning or trajectory optimization, you can JIT-compile the core computations using `torch.compile`. This can significantly speed up execution after an initial warm-up phase.

You can enable compilation at instantiation or afterward.

.. code-block:: python

   # Option 1: Enable compilation when creating the object
   # This is the recommended approach.
   rnea_compiled = RNEA(chain, max_batch_size=100, compile_enabled=True)

   # Option 2: Enable compilation on an existing object
   crba = CRBA(chain, max_batch_size=100)
   crba.enable_compilation(True) 

   # Now, calls to .calc() will use the compiled version
   q = torch.rand(100, chain.n_joints, dtype=torch.float64)
   qd = torch.rand(100, chain.n_joints, dtype=torch.float64)
   qdd = torch.rand(100, chain.n_joints, dtype=torch.float64)

   # The first call will have some overhead for compilation...
   print("Performing first call (with compilation warm-up)...")
   tau = rnea_compiled.calc(q, qd, qdd)

   # ...subsequent calls will be much faster.
   print("Performing second call (cached)...")
   tau = rnea_compiled.calc(q, qd, qdd)

   print(f"Computed torques with shape: {tau.shape}")