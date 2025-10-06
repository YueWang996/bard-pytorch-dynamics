Quick Start
===========

Here is a simple example of performing batched forward kinematics and Jacobian
calculations for a 2-link robot.

.. code-block:: python

   import torch
   from bard.parsers.urdf import build_chain_from_urdf
   from bard.core.forward_kinematics import calc_forward_kinematics
   from bard.core.jacobian import calc_jacobian

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

   # 1. Build the kinematic chain
   chain = build_chain_from_urdf(urdf_string).to(dtype=torch.float64)

   # 2. Define a batch of joint configurations (N=100)
   N = 100
   q = torch.rand(N, chain.n_joints, dtype=torch.float64) * torch.pi

   # 3. Get the index of the end-effector frame
   ee_frame = "link3"
   ee_idx = chain.get_frame_indices(ee_frame).item()

   # 4. Perform batched forward kinematics
   transforms = calc_forward_kinematics(chain, q, ee_idx)
   ee_positions = transforms.get_matrix()[:, :3, 3]

   # 5. Perform batched Jacobian calculation
   J = calc_jacobian(chain, q, ee_idx, reference_frame="world")

   print(f"Batch size: {N}")
   print(f"End-effector position shape: {ee_positions.shape}")
   print(f"Jacobian shape: {J.shape}")