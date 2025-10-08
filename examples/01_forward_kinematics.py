import torch
from bard.parsers.urdf import build_chain_from_urdf
from bard.core.kinematics import ForwardKinematics


def main():
    """
    A simple example of computing forward kinematics for a fixed-base robot.
    """
    # A simple 2-link robot URDF defined as a string
    urdf_string = """
    <robot name="simple_robot">
        <link name="link1"/>
        <link name="link2"/>
        <link name="end_effector_link"/>
        <joint name="joint1" type="revolute">
            <parent link="link1"/> <child link="link2"/>
            <origin xyz="0 0 1"/> <axis xyz="0 0 1"/>
        </joint>
        <joint name="joint2" type="revolute">
            <parent link="link2"/> <child link="end_effector_link"/>
            <origin xyz="0 0 1"/> <axis xyz="0 1 0"/>
        </joint>
    </robot>
    """

    # 1. Build the kinematic chain from the URDF string
    chain = build_chain_from_urdf(urdf_string).to(dtype=torch.float64, device="cpu")
    print(f"Robot loaded with {chain.n_joints} joints: {chain.get_joint_parameter_names()}")

    # Instantiate the ForwardKinematics class once
    fk = ForwardKinematics(chain, max_batch_size=1)

    # 2. Define a joint configuration
    q = torch.tensor([torch.pi / 2, torch.pi / 2], dtype=torch.float64)

    # 3. Get the integer index of the end-effector frame
    ee_frame_name = "end_effector_link"
    ee_frame_idx = chain.get_frame_indices(ee_frame_name).item()
    print(f"Calculating FK for frame '{ee_frame_name}' (index {ee_frame_idx})")

    # 4. Compute the forward kinematics using the class method
    # Note: The class-based API expects a batched input, so we add a batch dimension with unsqueeze(0)
    transforms_batch = fk.calc(q.unsqueeze(0), ee_frame_idx)

    # 5. Extract and print the 4x4 homogeneous transformation matrix
    pose_matrix = transforms_batch[0]  # Get the first (and only) matrix from the batch
    position = pose_matrix[:3, 3]

    print("\nEnd-effector pose matrix at q = [pi/2, pi/2]:")
    print(pose_matrix.numpy())
    print(f"\nEnd-effector XYZ position: {position.numpy()}")


if __name__ == "__main__":
    main()
