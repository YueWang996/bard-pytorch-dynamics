from pathlib import Path
import torch
import bard

script_dir = Path(__file__).parent
urdf_path = script_dir / "simple_arm.urdf"


def main():
    """
    A simple example of computing forward kinematics for a fixed-base robot.
    """

    # 1. Build the model from the URDF file
    model = bard.build_model_from_urdf(urdf_path).to(dtype=torch.float64, device="cpu")
    data = bard.create_data(model, max_batch_size=1)
    print(f"Robot loaded with {model.n_joints} joints: {model.get_joint_parameter_names()}")

    # 2. Define a joint configuration
    q = torch.tensor([torch.pi / 2, torch.pi / 2], dtype=torch.float64)

    # 3. Get the integer index of the end-effector frame
    ee_frame_name = "end_effector_link"
    ee_frame_idx = model.get_frame_id(ee_frame_name)
    print(f"Calculating FK for frame '{ee_frame_name}' (index {ee_frame_idx})")

    # 4. Compute the forward kinematics
    # Note: The API expects a batched input, so we add a batch dimension with unsqueeze(0)
    transforms_batch = bard.forward_kinematics(model, data, ee_frame_idx, q=q.unsqueeze(0))

    # 5. Extract and print the 4x4 homogeneous transformation matrix
    pose_matrix = transforms_batch[0]  # Get the first (and only) matrix from the batch
    position = pose_matrix[:3, 3]

    print("\nEnd-effector pose matrix at q = [pi/2, pi/2]:")
    print(pose_matrix.numpy())
    print(f"\nEnd-effector XYZ position: {position.numpy()}")


if __name__ == "__main__":
    main()
