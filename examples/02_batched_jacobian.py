from pathlib import Path
import torch
import bard

script_dir = Path(__file__).parent
urdf_path = script_dir / "example_robots/xarm_description/urdf/xarm7.urdf"


def main():
    """
    An example of computing the Jacobian for a batch of configurations.
    """
    # Load a robot from a URDF file
    model = bard.build_model_from_urdf(urdf_path).to(dtype=torch.float32, device="cpu")

    # Create data workspace with the desired batch size
    batch_size = 1000
    data = bard.create_data(model, max_batch_size=batch_size)

    # 1. Define a large batch of random joint configurations
    q_batch = torch.rand(batch_size, model.n_joints) * torch.pi

    # 2. Select the end-effector frame
    ee_frame_name = model.get_frame_names(exclude_fixed=True)[-1]
    ee_frame_idx = model.get_frame_id(ee_frame_name)
    print(
        f"Calculating Jacobian for frame '{ee_frame_name}' with a batch of {batch_size} configurations."
    )

    # 3. Update kinematics for the batch
    bard.update_kinematics(model, data, q_batch)

    # 4. Compute the Jacobian in the world frame for the entire batch
    J_world_batch = bard.jacobian(model, data, ee_frame_idx, reference_frame="world")

    # 5. Compute the Jacobian in the local (body) frame for the entire batch
    J_local_batch = bard.jacobian(model, data, ee_frame_idx, reference_frame="local")

    print("\n--- Output Shapes ---")
    print(f"Input q shape:          {q_batch.shape}")
    print(f"World Jacobian J_w shape: {J_world_batch.shape}")
    print(f"Local Jacobian J_l shape: {J_local_batch.shape}")
    print("\nNote: The shape is (Batch, 6, Num_Velocities), demonstrating batched output.")


if __name__ == "__main__":
    main()
