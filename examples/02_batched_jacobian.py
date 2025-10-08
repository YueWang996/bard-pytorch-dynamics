from pathlib import Path
import torch
from bard.parsers.urdf import build_chain_from_urdf
from bard.core.jacobian import Jacobian

script_dir = Path(__file__).parent
urdf_path = script_dir / "example_robots/xarm_description/urdf/xarm7.urdf"

def main():
    """
    An example of computing the Jacobian for a batch of configurations.
    """
    # Load a robot from a URDF file
    try:
        with open(urdf_path, "rb") as f:
            urdf_string = f.read()
    except FileNotFoundError:
        print(f"Error: {urdf_path} not found. Please provide a valid path.")
        return

    chain = build_chain_from_urdf(urdf_string).to(dtype=torch.float32, device="cpu")
    
    # Instantiate the Jacobian class once, specifying the max batch size
    batch_size = 1000
    jac = Jacobian(chain, max_batch_size=batch_size)

    # 1. Define a large batch of random joint configurations
    q_batch = torch.rand(batch_size, chain.n_joints) * torch.pi

    # 2. Select the end-effector frame
    ee_frame_name = chain.get_frame_names(exclude_fixed=True)[-1]
    ee_frame_idx = chain.get_frame_indices(ee_frame_name).item()
    print(f"Calculating Jacobian for frame '{ee_frame_name}' with a batch of {batch_size} configurations.")

    # 3. Compute the Jacobian in the world frame for the entire batch
    J_world_batch = jac.calc(q_batch, ee_frame_idx, reference_frame="world")

    # 4. Compute the Jacobian in the local (body) frame for the entire batch
    J_local_batch = jac.calc(q_batch, ee_frame_idx, reference_frame="local")

    print("\n--- Output Shapes ---")
    print(f"Input q shape:          {q_batch.shape}")
    print(f"World Jacobian J_w shape: {J_world_batch.shape}")
    print(f"Local Jacobian J_l shape: {J_local_batch.shape}")
    print("\nNote: The shape is (Batch, 6, Num_Velocities), demonstrating batched output.")

if __name__ == "__main__":
    main()