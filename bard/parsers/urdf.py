from .urdf_parser_py.urdf import URDF, Mesh, Cylinder, Box, Sphere
# from bard.structures import frame
from bard.structures import Frame, Joint, Link, Visual
from bard.core import chain
import torch
import pytorch_kinematics.transforms as tf


JOINT_TYPE_MAP = {
    'revolute': 'revolute',
    'continuous': 'revolute',
    'prismatic': 'prismatic',
    'fixed': 'fixed'
}


def _convert_transform(origin):
    """Convert URDF origin to Transform3d."""
    if origin is None:
        return tf.Transform3d()
    rpy = torch.tensor(origin.rpy, dtype=torch.float32, device="cpu")
    return tf.Transform3d(rot=tf.quaternion_from_euler(rpy, "sxyz"), pos=origin.xyz)


def _convert_inertial(inertial):
    """Convert URDF inertial to (transform, mass, inertia_matrix) tuple."""
    if inertial is None:
        return None
    origin = _convert_transform(inertial.origin)
    mass = inertial.mass
    inertia = torch.tensor(
        inertial.inertia.to_matrix(),
        dtype=torch.float32,
        device="cpu"
    )
    return (origin, mass, inertia)


def _convert_visual(visual):
    """Convert URDF visual to Visual frame."""
    if visual is None or visual.geometry is None:
        return Visual()
    
    v_tf = _convert_transform(visual.origin)
    
    if isinstance(visual.geometry, Mesh):
        g_type = "mesh"
        g_param = (visual.geometry.filename, visual.geometry.scale)
    elif isinstance(visual.geometry, Cylinder):
        g_type = "cylinder"
        g_param = (visual.geometry.radius, visual.geometry.length)
    elif isinstance(visual.geometry, Box):
        g_type = "box"
        g_param = visual.geometry.size
    elif isinstance(visual.geometry, Sphere):
        g_type = "sphere"
        g_param = visual.geometry.radius
    else:
        g_type = None
        g_param = None
    
    return Visual(v_tf, g_type, g_param)


def _build_chain_recurse(root_frame, lmap, joints):
    """Recursively build kinematic tree from URDF joints."""
    children = []
    for j in joints:
        if j.parent == root_frame.link.name:
            # Extract joint limits
            try:
                limits = (j.limit.lower, j.limit.upper)
            except AttributeError:
                limits = None
            
            try:
                velocity_limits = (-j.limit.velocity, j.limit.velocity)
            except AttributeError:
                velocity_limits = None
            
            try:
                effort_limits = (-j.limit.effort, j.limit.effort)
            except AttributeError:
                effort_limits = None
            
            # Create child frame
            child_frame = Frame(j.child)
            child_frame.joint = Joint(
                j.name,
                offset=_convert_transform(j.origin),
                joint_type=JOINT_TYPE_MAP[j.type],
                axis=j.axis,
                limits=limits,
                velocity_limits=velocity_limits,
                effort_limits=effort_limits
            )
            
            # Add link information
            link = lmap[j.child]
            child_frame.link = Link(
                link.name,
                offset=_convert_transform(link.origin),
                inertial=_convert_inertial(link.inertial),
                visuals=[_convert_visual(link.visual)]
            )
            
            # Recurse for children
            child_frame.children = _build_chain_recurse(child_frame, lmap, joints)
            children.append(child_frame)
    
    return children


def build_chain_from_urdf(
    data: str,
    *,
    floating_base: bool = False,
    base_frame_name: str = "floating_base",
    dtype=torch.float32,
    device="cpu",
):
    """
    Build a Chain from URDF data.

    Args:
        data: URDF XML string
        floating_base: If True, wrap URDF root in synthetic floating base
        base_frame_name: Name of synthetic base frame (only used if floating_base=True)
        dtype: Torch dtype for the chain
        device: Torch device for the chain

    Returns:
        Chain object with configuration format:
            Fixed-base: q = [joint_angles...]
            Floating-base: q = [tx, ty, tz, qw, qx, qy, qz, joint_angles]
    """
    robot = URDF.from_xml_string(data)
    lmap = robot.link_map
    joints = robot.joints
    n_joints = len(joints)

    # Find URDF root link (parent that is never a child)
    has_root = [True] * n_joints
    for i in range(n_joints):
        for j in range(i + 1, n_joints):
            if joints[i].parent == joints[j].child:
                has_root[i] = False
            elif joints[j].parent == joints[i].child:
                has_root[j] = False
    
    root_link = None
    for i in range(n_joints):
        if has_root[i]:
            root_link = lmap[joints[i].parent]
            break
    
    if root_link is None:
        raise ValueError("Could not find root link in URDF")

    # Build kinematic tree from URDF root
    root_frame = Frame(root_link.name)
    root_frame.joint = Joint()  # Fixed joint at root
    root_frame.link = Link(
        root_link.name,
        _convert_transform(root_link.origin),
        inertial=_convert_inertial(root_link.inertial),
        visuals=[_convert_visual(root_link.visual)],
    )
    root_frame.children = _build_chain_recurse(root_frame, lmap, joints)

    # Optionally wrap with floating base
    if floating_base:
        base_frame = Frame(base_frame_name)
        base_frame.joint = Joint(
            name=f"{base_frame_name}_joint",
            joint_type="fixed"
        )
        base_frame.link = Link(base_frame_name, tf.Transform3d())
        base_frame.children = [root_frame]
        root_frame = base_frame

    # Create Chain with floating base flag
    return chain.Chain(
        root_frame,
        floating_base=floating_base,
        base_name=base_frame_name,
        dtype=dtype,
        device=device
    )
