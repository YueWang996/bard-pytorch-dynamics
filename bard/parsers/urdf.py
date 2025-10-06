"""
URDF parser for building a `bard` kinematic Chain.

This module provides the `build_chain_from_urdf` function, which serves as the
primary entry point for creating a robot model from a URDF file's content. It
handles parsing of links, joints, and inertial properties, and supports the
creation of both fixed-base and floating-base robot representations.
"""
from .urdf_parser_py.urdf import URDF, Mesh, Cylinder, Box, Sphere
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
    """Converts a URDF <origin> tag into a bard Transform3d object."""
    if origin is None:
        return tf.Transform3d()
    rpy = torch.tensor(origin.rpy, dtype=torch.float32, device="cpu")
    return tf.Transform3d(rot=tf.quaternion_from_euler(rpy, "sxyz"), pos=origin.xyz)


def _convert_inertial(inertial):
    """Converts a URDF <inertial> tag into an (offset, mass, inertia_tensor) tuple."""
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
    """Converts a URDF <visual> tag into a bard Visual object."""
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
    """Recursively traverses the joint list to build the kinematic tree structure.

    Args:
        root_frame (bard.structures.Frame): The parent frame to which children will be attached.
        lmap (dict): A dictionary mapping link names to link objects from the URDF parser.
        joints (list): The list of all joint objects from the URDF parser.

    Returns:
        list[bard.structures.Frame]: A list of child frames attached to the `root_frame`.
    """
    children = []
    for j in joints:
        if j.parent == root_frame.link.name:
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
            
            link = lmap[j.child]
            child_frame.link = Link(
                link.name,
                offset=_convert_transform(link.origin),
                inertial=_convert_inertial(link.inertial),
                visuals=[_convert_visual(link.visual)]
            )
            
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
    """Builds a `bard` Chain object from a URDF data string.

    This is the main entry point for creating a kinematic chain from a URDF.
    It parses the XML data, constructs the kinematic tree, and can optionally
    prepend a 6-DOF floating base for models like quadrupeds or humanoids.

    Args:
        data (str or bytes): The URDF XML content as a string or bytes.
        floating_base (bool, optional): If True, a 6-DOF floating base is
            prepended to the root of the kinematic tree. Defaults to False.
        base_frame_name (str, optional): The name assigned to the synthetic base
            frame when `floating_base` is True. Defaults to "floating_base".
        dtype (torch.dtype, optional): The PyTorch data type to use for the
            chain's tensors. Defaults to torch.float32.
        device (str or torch.device, optional): The PyTorch device to place the
            chain's tensors on. Defaults to "cpu".

    Returns:
        bard.core.chain.Chain: The constructed kinematic chain object. The configuration
            vector `q` will have the following format:
            - **Fixed-base**: `q = [joint_angle_1, ...]`
            - **Floating-base**: `q = [tx, ty, tz, qw, qx, qy, qz, joint_angle_1, ...]`
    
    Raises:
        ValueError: If a root link cannot be determined from the URDF structure.
    """
    robot = URDF.from_xml_string(data)
    lmap = robot.link_map
    joints = robot.joints
    n_joints = len(joints)

    # Find URDF root link (a parent that is never a child)
    is_child = {j.child for j in joints}
    root_links = [j.parent for j in joints if j.parent not in is_child]
    
    if not root_links:
        raise ValueError("Could not find a root link in the URDF (a link that is a parent but not a child).")
    
    root_link_name = root_links[0]
    root_link = lmap[root_link_name]

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

    # Optionally wrap with a synthetic floating base
    if floating_base:
        base_frame = Frame(base_frame_name)
        base_frame.joint = Joint(
            name=f"{base_frame_name}_joint",
            joint_type="fixed"
        )
        base_frame.link = Link(base_frame_name, tf.Transform3d())
        base_frame.children = [root_frame]
        root_frame = base_frame

    # Create and return the final Chain object
    return chain.Chain(
        root_frame,
        floating_base=floating_base,
        base_name=base_frame_name,
        dtype=dtype,
        device=device
    )