"""Robot link representation with inertial and visual properties."""

from typing import Optional, Sequence, Tuple
import torch
from .visual import Visual


class Link:
    """
    Robot link with mass, inertia, and visual properties.
    
    A link represents a rigid body in the robot's kinematic chain.
    
    Attributes:
        name: Link identifier
        offset: Transform from parent joint frame to this link's reference frame
        inertial: Tuple of (offset, mass, inertia_tensor) or None
        visuals: List of Visual objects for rendering
    
    Inertial format:
        - offset: Transform3d to center of mass
        - mass: Scalar mass value
        - inertia_tensor: 3x3 rotational inertia matrix about COM
    """

    def __init__(
        self,
        name: Optional[str] = None,
        offset: Optional[object] = None,
        inertial: Optional[Tuple[object, float, torch.Tensor]] = None,
        visuals: Sequence['Visual'] = ()
    ):
        """
        Initialize robot link.
        
        Args:
            name: Link name/identifier
            offset: Transform3d from parent joint to link frame
            inertial: (com_offset, mass, inertia_matrix) tuple or None for massless
            visuals: Sequence of Visual geometries
        """
        self.name = name if name is not None else "unnamed_link"
        self.offset = offset
        self.inertial = inertial
        self.visuals = list(visuals)

    def to(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
    ) -> 'Link':
        """
        Move link data to specified dtype/device.
        
        Args:
            dtype: Target data type
            device: Target device
            
        Returns:
            Self for method chaining
        """
        if self.offset is not None:
            self.offset = self.offset.to(dtype=dtype, device=device)
        
        # Update inertial properties
        if self.inertial is not None:
            com_offset, mass, inertia = self.inertial
            if com_offset is not None:
                com_offset = com_offset.to(dtype=dtype, device=device)
            if isinstance(inertia, torch.Tensor):
                inertia = inertia.to(dtype=dtype, device=device)
            self.inertial = (com_offset, mass, inertia)
        
        return self

    def __repr__(self) -> str:
        has_inertia = self.inertial is not None
        num_visuals = len(self.visuals)
        return (f"Link(name='{self.name}', "
                f"offset={self.offset}, "
                f"has_inertial={has_inertia}, "
                f"num_visuals={num_visuals})")