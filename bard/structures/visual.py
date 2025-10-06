"""Visual geometry representation for robot links."""

from typing import Optional, Tuple, Union
import torch


class Visual:
    """
    Visual geometry information for a robot link.
    
    Used for visualization and rendering, not for collision detection.
    
    Attributes:
        offset: Transform from link frame to visual geometry origin
        geom_type: Geometry type ('box', 'cylinder', 'sphere', 'capsule', 'mesh')
        geom_param: Geometry-specific parameters
    """
    
    TYPES = ['box', 'cylinder', 'sphere', 'capsule', 'mesh']

    def __init__(
        self,
        offset: Optional[object] = None,
        geom_type: Optional[str] = None,
        geom_param: Optional[Union[Tuple, str]] = None
    ):
        """
        Initialize visual geometry.
        
        Args:
            offset: Transform3d offset from link frame
            geom_type: Type of geometry from Visual.TYPES
            geom_param: Parameters depending on geom_type:
                - 'box': (size_x, size_y, size_z)
                - 'cylinder': (radius, length)
                - 'sphere': radius
                - 'capsule': (radius, length)
                - 'mesh': (filename, scale)
        """
        self.offset = offset
        self.geom_type = geom_type
        self.geom_param = geom_param
        
        if geom_type is not None and geom_type not in self.TYPES:
            raise ValueError(f"Invalid geometry type '{geom_type}'. Must be one of {self.TYPES}")

    def __repr__(self) -> str:
        return (f"Visual(offset={self.offset}, "
                f"geom_type='{self.geom_type}', "
                f"geom_param={self.geom_param})")