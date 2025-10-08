"""Visual geometry representation for robot links."""

from typing import Optional, Tuple, Union
import torch


class Visual:
    """Represents the visual geometry of a robot link for rendering.

    This class stores information about a link's appearance, such as its shape,
    size, and offset from the link's reference frame. It is not used for
    physics or collision detection.

    Attributes:
        offset (Transform3d): The static transform from the link frame to the
            origin of the visual geometry.
        geom_type (Optional[str]): The type of geometry, e.g., 'box', 'cylinder',
            'sphere', or 'mesh'.
        geom_param (Optional[Union[Tuple, str]]): Parameters that define the
            geometry's shape and size.
    """

    TYPES = ["box", "cylinder", "sphere", "capsule", "mesh"]

    def __init__(
        self,
        offset: Optional[object] = None,
        geom_type: Optional[str] = None,
        geom_param: Optional[Union[Tuple, str]] = None,
    ):
        """Initializes a visual geometry.

        Args:
            offset (Optional[object], optional): A Transform3d object representing
                the offset from the link frame. Defaults to None.
            geom_type (Optional[str], optional): The type of geometry. Must be
                one of 'box', 'cylinder', 'sphere', 'capsule', 'mesh'. Defaults to None.
            geom_param (Optional[Union[Tuple, str]], optional): Parameters defining
                the geometry's shape. Varies by `geom_type`:
                - 'box': (width, depth, height)
                - 'cylinder': (radius, length)
                - 'sphere': radius
                - 'mesh': (filename, scale_tuple)
                Defaults to None.

        Raises:
            ValueError: If `geom_type` is not one of the supported types.
        """
        self.offset = offset
        self.geom_type = geom_type
        self.geom_param = geom_param

        if geom_type is not None and geom_type not in self.TYPES:
            raise ValueError(f"Invalid geometry type '{geom_type}'. Must be one of {self.TYPES}")

    def __repr__(self) -> str:
        """Returns a concise string representation of the Visual."""
        return (
            f"Visual(offset={self.offset}, "
            f"geom_type='{self.geom_type}', "
            f"geom_param={self.geom_param})"
        )
