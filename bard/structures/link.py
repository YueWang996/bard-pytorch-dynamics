"""Robot link representation with inertial and visual properties."""

from typing import Optional, Sequence, Tuple
import torch
from .visual import Visual


class Link:
    """Represents a rigid body in the robot's kinematic chain.

    A link holds the physical properties of a robot segment, including its
    mass, inertia, and visual geometries for rendering.

    Attributes:
        name (str): The unique identifier for the link.
        offset (Transform3d): The static transform from the parent joint frame to
            this link's reference frame.
        inertial (Optional[Tuple[object, float, torch.Tensor]]): A tuple
            containing the link's inertial properties: (Center of Mass offset,
            mass, 3x3 inertia tensor). None for massless links.
        visuals (List[Visual]): A list of Visual objects for rendering.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        offset: Optional[object] = None,
        inertial: Optional[Tuple[object, float, torch.Tensor]] = None,
        visuals: Sequence["Visual"] = (),
    ):
        """Initializes a robot link.

        Args:
            name (Optional[str], optional): The name/identifier of the link.
                Defaults to "unnamed_link".
            offset (Optional[object], optional): A Transform3d object from the
                parent joint frame to this link's frame. Defaults to None.
            inertial (Optional[Tuple[object, float, torch.Tensor]], optional):
                A tuple `(com_offset, mass, inertia_matrix)` for the link's
                inertial properties. Defaults to None (massless).
            visuals (Sequence[Visual], optional): A sequence of Visual
                geometries attached to the link. Defaults to an empty tuple.
        """
        self.name = name if name is not None else "unnamed_link"
        self.offset = offset
        self.inertial = inertial
        self.visuals = list(visuals)

    def to(
        self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None
    ) -> "Link":
        """Moves all tensor data in the link to a specified device/dtype.

        Args:
            dtype (Optional[torch.dtype], optional): The target data type.
            device (Optional[torch.device], optional): The target device.

        Returns:
            Link: Returns self for method chaining.
        """
        if self.offset is not None:
            self.offset = self.offset.to(dtype=dtype, device=device)

        if self.inertial is not None:
            com_offset, mass, inertia = self.inertial
            if com_offset is not None:
                com_offset = com_offset.to(dtype=dtype, device=device)
            if isinstance(inertia, torch.Tensor):
                inertia = inertia.to(dtype=dtype, device=device)
            self.inertial = (com_offset, mass, inertia)

        return self

    def __repr__(self) -> str:
        """Returns a concise string representation of the Link."""
        has_inertia = self.inertial is not None
        num_visuals = len(self.visuals)
        return (
            f"Link(name='{self.name}', "
            f"offset={self.offset}, "
            f"has_inertial={has_inertia}, "
            f"num_visuals={num_visuals})"
        )
