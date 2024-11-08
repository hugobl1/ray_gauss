import numpy as np

from .vecmath import length, normalize, cross
from .properties import get_member, set_float, set_float3


class Camera:
    """Implements a perspective camera."""

    __slots__ = ['_eye', '_look_at', '_up', '_fov_y', '_aspect_ratio']

    def __init__(self, eye=None, look_at=None, up=None, fov_y=None, aspect_ratio=None):
        self.eye = eye
        self.look_at = look_at
        self.up = up
        self.fov_y = fov_y
        self.aspect_ratio = aspect_ratio

    eye = property(get_member("_eye"), set_float3("_eye", 1.0))
    look_at = property(get_member("_look_at"), set_float3("_look_at", 0.0))
    up = property(get_member("_up"), set_float3("_up", [0.0,1.0,0.0]))

    fov_y = property(get_member("_fov_y"), set_float("_fov_y", 35.0))
    aspect_ratio = property(get_member("_aspect_ratio"), set_float("_aspect_ratio", 1.0))

    def _get_direction(self):
        return normalize(self.look_at - self.eye)
    def _set_direction(self, value):
        self.look_at = self.eye + length(self.look_at - self.eye)*value;
    direction = property(_get_direction, _set_direction)

    def uvw_frame(self):
        # do not normalize W -- it implies focal length
        W = self.look_at - self.eye
        wlen = length(W)
        assert wlen > 0, (self.eye, self.look_at)

        U = normalize(cross(W, self.up))
        V = normalize(cross(U, W))

        vlen = wlen * np.tan(0.5 * np.deg2rad(self.fov_y))
        V *= vlen

        ulen = vlen * self.aspect_ratio
        U *= ulen

        return (U,V,W)
