import enum

import numpy as np

from .properties import get_member, set_bool, set_float, set_float3
from .vecmath import dot, length, normalize
from .camera import Camera

class TrackballViewMode(enum.Enum):
    EyeFixed = 0
    LookAtFixed = 1

class Trackball:
    __slots__ = ['_gimbal_lock','_view_mode', '_camera', '_camera_eye_lookat_distance',
                 '_zoom_multiplier', '_move_speed', '_roll_speed', '_latitude', '_longitude',
                 '_previous_position_x', '_previous_position_y', '_perform_tracking',
                 '_u', '_v', '_w']

    def __init__(self):
        # initialize all attributes to default values
        for slot in self.__slots__:
            setattr(self, slot[1:], None)

    camera_eye_lookat_distance = property(get_member('_camera_eye_lookat_distance'),
                                          set_float('_camera_eye_lookat_distance', 0.0))
    zoom_multiplier = property(get_member('_zoom_multiplier'), set_float('_zoom_multiplier', 1.1))
    move_speed = property(get_member('_move_speed'), set_float('_move_speed', 1.0))
    roll_speed = property(get_member('_roll_speed'), set_float('_roll_speed', 0.5))
    latitude = property(get_member('_latitude'), set_float('_latitude', 0.0))
    longitude = property(get_member('_longitude'), set_float('_longitude', 0.0))
    previous_position_x = property(get_member('_previous_position_x'), set_float('_previous_position_x', 0))
    previous_position_y = property(get_member('_previous_position_y'), set_float('_previous_position_y', 0))

    gimbal_lock = property(get_member('_gimbal_lock'), set_bool('_gimbal_lock', False))
    perform_tracking = property(get_member('_perform_tracking'), set_bool('_perform_tracking', False))

    u = property(get_member("_u"), set_float3("_u", 0.0))
    v = property(get_member("_v"), set_float3("_v", 0.0))
    w = property(get_member("_w"), set_float3("_w", 0.0))

    def _get_view_mode(self):
        return self._view_mode
    def _set_view_mode(self, view_mode):
        if view_mode is None:
            view_mode = TrackballViewMode.LookAtFixed
        assert isinstance(view_mode, TrackballViewMode), type(view_mode)
        self._view_mode = view_mode
    view_mode = property(_get_view_mode, _set_view_mode)

    def _get_camera(self):
        return self._camera
    def _set_camera(self, camera):
        """
        Set the camera that will be changed according to user input.
        Warning, this also initializes the reference frame of the trackball from the camera.
        The reference frame defines the orbit's singularity.
        """
        if camera is None:
            camera = Camera()
        assert isinstance(camera, Camera), type(camera)
        self._camera = camera
        self.reinitialize_orientation_from_camera()
    camera = property(_get_camera, _set_camera)

    def start_tracking(self, x, y):
        self.previous_position_x = x
        self.previous_position_y = y
        self.perform_tracking = True

    def update_tracking(self, x, y, canvas_width, canvas_height):
        if not self._perform_tracking:
            return self.start_tracking(x, y)

        delta_x = x - self.previous_position_x
        delta_y = y - self.previous_position_y

        if delta_x == 0 and delta_y == 0:
            return

        self.previous_position_x = x
        self.previous_position_y = y

        self.latitude = np.deg2rad(min(+89.0, max(-89.0, np.rad2deg(self.latitude) + 0.5*delta_y)))
        self.longitude = np.deg2rad(np.fmod(np.rad2deg(self.longitude) - 0.5*delta_x, 360.0))

        self._update_camera()

        if not self.gimbal_lock:
            self.reinitialize_orientation_from_camera()
            self.camera.up = self.w

    def wheel_event(self, direction):
        self.zoom(direction)
        return True

    def zoom(self, direction):
        zoom = np.float32(1.0/self.zoom_multiplier if direction > 0 else self.zoom_multiplier)
        self.camera_eye_lookat_distance *= zoom

        look_at = self.camera.look_at
        eye = self.camera.eye
        self.camera.eye = look_at + (eye - look_at) * zoom

    def reinitialize_orientation_from_camera(self):
        """
        Adopts the reference frame from the camera.
        Note that the reference frame of the camera usually has a different 'up' than the 'up' of the camera.
        Though, typically, it is desired that the trackball's reference frame aligns with the actual up of the camera.
        """
        u, v, w = self.camera.uvw_frame()

        self.u = normalize(+u)
        self.v = normalize(-w)
        self.w = normalize(+v)

        self.latitude = 0.0
        self.longitude = 0.0

        self.camera_eye_lookat_distance = length(self.camera.look_at - self.camera.eye)
        assert(self.camera_eye_lookat_distance > 0)

    def set_reference_frame(self, u, v, w):
        """
        Specify the frame of the orbit that the camera is orbiting around.
        The important bit is the 'up' of that frame as this is defines the singularity.
        Here, 'up' is the 'w' component.
        Typically you want the up of the reference frame to align with the up of the camera.
        However, to be able to really freely move around, you can also constantly update
        the reference frame of the trackball. This can be done by calling reinitOrientationFromCamera().
        In most cases it is not required though (set the frame/up once, leave it as is).
        """
        self.u = u
        self.v = v
        self.w = w

        assert length(self.camera.look_at - self.camera.eye) != 0
        dir_ws = -normalize(self.camera.look_at - self.camera.eye)

        dirx = dot(dir_ws, u)
        diry = dot(dir_ws, v)
        dirz = dot(dir_ws, w)

        self.longitude = np.arctan2(dirx, diry)
        self.latitude = np.arcsin(dirz)


    def _update_camera(self):
        dirx = np.cos(self._latitude)*np.sin(self._longitude)
        diry = np.cos(self._latitude)*np.cos(self._longitude)
        dirz = np.sin(self._latitude)

        dir_ws = self.u * dirx + self.v * diry + self.w * dirz

        if self.view_mode is TrackballViewMode.EyeFixed:
            eye = self.camera.eye
            self.camera.look_at = eye - dir_ws * self.camera_eye_lookat_distance
        elif self.view_mode is TrackballViewMode.LookAtFixed:
            look_at = self.camera.look_at
            self.camera.eye = look_at + dir_ws * self.camera_eye_lookat_distance
        else:
            raise NotImplementedError(self.view_mode)

    def _move_backward(self, speed):
        dir_ws = normalize(self.camera.look_at - self.camera.eye)
        self.camera.eye -= dir_ws * speed
        self.camera.look_at -= dir_ws * speed

    def _move_forward(self, speed):
        dir_ws = normalize(self.camera.look_at - self.camera.eye)
        self.camera.eye += dir_ws * speed
        self.camera.look_at += dir_ws * speed

    def _move_left(self, speed):
        u = normalize( self.camera.uvw_frame()[0] )
        self.camera.eye -= u*speed
        self.camera.look_at -= u*speed

    def _move_right(self, speed):
        u = normalize( self.camera.uvw_frame()[0] )
        self.camera.eye += u*speed
        self.camera.look_at += u*speed

    def _move_down(self, speed):
        v = normalize( self.camera.uvw_frame()[1] )
        self.camera.eye -= v*speed
        self.camera.look_at -= v*speed

    def _move_up(self, speed):
        v = normalize( self.camera.uvw_frame()[1] )
        self.camera.eye += v*speed
        self.camera.look_at += v*speed

    def _roll_right(self, speed):
        u, v, _ = map(normalize, self.camera.uvw_frame())
        self.camera.up = u*np.cos(np.deg2rad(90.0 - speed)) + v*np.sin(np.deg2rad(90.0 - speed))

    def _roll_left(self, speed):
        u, v, _ = map(normalize, self.camera.uvw_frame())
        self.camera.up = u*np.cos(np.deg2rad(90.0 + speed)) + v*np.sin(np.deg2rad(90.0 + speed))
