import numpy as np
# import enum
from .camera import Camera
import glfw

class FPCameraControls:

    def __init__(self):
        self.camera = Camera()
        self.move_speed = 2.0
        self.strafe_speed = 2.0
        self.vertical_speed = 0.1
        self.mouse_sensitivity = 0.1
        self.yaw = 0.0
        self.pitch = 0.0
        self.keys_pressed = set()
        self.prev_x = 0
        self.prev_y = 0
        self.dx=0
        self.dy=0
        self.perform_tracking = False
        self.prev_t = 0.0

    def handle_key_event(self, key, state):
        if state == "pressed":
            self.keys_pressed.add(key)
        elif state == "released" and key in self.keys_pressed:
            self.keys_pressed.remove(key)

    
    def start_tracking(self, x, y):
        self.prev_x = x
        self.prev_y = y
        #Init yaw and pitch to the current direction of the camera
        direction = self.camera.look_at - self.camera.eye
        direction = direction / np.linalg.norm(direction)

        self.yaw = np.degrees(np.arctan2(direction[1], direction[0]))
        self.pitch = np.degrees(np.arcsin(direction[2]))

        self.perform_tracking = True

    def handle_mouse_motion(self, x, y):
        if not self.perform_tracking:
            return
        self.dx += x - self.prev_x
        self.dy += y - self.prev_y
        self.prev_x = x
        self.prev_y = y

    def update_rotation(self):
        if self.dx == 0 and self.dy == 0:
            return
        self.yaw -= self.dx * self.mouse_sensitivity # inverted the sign here
        self.pitch -= self.dy * self.mouse_sensitivity 

        # Limit pitch to prevent flipping
        self.pitch = np.clip(self.pitch, -89.0, 89.0)

        self.camera.look_at = np.array([
            np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
            np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
            np.sin(np.radians(self.pitch))
        ])+self.camera.eye
        
        self.dx=0
        self.dy=0
        return True



    def update_movement(self):
        update = False
        dt=glfw.get_time()-self.prev_t
        self.prev_t=glfw.get_time()
        # Update camera position, z forward, q left, s back, d right
        direction = self.camera.look_at - self.camera.eye
        direction = direction / np.linalg.norm(direction)
        # print("norm",np.linalg.norm(direction))
        if glfw.KEY_W in self.keys_pressed:
            self.camera.eye += self.move_speed * direction*dt
            self.camera.look_at += self.move_speed * direction*dt
            update = True
        if glfw.KEY_S in self.keys_pressed:
            self.camera.eye -= self.move_speed * direction*dt
            self.camera.look_at -= self.move_speed * direction*dt
            update = True
        if glfw.KEY_A in self.keys_pressed:
            self.camera.eye -= self.strafe_speed * np.cross(direction, self.camera.up)*dt
            self.camera.look_at -= self.strafe_speed * np.cross(direction, self.camera.up)*dt
            update = True
        if glfw.KEY_D in self.keys_pressed:
            self.camera.eye += self.strafe_speed * np.cross(direction, self.camera.up)*dt
            self.camera.look_at += self.strafe_speed * np.cross(direction, self.camera.up)*dt
            update = True
        return update
    
    def update(self):
        update = self.update_movement()
        update = self.update_rotation() or update
        return update


            