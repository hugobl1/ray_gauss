import numpy as np
# import enum
from .camera import Camera
import glfw

class FPCameraControls:

    def __init__(self):
        self.camera = Camera()
        self.move_speed = 2.0
        self.strafe_speed = 2.0
        self.vertical_speed = 0.5
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

        # Axes locaux de la caméra
        self.forward = None
        self.right = None
        self.up = None

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
        # direction = direction / np.linalg.norm(direction)

        # self.yaw = np.degrees(np.arctan2(direction[1], direction[0]))
        # self.pitch = np.degrees(np.arcsin(direction[2]))

        # self.perform_tracking = True
    
        self.forward = direction / np.linalg.norm(direction)
        # Définir le vecteur up
        self.up = self.camera.up / np.linalg.norm(self.camera.up)
        # Calculer le vecteur right
        self.right = np.cross(self.forward, self.up)
        self.right /= np.linalg.norm(self.right)
        self.perform_tracking = True

    def handle_mouse_motion(self, x, y):
        if not self.perform_tracking:
            return
        self.dx += x - self.prev_x
        self.dy += y - self.prev_y
        self.prev_x = x
        self.prev_y = y

    def rotation_matrix(self, axis, theta):
        """
        Crée une matrice de rotation autour d'un axe donné par un angle theta (en radians).
        Utilise la formule de Rodrigues.
        """
        axis = axis / np.linalg.norm(axis)
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        return np.array([
            [a*a + b*b - c*c - d*d, 2*(b*c - a*d),     2*(b*d + a*c)],
            [2*(b*c + a*d),     a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
            [2*(b*d - a*c),     2*(c*d + a*b),     a*a + d*d - b*b - c*c]
        ])
    
    def update_rotation(self):
        if self.dx == 0 and self.dy == 0:
            return False

        # Appliquer les rotations basées sur les axes locaux
        yaw_angle = self.dx * self.mouse_sensitivity
        pitch_angle = self.dy * self.mouse_sensitivity

        # Créer les matrices de rotation
        R_yaw = self.rotation_matrix(self.up, np.radians(yaw_angle))
        R_pitch = self.rotation_matrix(self.right, np.radians(pitch_angle))

        # Appliquer la rotation yaw
        self.forward = R_yaw @ self.forward
        self.right = R_yaw @ self.right
        # L'up vector reste inchangé lors du yaw
        # self.up = R_yaw @ self.up  # Si besoin de mise à jour, décommentez

        # Appliquer la rotation pitch
        self.forward = R_pitch @ self.forward
        self.up = R_pitch @ self.up
        # Le right vector reste inchangé lors du pitch
        # self.right = R_pitch @ self.right  # Si besoin de mise à jour, décommentez

        # Normaliser les vecteurs pour éviter l'accumulation d'erreurs
        self.forward /= np.linalg.norm(self.forward)
        self.up /= np.linalg.norm(self.up)
        self.right = np.cross(self.forward, self.up)
        self.right /= np.linalg.norm(self.right)

        # Limiter le pitch pour éviter les retournements
        # Calculer l'angle de pitch actuel
        pitch = np.degrees(np.arcsin(self.forward[2]))
        pitch = np.clip(pitch, -89.0, 89.0)

        # Si le pitch dépasse les limites, réajuster la forward et up vectors
        if pitch != np.degrees(np.arcsin(self.forward[2])):
            # Calculer l'angle de correction
            correction_angle = pitch - np.degrees(np.arcsin(self.forward[2]))
            # Appliquer une rotation corrective autour de l'axe right
            R_correction = self.rotation_matrix(self.right, np.radians(correction_angle))
            self.forward = R_correction @ self.forward
            self.up = R_correction @ self.up

            # Normaliser après correction
            self.forward /= np.linalg.norm(self.forward)
            self.up /= np.linalg.norm(self.up)

        # Mettre à jour le look_at basé sur la nouvelle forward vector
        self.camera.look_at = self.camera.eye + self.forward

        # Réinitialiser les deltas de la souris
        self.dx = 0
        self.dy = 0

        return True
    
    # def update_rotation(self):
    #     if self.dx == 0 and self.dy == 0:
    #         return
    #     self.yaw -= self.dx * self.mouse_sensitivity # inverted the sign here
    #     self.pitch -= self.dy * self.mouse_sensitivity 

    #     # print("self.camera.up",self.camera.up)
    #     # Limit pitch to prevent flipping
    #     self.pitch = np.clip(self.pitch, -89.0, 89.0)

    #     self.camera.look_at = np.array([
    #         np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
    #         np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
    #         np.sin(np.radians(self.pitch))
    #     ])+self.camera.eye
        
    #     self.dx=0
    #     self.dy=0
    #     return True



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
        if glfw.KEY_Q in self.keys_pressed:#Move down in z axis
            self.camera.eye -= self.vertical_speed * self.camera.up*dt
            self.camera.look_at -= self.vertical_speed * self.camera.up*dt
            update = True
        if glfw.KEY_E in self.keys_pressed: #Move up in z axis
            self.camera.eye += self.vertical_speed * self.camera.up*dt
            self.camera.look_at += self.vertical_speed * self.camera.up*dt
            update = True
        return update
    
    def update(self):
        update = self.update_movement()
        update = self.update_rotation() or update
        return update


            