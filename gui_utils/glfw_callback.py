from gui_utils.trackball import Trackball, TrackballViewMode
import glfw
import cupy as cp

#------------------------------------------------------------------------------
# GLFW callbacks
#------------------------------------------------------------------------------

def mouse_button_callback(window, button, action, mods):
    state = glfw.get_window_user_pointer(window)
    if state.camera_mode == 0: # Trackball mode
        (x, y) = glfw.get_cursor_pos(window)
        if action is glfw.PRESS:
            state.mouse_button = button
            state.trackball.start_tracking(x, y)
        else:
            state.mouse_button = -1
    if state.camera_mode == 1:
        #Si on appuie sur le bouton droit de la souris on active/désactive le mouvement de la souris
        (x, y) = glfw.get_cursor_pos(window)
        if button == glfw.MOUSE_BUTTON_RIGHT:
            if action is glfw.PRESS:
                state.fp_controls.start_tracking(x, y)
            elif action is glfw.RELEASE:
                state.fp_controls.perform_tracking = False


def cursor_position_callback(window, x, y):
    state = glfw.get_window_user_pointer(window)
    if state.is_window_focused:
        return
    if state.camera_mode == 0: # Trackball mode
        if state.mouse_button is glfw.MOUSE_BUTTON_LEFT:
            state.trackball.view_mode = TrackballViewMode.LookAtFixed
            state.trackball.update_tracking(x, y, state.params.width, state.params.height)
            state.camera_changed = True
        elif state.mouse_button is glfw.MOUSE_BUTTON_RIGHT:
            state.trackball.view_mode = TrackballViewMode.EyeFixed
            state.trackball.update_tracking(x, y, state.params.width, state.params.height)
            state.camera_changed = True
    if state.camera_mode == 1:
        state.fp_controls.handle_mouse_motion(x, y)
        state.camera_changed = True

def window_size_callback(window, res_x, res_y):
    state = glfw.get_window_user_pointer(window)
    if state.minimized:
        return

    res_x = max(res_x, 1)
    res_y = max(res_y, 1)

    state.params.width = res_x
    state.params.height = res_y
    state.params.hit_sphere_idx = cp.zeros((state.params.max_prim_slice*state.params.width*state.params.height),dtype=cp.uint32).data.ptr
    state.depth_buffer = cp.zeros((state.params.width*state.params.height),dtype=cp.float32)
    state.params.depth_buffer_ptr = state.depth_buffer.data.ptr
    # state.params.depth_buffer = cp.zeros((state.params.width*state.params.height),dtype=cp.float32).data.ptr
    state.camera_changed = True
    state.resize_dirty = True

def window_iconify_callback(window, iconified):
    state = glfw.get_window_user_pointer(window)
    state.minimized = (iconified > 0)

def key_callback(window, key, scancode, action, mods):
    if action is glfw.PRESS:
        if key in { glfw.KEY_ESCAPE}:
            glfw.set_window_should_close(window, True)
    state = glfw.get_window_user_pointer(window)
    #if state.which_item is 1 or 2 the user in interacting with the slider
    if state.which_item in {0,1,2,6,8,11}:
        if glfw.get_key(window,glfw.KEY_LEFT):
            if state.which_item == 0:
                state.slider_train_iteration = max(state.slider_train_iteration-10,state.min_iteration)
            elif state.which_item == 1:
                state.idx_test_im = max(state.idx_test_im-1,state.min_idx_test_im)
                state.update_test_im = True
            elif state.which_item == 2:
                state.idx_train_im = max(state.idx_train_im-1,state.min_idx_train_im)
                state.update_train_im = True
            elif state.which_item == 6:
                state.params.degree_sh=max(state.params.degree_sh-1,0)
            elif state.which_item == 11:
                state.params.num_sg=max(state.params.num_sg-1,0)
            elif state.which_item == 8:
                state.added_before_iter=max(state.added_before_iter-10,0)
                state.update_added_gaussians = True
        elif glfw.get_key(window,glfw.KEY_RIGHT):
            if state.which_item == 0:
                state.slider_train_iteration = min(state.slider_train_iteration+10,state.max_iteration)
            elif state.which_item == 1:
                state.idx_test_im = min(state.idx_test_im+1,state.max_idx_test_im)
                state.update_test_im = True
            elif state.which_item == 2:
                state.idx_train_im = min(state.idx_train_im+1,state.max_idx_train_im)
                state.update_train_im = True
            elif state.which_item == 6:
                state.params.degree_sh=min(state.params.degree_sh+1,state.params.max_sh_degree)
            elif state.which_item == 11:
                state.params.num_sg=min(state.params.num_sg+1,state.params.max_sg_display)
            elif state.which_item == 8:
                state.added_before_iter=min(state.added_before_iter+10,state.max_iteration)
                state.update_added_gaussians = True

    if state.camera_mode == 1:
        if action is glfw.PRESS:
                state.fp_controls.handle_key_event(key, "pressed")
        elif action is glfw.RELEASE:
                state.fp_controls.handle_key_event(key, "released")

def scroll_callback(window, xscroll, yscroll):
    state = glfw.get_window_user_pointer(window)
    if state.camera_mode == 0: # Trackball mode
        if state.trackball.wheel_event(yscroll):
            state.camera_changed = True