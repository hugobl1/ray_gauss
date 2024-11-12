
import glfw
import OpenGL.GL as gl

import imgui
from imgui.integrations.glfw import GlfwRenderer
from .vecmath import normalize
import numpy as np


def static_vars(**kwargs):
    """
    Attach a static variables local to decorated function.
    """
    def decorate(f):
        for k in kwargs:
            setattr(f, k, kwargs[k])
        return f
    return decorate

def init_gl():
    gl.glClearColor(0.212, 0.271, 0.31, 1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)

def init_imgui(window):
    imgui.create_context()
    impl = GlfwRenderer(window)
    impl.io.fonts.add_font_default()
    imgui.core.style_colors_dark();
    imgui.get_style().window_border_size = 0.0
    return impl

def init_ui(window_title, width, height):
    if not glfw.init():
        raise RuntimeError("Could not initialize OpenGL context")

    window = glfw.create_window(int(width), int(height), window_title, None, None)
    glfw.make_context_current(window)

    if not window:
        raise RuntimeError("Could not initialize Window")

    glfw.swap_interval(0)

    init_gl()
    impl = init_imgui(window)

    return window, impl

def display_text(text, x, y):
    imgui.set_next_window_bg_alpha(0.0)
    imgui.set_next_window_position(x, y)

    flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE |
             imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_INPUTS)

    imgui.begin("TextOverlayFG", None, flags)
    imgui.push_style_color(imgui.COLOR_TEXT, 0.7, 0.7, 0.7, 1.0)
    imgui.text(text)
    imgui.pop_style_color()
    imgui.end()

@static_vars(total_subframe_count=0, last_update_frames=0,
        last_update_time=None, display_text=""
        )
def display_stats(state,state_update_time, render_time, display_time):
    display_update_min_interval_time = 0.5

    cur_time = glfw.get_time()

    display_stats.last_update_frames += 1
    last_update_time = display_stats.last_update_time or cur_time - 0.5
    last_update_frames = display_stats.last_update_frames
    total_subframe_count = display_stats.total_subframe_count

    dt = cur_time - last_update_time

    do_update = (dt > display_update_min_interval_time) or (total_subframe_count == 0)

    if do_update:
        fps = last_update_frames / dt
        state_ms = 1000.0 * state_update_time / last_update_frames
        render_ms = 1000.0 * render_time / last_update_frames
        display_ms = 1000.0 * display_time / last_update_frames

        display_stats.last_update_time = cur_time
        display_stats.last_update_frames = 0

        display_stats.display_text = \
f"""{fps:5.1f} fps

state update: {state_ms:8.1f} ms
render      : {render_ms:8.1f} ms
display     : {display_ms:8.1f} ms
"""

    imgui.new_frame()
    display_text(display_stats.display_text, 10.0, 10.0)

    ############################## Test set Image ########################################
    # imgui.set_next_window_position(10.0, 150.0)  # Adjust the y position as needed
    # imgui.set_next_window_size(300, (180+60+40))  # Ensure window has a size
    flag = imgui.WINDOW_ALWAYS_AUTO_RESIZE
    imgui.begin("Menu", False, flag)
    # imgui.begin("Test-Train set Image", False, imgui.WINDOW_NO_RESIZE)
    #Create a child window to choose the iteration

    state.is_window_focused = imgui.is_window_focused() 
    ############################  Sous-menu 1 #############################################
    ############################  Choix de l'itération ####################################
    if state.gui_mode:
        sub_menu_train_iteration_expanded = True
        sub_menu_train_iteration_expanded, _ = imgui.collapsing_header("Itération durant l'entraînement", sub_menu_train_iteration_expanded)
        if sub_menu_train_iteration_expanded:
            imgui.begin_child("Choose iteration", 280, 60, border=True)
            imgui.text("Choose iteration")
            changed, state.slider_train_iteration = imgui.slider_int("", state.slider_train_iteration, state.min_iteration, state.max_iteration)
            if imgui.is_item_active():
                state.which_item = 0

            state.is_window_focused = imgui.is_window_focused() or state.is_window_focused 
            imgui.end_child()
    ######################################################################################
        
    ############################  Sous-menu 2 #############################################
    ################  Comparaison avec l'ensemble d'entraînement/test #####################
    if state.gui_mode:
        sub_menu_train_test_idx_expanded = True
        sub_menu_train_test_idx_expanded, _ = imgui.collapsing_header("Comparaison avec Train/Test set", sub_menu_train_test_idx_expanded)
        if sub_menu_train_test_idx_expanded:
            imgui.begin_child("Test-train set Image", 280, 180, border=True)
            imgui.text("Test set Image")
            state.update_test_im, state.idx_test_im = imgui.slider_int("Test index", state.idx_test_im, state.min_idx_test_im, state.max_idx_test_im)
            if imgui.is_item_active():
                state.which_item = 1

            #Slight vertical offset to avoid overlap
            imgui.dummy(0.0, 10.0)
            imgui.text("Train set Image")
            state.update_train_im, state.idx_train_im = imgui.slider_int("Train index", state.idx_train_im, state.min_idx_train_im, state.max_idx_train_im)
            if imgui.is_item_active():
                state.which_item = 2
            #Slight vertical offset to avoid overlap
            imgui.dummy(0.0, 10.0)

            #Two colors depending on state.print_error_image
            #Show default color of the button
            # print("Default imgui.COLOR_BUTTON: ", imgui.get_style().colors[imgui.COLOR_BUTTON])

            if state.print_error_image:
                #Darker than the default imgui.COLOR_BUTTON
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.2599*0.5,0.58999*0.5,0.98*0.5,0.4)
            else:
                #Default imgui.COLOR_BUTTON
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.2599,0.58999,0.98,0.4)
            if imgui.button("Error Image")and state.slider_used:
                #Switch to the error image
                state.print_error_image = not state.print_error_image
                if state.print_error_image:
                    state.print_gt_image = False
                    state.print_depth= False
            if imgui.is_item_active():
                state.which_item = 3
            imgui.pop_style_color()
            #Add a tick button to choose log scale
            imgui.same_line()
            changed, state.log_scale_error = imgui.checkbox("Log scale", state.log_scale_error)
            if imgui.is_item_active():
                state.which_item = 4

            imgui.dummy(0.0, 5.0)
            if state.print_gt_image:
                #Darker than the default imgui.COLOR_BUTTON
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.2599*0.5,0.58999*0.5,0.98*0.5,0.4)
            else:
                #Default imgui.COLOR_BUTTON
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.2599,0.58999,0.98,0.4)
            if imgui.button("GT Image")and state.slider_used:
                #Switch to the error image
                state.print_gt_image=not state.print_gt_image
                if state.print_gt_image:
                    state.print_error_image = False
                    state.print_depth= False
            if imgui.is_item_active():
                state.which_item = 5
            imgui.pop_style_color()
                
            state.is_window_focused = imgui.is_window_focused() or state.is_window_focused 
            imgui.end_child()
    ######################################################################################
        
    ############################  Sous-menu 3 #############################################
    ##################  Degré des harmoniques sphériques ##################################
    sub_menu_sh_degree_expanded = True
    sub_menu_sh_degree_expanded, _ = imgui.collapsing_header("Colorimetric properties", sub_menu_sh_degree_expanded)
    if sub_menu_sh_degree_expanded:
        imgui.begin_child("Spherical Harmonics Degree", 280, 50, border=True)
        #Slider for the spherical harmonics degree
        imgui.text("Spherical Harmonics Degree")
        changed, state.params.degree_sh = imgui.slider_int("", state.params.degree_sh, 0, state.params.max_sh_degree)
        if imgui.is_item_active():
            state.which_item = 6
        state.is_window_focused = imgui.is_window_focused() or state.is_window_focused
        imgui.end_child()

        imgui.begin_child("Number of Spherical Gaussians", 280, 50, border=True)
        #Slider for the number of spherical gaussians
        imgui.text("Number of Spherical Gaussians")
        changed, state.params.num_sg = imgui.slider_int("", state.params.num_sg, 0, state.params.max_sg_display)
        if imgui.is_item_active():
            state.which_item = 11
        state.is_window_focused = imgui.is_window_focused() or state.is_window_focused
        imgui.end_child()

    state.is_window_focused = state.is_window_focused  or imgui.is_window_focused()  

    ##############################################################################

    ############################  Sous-menu 4 #############################################
    ##################  Affichage des gaussiennes ajoutées ################################
    if False:
        if state.gui_mode:
            sub_menu_gaussians_expanded = True
            sub_menu_gaussians_expanded, _ = imgui.collapsing_header("Affichage des gaussiennes ajoutées", sub_menu_gaussians_expanded)
            if sub_menu_gaussians_expanded:
                #Tick button to only show added gaussians
                state.update_added_gaussians, state.show_added_gaussians = imgui.checkbox("Show added gaussians", state.show_added_gaussians)
                if imgui.is_item_active():
                    state.which_item = 7
                #if the tick button is checked add a slider
                if state.show_added_gaussians:
                    imgui.begin_child("Added Gaussians before iteration", 280, 50, border=True)
                    imgui.text("Added gaussian before this iteration:")
                    added_gaussian_slider_update, state.added_before_iter = imgui.slider_int("", state.added_before_iter, 0, state.max_iteration)
                    state.update_added_gaussians = state.update_added_gaussians or added_gaussian_slider_update
                    if imgui.is_item_active():
                        state.which_item = 8

                    state.is_window_focused = imgui.is_window_focused() or state.is_window_focused
                    imgui.end_child()

    ##############################################################################
    
    ############################  Sous-menu 5 #############################################
    ##################  Choix de la caméra: First person ou Trackball ####################
    sub_menu_camera_expanded = True
    sub_menu_camera_expanded, _ = imgui.collapsing_header("Choix de la caméra", sub_menu_camera_expanded)
    if sub_menu_camera_expanded:
        imgui.begin_child("Camera choice", 280, 60, border=True)
        #Radio button to choose the camera
        if imgui.radio_button("Trackball", state.camera_mode == 0):
            state.camera_mode = 0
            state.which_item = 9
            ##############################
            #Look at is the projection barycenter on the direction of the camera
            direction = state.camera.look_at - state.camera.eye
            direction = direction / np.linalg.norm(direction)
            state.camera.look_at = state.camera.eye + np.dot(state.barycenter - state.camera.eye, direction) * direction
            #Alternative: look at is the barycenter
            # state.camera.look_at[:] = state.barycenter[:] #Careful to not copy the reference but the values
            state.trackball.reinitialize_orientation_from_camera()
            ##############################
            state.fp_controls.keys_pressed.clear()
        #On the same line add the radio button for the trackball
        imgui.same_line()
        if imgui.radio_button("First person", state.camera_mode == 1):
            state.camera_mode = 1
            state.which_item = 9
        #Add a button with "Reset Global Up direction" to reset the global up direction
        if imgui.button("Reset global up direction"):
            _,v,_=state.camera.uvw_frame()
            state.fp_controls.camera.up = normalize(+v)
        state.is_window_focused = imgui.is_window_focused() or state.is_window_focused
        imgui.end_child()
    ##############################################################################
    
    ############################  Sous-menu 6 #############################################
    ##################  Affichage de la profondeur ####################################
    sub_menu_depth_expanded = True
    sub_menu_depth_expanded, _ = imgui.collapsing_header("Affichage de la profondeur", sub_menu_depth_expanded)
    if sub_menu_depth_expanded:
        imgui.begin_child("Depth display", 280, 50, border=True)
        if state.print_depth:
            #Darker than the default imgui.COLOR_BUTTON
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.2599*0.5,0.58999*0.5,0.98*0.5,0.4)
        else:
            #Default imgui.COLOR_BUTTON
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.2599,0.58999,0.98,0.4)
        if imgui.button("Depth Image"):
            #Switch to the error image
            state.print_depth=not state.print_depth
            if state.print_depth:
                state.print_error_image = False
                state.print_gt_image= False
        if imgui.is_item_active():
            state.which_item = 10
        imgui.pop_style_color()
        state.is_window_focused = imgui.is_window_focused() or state.is_window_focused
        imgui.end_child()


    imgui.end()

    ######################################################################################
    ############################## Train set Image ########################################
    # imgui.set_next_window_position(10.0, 250.0)  # Adjust the y position as needed
    # imgui.set_next_window_size(300, 50)  # Ensure window has a size
    # imgui.begin("Train set Image", False, imgui.WINDOW_NO_RESIZE)
    # display_stats.update_train_im, display_stats.idx_train_im = imgui.slider_int("", display_stats.idx_train_im, display_stats.min_idx_train_im, display_stats.max_idx_train_im)
    # state.is_window_focused = imgui.is_item_active() or state.is_window_focused  # Check if the slider is active
    # imgui.end()
    ######################################################################################
    ############################## Error Image ###########################################
    # imgui.set_next_window_position(10.0, 300.0)
    # #Make a button that switch to the error image
    # imgui.set_next_window_size(300, 50)
    # imgui.begin("Error Image", False, imgui.WINDOW_NO_RESIZE)
    # if imgui.button("Error Image"):
    #     #Switch to the error image
    #     display_stats.switch_error_image = True
    # #Add a tick button to choose log scale
    # imgui.same_line()
    # changed, state.log_scale_error = imgui.checkbox("Log scale", state.log_scale_error)
    # # print("Log scale: ", state.log_scale_error)
    # # print("Changed: ", changed)
    # imgui.end()
    ######################################################################################
    ############################## GT Image ##############################################
    # imgui.set_next_window_position(10.0, 350.0)
    # #Make a button that switch to the error image
    # imgui.set_next_window_size(300, 50)
    # imgui.begin("GT Image", False, imgui.WINDOW_NO_RESIZE)
    # if imgui.button("GT Image"):
    #     #Switch to the error image
    #     display_stats.switch_gt_image = True
    # # print("Switch to groundtruth image: ", display_stats.switch_error_image)
    # imgui.end()
    #######################################################################################

    imgui.render()

    imgui.end_frame()

    display_stats.total_subframe_count += 1

    return do_update
