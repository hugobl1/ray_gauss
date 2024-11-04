import collections
from gui_utils.trackball import Trackball
from gui_utils.fps_controls import FPCameraControls
import optix as ox
import numpy as np
#------------------------------------------------------------------------------
# Local types
#------------------------------------------------------------------------------

class Params:
    _params = collections.OrderedDict([
            ('max_prim_slice', 'u4'),
            ('degree_sh',        'u4'),
            ('max_sh_degree',    'u4'),
            ('num_sg',           'u4'),
            ('max_sg_display',   'u4'),
            ('bbox_min',         '3f4'),
            ('bbox_max',         '3f4'),
            ('densities',        'u8'),
            ('color_features',   'u8'),
            ('sph_gauss_features','u8'),
            ('bandwidth_sharpness','u8'),
            ('lobe_axis',        'u8'),
            ('positions',        'u8'),
            ('scales',           'u8'),
            ('quaternions',      'u8'),
            ('hit_sphere_idx',   'u8'),
            ('frame_buffer',   'u8'),
            ('depth_buffer_ptr',   'u8'),
            ('width',          'u4'),
            ('height',         'u4'),
            ('eye',            '3f4'),
            ('u',              '3f4'),
            ('v',              '3f4'),
            ('w',              '3f4'),
            ('trav_handle',    'u8'),
            ('subframe_index', 'i4'),
        ])

    def __init__(self):
        self.handle = ox.LaunchParamsRecord(names=tuple(self._params.keys()),
                                            formats=tuple(self._params.values()))

    def __getattribute__(self, name):
        if name in Params._params.keys():
            item = self.__dict__['handle'][name]
            if isinstance(item, np.ndarray) and item.shape in ((0,), (1,)):
                return item.item()
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in Params._params.keys():
            self.handle[name] = value
        elif name in {'handle'}:
            super().__setattr__(name, value)
        else:
            raise AttributeError(name)

    def __str__(self):
        return '\n'.join(f'{k}:  {self.handle[k]}' for k in self._params)

def init_main_params(params, max_prim_slice, degree_sh,max_sh_degree,num_sg,max_sg_display,
                     bbox_min,bbox_max,densities,color_features,sph_gauss_features,bandwidth_sharpness,
                     lobe_axis,positions,scales,quaternions,width,height,hit_sphere_idx, depth_buffer_ptr):
    params.max_prim_slice = max_prim_slice
    params.degree_sh = degree_sh
    params.max_sh_degree = max_sh_degree
    params.num_sg = num_sg
    params.max_sg_display = max_sg_display
    params.bbox_min = bbox_min
    params.bbox_max = bbox_max
    params.densities = densities.data.ptr
    params.color_features = color_features.data.ptr
    params.sph_gauss_features = sph_gauss_features.data.ptr
    params.bandwidth_sharpness = bandwidth_sharpness.data.ptr
    params.lobe_axis = lobe_axis.data.ptr
    params.positions = positions.data.ptr
    params.scales = scales.data.ptr
    params.quaternions = quaternions.data.ptr
    params.width = width
    params.height = height
    params.hit_sphere_idx = hit_sphere_idx.data.ptr
    params.depth_buffer_ptr = depth_buffer_ptr.data.ptr


class GeometryState:
    __slots__ = ['params', 'time', 'ctx', 'module', 'pipeline', 'pipeline_opts',
            'raygen_grp', 'miss_grp', 'hit_grp', 'sbt',
            'gas', #'build_input',
            'camera_mode','trackball', 'fp_controls',
            'camera_changed', 'mouse_button', 'resize_dirty', 'minimized','is_window_focused',
            'barycenter',
            'path_available_iterations','available_iterations','current_iteration',
            'slider_train_iteration','min_iteration','max_iteration',
            'idx_train_im','idx_test_im','update_train_im','update_test_im',
            'min_idx_test_im','max_idx_test_im','min_idx_train_im','max_idx_train_im',
            'gt_image', 'depth_buffer',
            'slider_used','print_error_image','print_gt_image','print_depth',
            'log_scale_error',
            'show_added_gaussians','update_added_gaussians', 'added_before_iter',
            'train_cam_infos','test_cam_infos',
            'train_images','test_images',
            'pointcloud',
            'which_item','gui_mode']

    def __init__(self):
        for slot in self.__slots__:
            setattr(self, slot, None)
        self.params = Params()

        self.camera_mode = 0 # 0 for trackball, 1 for FP camera
        self.trackball = Trackball()
        self.fp_controls = FPCameraControls()
        #FP controls camera is the same as trackball camera
        self.fp_controls.camera = self.trackball.camera

        self.camera_changed = True
        self.mouse_button = -1
        self.resize_dirty = False
        self.minimized = False
        self.is_window_focused = False

        self.current_iteration = -1
        self.slider_train_iteration = 0
        self.min_iteration = 0
        self.max_iteration = 0

        self.idx_train_im = 0
        self.idx_test_im = 0
        self.update_train_im = False
        self.update_test_im = False

        self.min_idx_test_im = 0
        self.max_idx_test_im = 0
        self.min_idx_train_im = 0
        self.max_idx_train_im = 0

        self.slider_used = False
        self.print_error_image = False
        self.print_gt_image = False
        self.print_depth = False

        self.log_scale_error = False

        self.show_added_gaussians = False
        self.update_added_gaussians = False
        self.added_before_iter = 0
        
        self.train_cam_infos = []
        self.test_cam_infos = []
        self.train_images = []
        self.test_images = []
        self.pointcloud = None

        self.which_item = -1

    @property
    def camera(self):
        if self.camera_mode == 0:
            return self.trackball.camera
        else:
            return self.fp_controls.camera
        #Aniway that should be the same
        # return self.trackball.camera

    @property
    def launch_dimensions(self):
        return (int(self.params.width), int(self.params.height))
