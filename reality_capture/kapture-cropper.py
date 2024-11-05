# py kapture-cropper.py -i dataset-kapture\ --border_px 0 --scale_factor 1 -v

import kapture
import kapture.io.csv as csv
from PIL import Image
from kapture.io.csv import kapture_to_dir
import os, logging, argparse
import kapture.utils.logging

logger = logging.getLogger("kapture-cropper")

DEFAULT_FOCAL_LENGTH_FACTOR = 1.2
EPSILON = 1e-5


def crop_command_line() -> None:
    """
    Crop a set of raster images from a kapture project, adapting their intrinsics using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(
        description="Crop a set of raster images from a kapture project, adapting their intrinsics"
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="input path to kapture project"
    )
    parser.add_argument(
        "-b",
        "--border_px",
        type=int,
        required=False,
        default=0,
        help="Crop amount in pixels, for each side the same",
    )
    parser.add_argument(
        "-s",
        "--scale_factor",
        type=check_positive_strictly,
        required=False,
        default=1,
        help="Rescale factor, to change the coordinate system of cams and points3d",
    )
    # Logging
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        "-v",
        "--verbose",
        nargs="?",
        default=logging.WARNING,
        const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help="verbosity level (debug, info, warning, critical, ... or int value) [warning]",
    )
    parser_verbosity.add_argument(
        "-q",
        "--silent",
        "--quiet",
        action="store_const",
        dest="verbose",
        const=logging.CRITICAL,
    )

    # Parse args and do the crop
    args = parser.parse_args()
    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:  # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
    kapture_crop(args.input, args.border_px, args.scale_factor)


def check_positive_strictly(value):
    try:
        value = float(value)
        if value <= 0:
            raise argparse.ArgumentTypeError(
                "{} is not a positive number".format(value)
            )
    except ValueError:
        raise Exception("{} is not an integer".format(value))
    return value


def rescale_kapture_coords(kapture_data, scale_factor):
    # Rescale sensors' trajectories
    trajectories = kapture_data.trajectories
    for traj_idx in trajectories:
        for poseTransform in trajectories[traj_idx].values():
            poseTransform.rescale(scale_factor)
    # rescale points3d observations
    points3d = kapture_data.points3d
    points3d[:, :3] *= scale_factor


def kapture_crop(kapture_dir_path, border_px=0, scale_factor=1):
    # Create dir and move other files
    if border_px != 0:
        image_dir_path = os.path.join(kapture_dir_path, "sensors/records_data")
        original_dir_path = os.path.join(
            kapture_dir_path, "sensors/records_data_original"
        )
        if os.path.exists(original_dir_path):
            logger.error(
                f"{original_dir_path} already exists, you probably already applied the cropping"
            )
            return
        os.rename(image_dir_path, original_dir_path)
        os.makedirs(image_dir_path, exist_ok=True)
    #
    # with csv.get_all_tar_handlers(kapture_dir_path) as tar_handlers:
    pairsfile_path = None
    logger.info(f"Reading Kapture project from {kapture_dir_path}...")
    kapture_data = csv.kapture_from_dir(
        kapture_dir_path,
        pairsfile_path,
        # tar_handlers=tar_handlers
    )

    records_camera = kapture_data.records_camera
    sensors = kapture_data.sensors
    logger.info(
        f"Found {len(records_camera)} records_camera and {len(sensors)} sensors"
    )
    #
    # Could do the conversion in place if no crop via PILLOW
    # converted_sensors = [
    #     (cam_id, get_cropped_sensor(cam, border_px)) for cam_id, cam in sensors.items()
    # ]
    # cropped_kapture = kapture.Kapture(sensors=converted_sensors, records_camera=records_camera)
    #
    # Loop for image crop via pillow + sensor crop via kapture
    logger.info("Cropping image rasters + computing cropped kapture sensors...")
    # handler = logging.StreamHandler()
    for handler in logger.handlers:
        handler.terminator = "\r"
    for cam_id in records_camera:  # cam = records_camera[0]
        cam = records_camera[cam_id]
        for (
            sensor_id,
            img_fp,
        ) in cam.items():  # sensor_id, img_fp = list(cam.items())[0]
            sensor = sensors[sensor_id]
            logger.info(
                f"Cropping image with id {sensor_id} and fp {img_fp}, sensor {sensor}",
                # end="\r",
            )
            # Always simplify sensor model if radial distortion coefficients are zero
            cropped_sensor = simplify_sensors_model(sensor)
            # Crop sensor if user asks for it
            if border_px != 0:
                cropped_sensor = get_cropped_sensor(cropped_sensor, border_px)
                # Crop via Pillow
                with Image.open(os.path.join(original_dir_path, img_fp)) as im:
                    width, height = im.size
                    left, top, right, bottom = (
                        border_px,
                        border_px,
                        width - border_px,
                        height - border_px,
                    )
                    cropped_im = im.crop((left, top, right, bottom))
                    cropped_im.save(os.path.join(image_dir_path, img_fp))
            #
            # write back that cropped sensor to the current kapture db
            sensors[sensor_id] = cropped_sensor
    # Rescale coordinate system
    if scale_factor != 1:
        rescale_kapture_coords(kapture_data, scale_factor)
    # Export to disk
    for handler in logger.handlers:
        handler.terminator = "\n"
    logger.info("\nImage raster files cropped, Writing cropped data to Kapture file...")
    cropped_kapture = kapture.Kapture(
        sensors=sensors,
        records_camera=records_camera,
        trajectories=kapture_data.trajectories,
        points3d=kapture_data.points3d,
    )
    kapture_to_dir(kapture_dir_path, cropped_kapture)


# Adapted from get_colmap_camera in https://github.com/naver/kapture/blob/e58e244f35fe8db47dbb2b149178456f513ef6f8/kapture/converter/colmap/cameras.py#L42
def get_cropped_sensor(camera: kapture.Camera, border: int):
    """
    Compute the cropped camera definition - uniform border width given in pixels
    #
    :param camera: a kapture camera definition
    :param border: a pixel crop count
    :return: cropped camera parameters.
    """
    assert isinstance(camera, kapture.Camera)
    assert len(camera.camera_params) >= 2
    #
    old_width = camera.camera_params[0]
    old_height = camera.camera_params[1]
    #
    # Update width with cropped value (in pixels units)
    width = old_width - 2 * border
    height = old_height - 2 * border
    # will apply to cx,cy and focal-length
    if camera.camera_type in [
        kapture.CameraType.SIMPLE_PINHOLE,
        kapture.CameraType.SIMPLE_RADIAL,
        kapture.CameraType.RADIAL,
    ]:
        # [SIMPLE_]RADIAL params: w, h, f, cx, cy, k1 [, k2]
        params = camera.camera_params[2:]
        params[0] *= width / old_width  # focal f
        params[1] -= border  # cx
        params[2] -= border  # cy
    elif camera.camera_type in [
        kapture.CameraType.PINHOLE,
        kapture.CameraType.OPENCV,
        kapture.CameraType.FULL_OPENCV,
    ]:
        # PINHOLE/OPENCV params: w, h, fx, fy, cx, cy, k1 [, k2]
        params = camera.camera_params[2:]
        params[0] *= width / old_width  # focal f
        params[1] *= width / old_width  # focal f
        params[2] -= border  # cx
        params[3] -= border  # cy
    else:
        raise ValueError(
            f"This sensor model: {camera.camera_type} is not supported by the intrinsics img cropper yet"
        )
    return kapture.Camera(camera.camera_type, [width, height, *params])


# The following table stores distortion_idx and camera_type for a given input camera_type for the simplified version
# { input_camera_type: (distortion_idx, simplified_camera_type) }
# [SIMPLE_]RADIAL params: w, h, f, cx, cy | k1 [, k2]
# [FULL_]OPENCV params: w, h, fx, fy, cx, cy | k1, k2, p1, p2 [, k3, k4, k5, k6]
#
# See https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h
# UNKNOWN_CAMERA w, h
# SIMPLE_PINHOLE w, h, f, cx, cy
# PINHOLE w, h, fx, fy, cx, cy
# SIMPLE_RADIAL  w, h, f, cx, cy, k
# RADIAL w, h, f, cx, cy, k1, k2
# OPENCV w, h, fx, fy, cx, cy, k1, k2, p1, p2
# FULL_OPENCV w, h, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
SENSOR_SIMPLIFICATION_TABLE = {
    kapture.CameraType.SIMPLE_RADIAL: (5, kapture.CameraType.SIMPLE_PINHOLE),
    kapture.CameraType.RADIAL: (5, kapture.CameraType.SIMPLE_PINHOLE),
    kapture.CameraType.OPENCV: (6, kapture.CameraType.PINHOLE),
    kapture.CameraType.FULL_OPENCV: (6, kapture.CameraType.PINHOLE),
}


def simplify_sensors_model(camera: kapture.Camera):
    """
    Convert the sensor to pinhole if it is a radial which has zero k1 k2 distortion coefficients
    #
    :param camera: a kapture camera definition
    :return: cropped camera parameters.
    """
    assert isinstance(camera, kapture.Camera)
    assert len(camera.camera_params) >= 2
    #
    if camera.camera_type in SENSOR_SIMPLIFICATION_TABLE:
        distortion_idx, camera_type = SENSOR_SIMPLIFICATION_TABLE[camera.camera_type]
    else:
        raise ValueError(
            f"This sensor model: {camera.camera_type} is not supported by the sensor simplifier yet"
        )
    # Distangle pinhole params from distortion params, check if distortion is zero, otherwise return original camera
    pinhole_params = camera.camera_params[:distortion_idx]  # w, h, f[x, fy], cx, cy
    distortion_params = camera.camera_params[distortion_idx:]  # k...
    if sum([x**2 for x in distortion_params]) <= EPSILON:
        camera_params = pinhole_params
        return kapture.Camera(camera_type, camera_params)
    else:
        return camera


if __name__ == "__main__":
    crop_command_line()