"""Utils module contains functions possibly used by all modules."""

import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import OpenEXR
import Imath

def check_dir(directory):
    """Resolves directory and creates it, if it doesn't existing."""
    dir_resolved = directory.resolve()

    if not dir_resolved.exists():
        Path.mkdir(dir_resolved)

    return dir_resolved


def read_vec_string(string):
    """Converts vector string into numpy array."""
    string = string.strip("[]")
    string = string.split(",")
    vec = np.asarray(string, dtype=np.float64)

    return vec


def write_vec_string(vec, prec):
    """Write data vector into string."""
    o = "["

    for (n, v) in enumerate(vec):
        o += f"{v:.{prec}f}"
        if n < len(vec) - 1:
            o += ","

    return o + "]"


def read_mat_string(string):
    """Converts matrix string into numpy array."""
    string = string.strip("[]")
    string = string.split("],[")

    mat = []
    for elem in string:
        mat.append(read_vec_string(elem))

    mat = np.asarray(mat, dtype=np.float64)

    return mat

def write_mat_string(mat, prec):
    """Write data matrix into string."""
    o = "["

    for (n, m) in enumerate(mat):
        o += (write_vec_string(m, prec))
        if n < len(mat) - 1:
            o += ","

    return o + "]"


def serialise(o):
    """Serialise numpy arrays to json object."""
    try:
        return np.asarray(o, dtype="float64").tolist()
    except TypeError:
        try:
            return float(o)
        except TypeError:
            return str(o)


def read_openexr_image(filename):
    """Read image in OpenEXR file format."""
    filename = str(filename)

    file_extension = ".exr"
    if filename[-len(file_extension):] != file_extension:
        filename += file_extension

    if not OpenEXR.isOpenExrFile(filename):
        return None

    image = OpenEXR.InputFile(filename)

    if not image.isComplete():
        return None

    header = image.header()
    
    size = header["displayWindow"]
    resolution = (size.max.x - size.min.x + 1, size.max.y - size.min.y + 1)

    ch_info = header["channels"]
    if "R" in ch_info and "G" in ch_info and "B" in ch_info:
        if "A" in ch_info:
            channels = 4
        else:
            channels = 3
    else:
        return None
    
    image_o = np.zeros((resolution[1],resolution[0],channels), np.float32)
    
    ch = ["R", "G", "B", "A"]
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    for c in range(0, channels):
        image_channel = np.fromstring(image.channel(ch[c], pt), dtype=np.float32)
        image_o[:, :, c] = image_channel.reshape(resolution[1], resolution[0])
    
    image.close()

    return image_o


def write_openexr_image(filename, image):
    """Save image in OpenEXR file format."""
    filename = str(filename)

    file_extension = ".exr"
    if filename[-len(file_extension):] != file_extension:
        filename += file_extension

    height = len(image)
    width = len(image[0])
    channels = len(image[0][0])

    # Default header only has RGB channels
    hdr = OpenEXR.Header(width, height)

    if channels == 4:
        data_r = image[:, :, 0].tobytes()
        data_g = image[:, :, 1].tobytes()
        data_b = image[:, :, 2].tobytes()
        data_a = image[:, :, 3].tobytes()
        image_data = {"R": data_r, "G": data_g, "B": data_b, "A": data_a}
        # Add alpha channel to header
        alpha_channel = {"A": Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))}
        hdr["channels"].update(alpha_channel)

    elif channels == 3:
        data_r = image[:, :, 0].tobytes()
        data_g = image[:, :, 1].tobytes()
        data_b = image[:, :, 2].tobytes()
        image_data = {"R": data_r, "G": data_g, "B": data_b}
    else:
        raise RuntimeError("Invalid number of channels of starmap image.")

    file_handler = OpenEXR.OutputFile(filename, hdr)
    file_handler.writePixels(image_data)
    file_handler.close()


def create_logger(name):
    """Creates a logger with the common formatting."""
    now = datetime.now().strftime("%Y-%m-%dT%H%M%S%z")
    file_name = (now + "_" + name + ".log")
    log_dir = Path(__file__).parent.parent / "data" / "logs"
    log_dir = check_dir(log_dir)
    log_file = log_dir / file_name

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(funcName)s - %(message)s")
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logger_formatter)
    logger.addHandler(file_handler)
    logger.info("\n\n#################### NEW LOG ####################\n")

    return logger


if __name__ == "__main__":
    print("Starting opencv vs skimage benchmarking")
    
    import timeit

    setup = "#gc.enable()"
    setup += "\n" + "import numpy as np"
    setup += "\n" + "import utils"
    setup += "\n" + "from pathlib import Path"
    setup += "\n" + "path = Path('.').resolve()"
    setup += "\n" + "path = path / '..' / 'data' / 'results' / 'Didymos' / 'rendering'"
    setup += "\n" + "file = path / 'Composition_2017-08-15T115840-817000.exr'"
    setup += "\n" + "file = file.resolve()"
    setup += "\n" + "img = utils.read_openexr_image(str(file))"
    setup += "\n" + "sigma = 5"
    setup += "\n" + "kernel = 5"

    setup_skimage = setup + "\n" + "truncation = (kernel - 1) / 2 / sigma" \
                    + "\n" + "from skimage.filters import gaussian"
    cmd_skimage = "sk_image = np.zeros(img.shape, dtype=np.float32)"
    cmd_skimage += "\n" + "sk_image = gaussian(img, sigma, truncate=truncation, multichannel=True)"

    setup_cv2 = setup + "\n" + "from cv2 import GaussianBlur"
    cmd_cv2 = "cv_image = np.zeros(img.shape, dtype=np.float32)"
    cmd_cv2 += "\n" + "cv_image = GaussianBlur(img,(kernel,kernel),sigma,sigma)"

    iterations = 10000
    times_sk = timeit.timeit(cmd_skimage, number=iterations, setup=setup_skimage)
    times_cv2 = timeit.timeit(cmd_cv2, number=iterations, setup=setup_cv2)
    
    print(f"skimage: {times_sk / iterations} s")
    print(f"opencv: {times_cv2 / iterations} s")
    print(f"Ratio Skimage/opencv: {times_sk / times_cv2}")

    cmd_skimage += "\n" + "print('SK type: ', sk_image.dtype)"
    cmd_cv2 += "\n" + "print('CV2 type: ', cv_image.dtype)"

    exec(setup_skimage + "\n" + cmd_skimage + "\n" + "utils.write_openexr_image(str(file) + '_sk', sk_image)")
    exec(setup_cv2 + "\n" + cmd_cv2 + "\n" + "utils.write_openexr_image(str(file) + '_cv', cv_image)")


    statistics = setup_skimage + "\n" + setup_cv2 + "\n" + cmd_skimage + "\n" + cmd_cv2
    statistics += "\n" + "diff = sk_image - cv_image"
    statistics += "\n" + "print('Diff min, max: ', np.min(diff), np.max(diff))"
    statistics += "\n" + "print('Alpha min sk, cv; max, sk, cv: ', np.min(sk_image[:,:,3]), np.min(cv_image[:,:,3]), np.max(sk_image[:,:,3]), np.max(cv_image[:,:,3]))"
    exec(statistics)
