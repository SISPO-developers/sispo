"""
The compositor module combines the different output files of the simulation.

As the simulation module outputs different files for background and foreground
and because the intensity of the blender rendered images are not constant, the
compositor is required to fix the intensity issue and add the star background.
"""

from datetime import datetime
import json

from astropy import constants as const
from astropy import units as u
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import utils

logger = utils.create_logger("compositor")

#Astrometric calibrations 
#https://www.cfa.harvard.edu/~dfabricant/huchra/ay145/mags.html
FLUX0_VBAND = 3640 * 1.51E7 * 0.16 * u.ph / (u.s * u.m ** 2) # Photons per m^2
SUN_MAG_VBAND = -26.74 * u.mag # 1 AU distance
SUN_FLUX_VBAND_1AU = np.power(10., -0.4 * SUN_MAG_VBAND.value) * FLUX0_VBAND

class ImageCompositorError(RuntimeError):
    """This is a generic error for the compositor."""
    pass


class Frame():
    """Class to wrap all data of a single frame."""

    metadata = None
    main_scene = None
    stars = None
    sssb_only = None
    sssb_const_dist = None
    light_ref = None

    def __init__(self,
                 frame_id,
                 image_dir=None,
                 main=None,
                 stars=None,
                 sssb_only=None,
                 sssb_const_dist=None,
                 light_ref=None):

        self.id = frame_id

        if None not in (main, stars, sssb_only, sssb_const_dist, light_ref):
            self.main_scene = main
            self.stars = stars
            self.sssb_only = sssb_only
            self.sssb_const_dist = sssb_const_dist
            self.light_ref = light_ref

        elif image_dir is not None:
            self.read_complete_frame(self.id, image_dir)

        else:
            raise ImageCompositorError("Unable to create frame.")

    def calc_ref_intensity(self):
        """Calculates reference intensitiy using the light reference scene."""

        (height, width, _) = self.light_ref.shape
        h_slice = (height // 2 - 35, height // 2 + 35)
        w_slice = (width // 2 - 35, width // 2 + 35)

        area = self.light_ref[h_slice[0]:h_slice[1], w_slice[0]:w_slice[1], 0]
        intensities = np.mean(area)
        return intensities

    def calc_stars_stats(self):
        """Calculate star scene parameters."""
        star_c_max = []
        star_c_sum = []

        for i in range(3):
            star_c_max.append(np.max(self.stars[:, :, i]))
            star_c_sum.append(np.sum(self.stars[:, :, i]))

        return (star_c_max, star_c_sum)

    def calc_sssb_stats(self, const_dist=False):
        """Calculate SSSB max and sum corrected with alpha channel.

        If const_dist is True, stats of const distant images are calculated.
        """
        if const_dist:
            sssb_max = np.max(
                self.sssb_const_dist[:, :, 0] * self.sssb_const_dist[:, :, 3])
            sssb_sum = np.sum(
                self.sssb_const_dist[:, :, 0] * self.sssb_const_dist[:, :, 3])
        else:
            sssb_max = np.max(
                self.sssb_only[:, :, 0] * self.sssb_only[:, :, 3])
            sssb_sum = np.sum(
                self.sssb_only[:, :, 0] * self.sssb_only[:, :, 3])

        return (sssb_max, sssb_sum)

    def read_complete_frame(self, frame_id, image_dir):
        """Reads all images for a given frame id.

        This includes MainScene, Stars, SssbOnly, SssbConstDist, and LightRef.
        """
        frame_fmt_str = image_dir / ("{}_" + frame_id + ".exr")
        frame_fmt_str = str(frame_fmt_str)

        self.metadata = self.read_meta_file(frame_id, image_dir)

        filename = frame_fmt_str.format("MainScene")
        self.main = utils.read_openexr_image(filename)

        filename = frame_fmt_str.format("Stars")
        self.stars = utils.read_openexr_image(filename)

        filename = frame_fmt_str.format("SssbOnly")
        self.sssb_only = utils.read_openexr_image(filename)
        
        filename = frame_fmt_str.format("SssbConstDist")
        self.sssb_const_dist = utils.read_openexr_image(filename)
        
        filename = frame_fmt_str.format("LightRef")
        self.light_ref = utils.read_openexr_image(filename)

    def read_meta_file(self, frame_id, image_dir):
        """Reads metafile of a frame."""
        filename = image_dir / ("Metadata_" + frame_id + ".json")

        with open(str(filename), "r") as metafile:
            metadata = json.load(metafile)

            date = datetime.strptime(metadata["date"], "%Y-%m-%dT%H%M%S-%f")
            metadata["date"] = date
            metadata["distance"] = metadata["distance"] * u.m

            metadata["sc_pos"] = np.asarray(metadata["sc_pos"]) * u.m
            metadata["sc_rel_pos"] = np.asarray(metadata["sc_rel_pos"]) * u.m
            metadata["sssb_pos"] = np.asarray(metadata["sssb_pos"]) * u.m

        return metadata


class ImageCompositor():
    """This class provides functions to combine the final simulation images."""

    def __init__(self, res_dir, filename=None):

        self.res_dir = res_dir
        self.image_dir = res_dir / "rendering"

        self.image_extension = ".exr"

        self.inst = {}
        self.inst["chip_noise"] = 10
        self.inst["pixel_l"] = 3.45 * u.micron
        self.inst["pixel_a"] = self.inst["pixel_l"] ** 2 * (1 / u.pix)
        self.inst["quantum_eff"] = 0.25
        self.inst["focal_l"] = 230 * u.mm
        self.inst["aperture_d"] = 4 * u.cm
        self.inst["aperture_a"] = ((2 * u.cm) ** 2 - (1.28 * u.cm) ** 2) \
                                  * np.pi / 4
        self.inst["wavelength"] = 550 * u.nm
        self.dlmult = 2

        self.sssb = {}
        self.sssb["albedo"] = 0.15
        self.sssb["max_dim"] = 512

        self.with_infobox = True

        self.frame_ids = self.get_frame_ids()
        self.frames = []
        
        for frame_id in self.frame_ids:
            new_frame = Frame(frame_id, self.image_dir)
            self.frames.append(new_frame)

        logger.info("Number of files: %d", len(self.frames))

        self.compose()

    def get_frame_ids(self):
        """Extract list of frame ids from file names of SssbOnly scenes."""
        scene_name = "SssbOnly"
        image_names = scene_name + "*" + self.image_extension
        filenames = self.image_dir.glob(image_names)

        ids = []
        for filename in filenames:
            filename = str(filename.name).strip(self.image_extension)
            filename = filename.strip(scene_name)
            ids.append(filename.strip("_"))

        return ids

    def calc_relative_intensity_curve(self):
        """Calculates the relative intensity curve for all sssb frames."""
        only_stats = np.zeros(len(self.frames))
        const_dist_stats = np.zeros(len(self.frames))
        distances = np.zeros(len(self.frames))

        for i, frame in enumerate(self.frames):
            only_stats[i] = frame.calc_sssb_stats()[1]
            const_dist_stats[i] = frame.calc_sssb_stats(True)[1]
            distances[i] = frame.metadata["distance"]

        rel_intensity = only_stats / const_dist_stats

        ind_sorted = distances.argsort()
        distances = distances[ind_sorted]
        rel_intensity = rel_intensity[ind_sorted]


        for last in range(len(distances)):
            if rel_intensity[last] == 0:
                break
        #last -= 1

        rel_intensity = rel_intensity[:last]

        return rel_intensity

    def compose(self):
        """Composes raw images and adjusts light intensities."""
        
        # Calculate Gaussian standard deviation for approx diffraction pattern
        sigma = self.dlmult * 0.45 * self.inst["wavelength"] \
                * self.inst["focal_l"] / (self.inst["aperture_d"] \
                * self.inst["pixel_l"])

        # SSSB reference when SSSB is too small
        resolution = self.frames[0].sssb_only.shape[0:2]
        sssb_ref_img = self.create_sssb_ref(resolution)

        for frame in self.frames:

            # SSSB photometry
            sc_sun_dist = np.linalg.norm(frame.metadata["sc_pos"]) * u.m
            ref_flux = SUN_FLUX_VBAND_1AU * ((const.au / sc_sun_dist) ** 2)
            ref_flux *= self.inst["aperture_a"] * self.inst["pixel_a"] 
            ref_flux /= ((self.inst["focal_l"] ** 2) * np.pi)
            ref_flux = ref_flux.decompose()

            # Star photometry
            starmap_flux = FLUX0_VBAND * frame.metadata["total_flux"]
            starmap_flux *= self.inst["aperture_a"]
            starmap_flux = starmap_flux.decompose()
            
            # Calibrate starmap
            (_, stars_sums) = frame.calc_stars_stats()
            frame.stars[:, :, 0:3] *= starmap_flux.value / stars_sums[0]

            # Create composition image array
            composed_img = np.zeros(frame.stars.shape, dtype=np.float32)

            # Calibrate SSSB, depending on visible size 
            dist_scale = np.power(1E6 * u.m / frame.metadata["distance"], 2.)  
            vis_dim = self.sssb["max_dim"] * dist_scale

            # Kernel size calculated to equal skimage.filters.gaussian
            # Reference:
            # https://github.com/scipy/scipy/blob/4bfc152f6ee1ca48c73c06e27f7ef021d729f496/scipy/ndimage/filters.py#L214
            kernel = int((4 * sigma + 0.5) * 2)
            kernel = max(kernel, 5) # Don't use smaller than 5
            ksize = (kernel, kernel)

            if vis_dim < 0.1:
                # Use point source sssb
                # Generate point source reference image
                sssb_ref = sssb_ref_img.copy()
                alpha = frame.sssb_const_dist[:, :, 3]
                scale = frame.sssb_const_dist[:, :, 0:3] * alpha 
                sssb_ref[:, :, 0:3] *= np.sum(scale, axis=-1) * dist_scale

                composed_img = (sssb_ref[:, : , 0:3] + frame.stars)
                composed_img *= self.inst["quantum_eff"]
                composed_img = cv2.GaussianBlur(composed_img, ksize, sigma)
                composed_img += np.random.poisson(composed_img)
                
                composed_max = np.max(composed_img)
                ref_sssb_max = np.max(sssb_ref[:, :, 0:3])
                if composed_max > ref_sssb_max * 5:
                    composed_max = ref_sssb_max * 5

            else:
                # Calibrate sssb images
                ref_int = frame.calc_ref_intensity()
                sssb_cal_factor = ref_flux * self.sssb["albedo"] / ref_int
                sssb_cal_factor = sssb_cal_factor.decompose().value
                frame.sssb_only[:, :, 0:3] *= sssb_cal_factor
                frame.sssb_const_dist[:, :, 0:3] *= sssb_cal_factor

                # Merge images taking alpha channel and q.e. into account
                alpha = frame.sssb_only[:, :, 3]
                for c in range(3):
                    sssb = frame.sssb_only[:, :, c]
                    stars = frame.stars[:, :, c]
                    composed_img[:, :, c] = alpha * sssb + (1 - alpha) * stars
                
                composed_img[:, :, 0:3] *= self.inst["quantum_eff"]
                composed_img = cv2.GaussianBlur(composed_img, ksize, sigma)
                composed_img += np.random.poisson(composed_img)
                composed_max = np.max(composed_img)

            composed_img[:, :, :] /= composed_max

            filename = self.image_dir / ("Comp_" + str(frame.id))
            utils.write_openexr_image(filename, composed_img)

            composed_img[:, :, :] *= 255
            composed_img = composed_img.astype(np.uint8)
            
            if self.with_infobox:
                self.add_infobox(composed_img, frame.metadata)
            
            filename = self.image_dir / ("Comp_" + str(frame.id) + ".png")
            cv2.imwrite(str(filename), composed_img)

    def create_sssb_ref(self, res, scale=5):
        """Creates a reference sssb image for calibration.
        
        Sort of natural look by using image increased by factor of scale,
        gaussian blur the result and decimate to match size of other images.
        opencv resize algorithm needs integer divisable number of pixels
        to have the same behaviour as skimage.transform.downscale_local_mean.
        Zero-padding of skimage.transform.downscale_local_mean would be 
        necessary without scaling.
        """
        res_x, res_y = res

        # Rescale
        res_x_sc = res_x * scale
        res_y_sc = res_y * scale
        sssb_point = np.zeros((res_x_sc, res_y_sc, 4), np.float32)

        sig = scale / 2.
        kernel = int((4 * sig + 0.5) * 2)
        ksize = (kernel, kernel)
        
        # Create point source and blur
        sssb_point[res_x_sc//2, res_y_sc//2, :] = [1., 1., 1., 1.]
        sssb_point = cv2.GaussianBlur(sssb_point, ksize, sig)

        sssb = np.zeros((res_x, res_y, 4), np.float32)
        sssb = cv2.resize(sssb_point, None, fx=1/scale, fy=1/scale,
                            interpolation=cv2.INTER_AREA)

        sssb *= (scale * scale)
        sssb[:, :, 0:3] /= np.sum(sssb[:, :, 0:3])
            
        return sssb

    def add_infobox(self, img, metadata, tb_height=100, tb_width=400):
        """Overlays an infobox to a given image in the lower right corner."""
        sig = 3
        textbox = np.zeros((tb_height * sig, tb_width * sig, 4), np.float32)

        pt1 = (0, 0)
        pt2 = (tb_width * sig, tb_height * sig)
        color = (128, 128, 128, 128)
        cv2.rectangle(textbox, pt1, pt2, color, cv2.FILLED)

        org_date = (10 * sig, 40 * sig)
        org_dist = (10 * sig, 70 * sig)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_size = 1.0 * sig
        color = (255, 255, 255, 255)
        date = str(metadata["date"])
        dist = str(metadata["distance"])
        cv2.putText(textbox, date, org_date, font, font_size, color, sig)
        cv2.putText(textbox, dist, org_dist, font, font_size, color, sig)

        # See link above for explanation
        sigma = sig / 2.
        kernel = int((4 * sigma + 0.5) * 2)
        ksize = (kernel, kernel)
        textbox = cv2.GaussianBlur(textbox, ksize, sigma)
        textbox = cv2.resize(textbox, (tb_width, tb_height), 
                                interpolation=cv2.INTER_AREA)
        alpha_s = textbox[:, :, 3] / 255.0
        alpha_l = 1. - alpha_s

        for c in range(3):
            img[1800:1800+tb_height, 2000:2000+tb_width, c] = (alpha_s * textbox[:, :, c] + alpha_l * img[1800:1800+tb_height, 2000:2000+tb_width, c])
        
        return img

if __name__ == "__main__":
    pass
