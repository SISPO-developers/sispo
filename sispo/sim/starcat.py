"""
Interface for handling data from a star catalogue. Retrieve data as well as
render and write images.
"""

import subprocess
import sys
from pathlib import Path

import astropy.coordinates as coord
import astropy.units as u
from astroquery.vizier import Vizier


class StarCatalogError(RuntimeError):
    """Generic error for star catalog module."""
    pass


class StarCatalog:
    """Class to access star catalogs and render stars."""

    def __init__(self, res_dir, ext_logger, starcat_dir=None):
        """."""
        self.logger = ext_logger

        if starcat_dir is None:
            self.catalog = Vizier(catalog="UCAC4", row_limit=1000000)
        else:
            self.root_dir = Path(__file__).parent.parent.parent

            starcat_dir = Path(starcat_dir)

            try:
                starcat_dir = starcat_dir.resolve()
            except OSError as e:
                raise StarCatalogError(e)

            if not starcat_dir.is_dir():
                raise StarCatalogError("Given star cat dir does not exist.")
            self.starcat_dir = starcat_dir

            self.res_dir = res_dir

            exe_dir = self.root_dir / "software" / "star_cats"

            if (exe_dir / "u4test").is_file() or (exe_dir / "u4test.exe").is_file():
                self.exe = exe_dir / "u4test"
            elif ((exe_dir / "star_cats" / "u4test").is_file() or
                    (exe_dir / "star_cats" / "u4test.exe").is_file()):
                self.exe = exe_dir / "star_cats" / "u4test"
            elif ((exe_dir / "build_star_cats" / "u4test").is_file() or
                    (exe_dir / "build_star_cats" / "u4test.exe").is_file()):
                self.exe = exe_dir / "build_star_cats" / "u4test"
            else:
                raise StarCatalogError("UCAC4 interface could not be found.")

            # Don't display the Windows GPF dialog if the invoked program dies.
            # See comp.os.ms-windows.programmer.win32
            # How to suppress crash notification dialog?, Jan 14,2004 -
            # Raymond Chen"s response [1]
            if sys.platform.startswith("win"):
                self.logger.debug("Windows system, surrpressing GPF dialog.")
                import ctypes

                SEM_NOGPFAULTERRORBOX = 0x0002  # From MSDN
                ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX)
        
    def get_stardata(self, ra, dec, width, height, filename="ucac4.txt"):
        """."""
        if hasattr(self, "catalog"):
            return self.get_stardata_vizier(ra, dec, width, height)
        else:
            return self.get_stardata_ucac4(ra, dec, width, height, filename)

    def get_stardata_ucac4(self, ra, dec, width, height, filename="ucac4.txt"):
        """Retrieve star data from given field of view using UCAC4 catalog."""
        res_file = self.res_dir / filename
        res_file = res_file.with_suffix(".txt")

        command = [
            str(self.exe),
            str(ra),
            str(dec),
            str(width),
            str(height),
            "-h",
            str(self.starcat_dir),
            str(res_file)
        ]

        for _ in range(5):
            ret = subprocess.run(command)

            if ret.returncode > 0:
                break

            self.logger.debug("Error code from star cat %d", ret.returncode)

        with open(str(res_file), "r") as rfile:
            complete_data = rfile.readlines()

        star_data = []
        for line in complete_data[1:]:
            line_data = line.split()

            ra_star = float(line_data[1])
            dec_star = float(line_data[2])
            mag_star = float(line_data[3])

            star_data.append((ra_star, dec_star, mag_star))

        self.logger.debug("Found %d stars in catalog", len(star_data))

        return star_data

    def get_stardata_vizier(self, ra, dec, width, height):
        """
        Retrieves star data from Vizier
        """

        crds = coord.SkyCoord(
            ra=ra, dec=dec, unit=(u.deg, u.deg))
        result = self.catalog.query_region(
            crds, width=width*u.deg, height=height*u.deg)[0]

        star_data = zip(result['RAJ2000'], result['DEJ2000'], result['f.mag'])
        star_data = [(ra, de, mag) for ra, de, mag in star_data]

        star_res = []
        for star in star_data:
            if star[0] >= ra - width/2 and star[0] <= ra + width/2 and star[1] >= dec - height/2 and star[1] <= dec + height/2:
                star_res.append(star)

        return star_res
