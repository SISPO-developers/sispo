from pathlib import Path
import re
import setuptools

here = Path(__file__).parent.resolve()


def read(*parts):
    with open(Path(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def find_readme(*file_paths):
    readme_file = read(*file_paths)

    if readme_file:
        return readme_file
    raise RuntimeError("Unable to find README.md file.")


setuptools.setup(
    name="sispo",
    version=find_version("sispo", "__init__.py"),
    description="Space Imaging Simulator for Proximity Operations",
    long_description=find_readme("README.md"),
    platforms=["Windows"],
    url="https://github.com/YgabrielsY/sispo",
    author="Gabriel J. Schwarzkopf, Mihkel Pajusalu",
    license="GNU General Public License v3",
    # Install package and all subpackages, ignore other folders
    packages=setuptools.find_namespace_packages(
        include=[
            "sispo",
            "sispo.sim",
            "sispo.compression",
            "sispo.reconstruction",
            "sispo.plugins",
        ],
        exclude=[
            "*test*",
            "*software*",
            "*build*",
            "*doc*",
            "*.vs*",
            "*.vscode*",
            "*.mypy_cache*",
        ],
    ),
    # Check dependencies
    install_requires=[
        "astropy",
        "numpy",
        "opencv-contrib-python",
        "openexr",
        "orekit",
        "mathutils",
    ],
    entry_points={"console_scripts": ["sispo = sispo:main"]},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 2 - Pre-Alpha",
        "Natural Language :: English",
    ],
    command_options={
        "build_sphinx": {
            "source_dir": ("setup.py", "doc/source"),
            "build_dir": ("setup.py", "doc/build"),
        }
    },
)
