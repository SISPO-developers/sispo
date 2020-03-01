import setuptools

try:
    with open("README.md", "r") as fh:
        long_description = fh.read()
except Exception:
    long_description = "README file could not be opened"

setuptools.setup(
    name="sispo",
    version="0.1.1",
    description="Space Imaging Simulator for Proximity Operations",
    long_description=long_description,
    platforms=["Windows"],
    url="https://github.com/YgabrielsY/sispo",
    author="Gabriel J. Schwarzkopf, Mihkel Pajusalu",
    license="BSD 2-Clause",

    # Install package and all subpackages, ignore other folders
    packages=setuptools.find_namespace_packages(
        include=[
            "sispo",
            "sispo.sim",
            "sispo.compression",
            "sispo.reconstruction"
        ],
        exclude=[
            "*test*",
            "*software*",
            "*build*",
            "*doc*",
            "*.vs*",
            "*.vscode*",
            "*.mypy_cache*"
        ]
    ),
    # Check dependencies
    install_requires=[
        "astropy",
        "numpy",
        "opencv-contrib-python",
        "openexr",
        "orekit"
    ],
    entry_points={
        "console_scripts": [
            "sispo = sispo:main"
        ]
    },
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 2 - Pre-Alpha"
    ],

    command_options={
        'build_sphinx': {
            'source_dir': ('setup.py', 'doc/source'),
            'build_dir': ('setup.py', 'doc/build')
        }
    }
)
