Installation instructions
=========================
SISPO can be setup with Linux and Windows. The default case used in this description is a Windows setup. It is recommended to set SISPO up in a Windows environment since e.g. the reconstruction algorithms seemed to be more stable. Known differences or problems under Linux will be pointed out. While it should be possible to use a plain Python environment and pip, a miniconda environment manager was used for development. Also a C compiler is necessary. Linux provides the GCC, for Windows it is easiest to install Microsoft Visual Studio with MSVC and MSBuild. Another possibility when using Windows is to use `vcpkg <https://github.com/microsoft/vcpkg>`_. However, previously the openMVG and openMVS ports in vcpkg did not work. Vcpkg can also be used with Linux. However, there were unsolvable problems when using vcpkg so everything was installed natively.

For `OpenMVG <https://github.com/openMVG/openMVG>`_, `OpenMVS <https://github.com/cdcseacave/openMVS>`_ and `star_cats <https://github.com/Bill-Gray/star_cats>`_ it is necessary to have the executables in the correct folder for SISPO to function.

Directory structure
-------------------
```bash
sispo
├── build
├── data
│   ├── input
│   ├── models
│   ├── sensor_database
│   ├── UCAC4
│   │   ├── u4b
│   │   ├── u4i
├── doc
│   ├── thesis
├── sispo
│   ├── compression
│   ├── reconstruction
│   └── sim
├── software
│   ├── blender
│   ├── miniconda
│   ├── openMVG
│   │   ├── openMVG_build
│   │   │   ├── install
│   │   ├── openMVG
│   ├── openMVS
│   │   ├── openMVS_build
│   │   │   ├── install
│   │   ├── openMVS
│   ├── star_cats
│   ├── vcpkg
```

The directory tree shows the assumed overall folder structure after installation. No sub-folders of the build folder or any files are shown.

To make SISPO perform well, it is beneficial to install the `Nvidia CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_ in case an Nvidia graphics card is available.

Step-by-step instructions
-------------------------
In the following enumeration, commands intended to be run in a shell are highlighted with a grey box.

* Clone the GitHub repository onto the local machine ``git clone https://github.com/YgabrielsY/sispo.git``. The project provides a software folder which is intended to be used to install all following software.
* Setup (conda) environment with dependencies (to software/miniconda folder):

  * orekit 9.3.1, the current version 10.0 had issues when attempted to install. Also orekit needs a data package to function, it is distributed with SISPO in the sim module folder.
  * astropy
  * opencv
  * OpenEXR (For Windows the pre-compiled package found at `OpenEXR binaries for Windows <https://www.lfd.uci.edu/~gohlke/pythonlibs/#openexr>`_ needs to be used because the pip or conda version do not work.
  * NumPy
  * Python

* (Especially Windows) Install vcpkg to software/vcpkg folder, follow instructions at `vcpkg <https://github.com/microsoft/vcpkg>`_
* Install Blender as a python module (bpy), during development Blender version 2.8 was used.

  * Clone Blender git repository to software/blender/blender ``git clone git://git.blender.org/blender.git``
  * Compile target bpy ``make bpy``, this works also for Windows through the `make.bat` file provided with Blender
  * If available: Activate CUDA in the cmake project and recompile
  * Install bpy to python environment. Follow `these instructions <https://blender.stackexchange.com/questions/117200/how-to-build-blender-as-a-python-module>`_

* Install OpenMVG, follow instructions at `OpenMVG Build <https://github.com/openMVG/openMVG/blob/master/BUILD.md>`_ or look for hints in the OpenMVG install script in the build folder.

  * Install dependencies according to instructions
  * Clone OpenMVG GitHub repository to ``software/openMVG/openMVG`` ``git clone --recursive https://github.com/openMVG/openMVG.git``
  * Build to ``software/openMVG/build_openMVG`` folder
  * Install to ``software/openMVG/build_openMVG/install`` folder

* Install OpenMVS, follow instructions at `OpenMVS Build <https://github.com/cdcseacave/openMVS/wiki/Building>`_ or look at the OpenMVS install script in the build folder for hints.

  * Install dependencies according to instructions
  * Clone OpenMVS GitHub repository to ``software/openMVS/openMVS`` ``git clone https://github.com/cdcseacave/openMVS.git``
  * Build to ``software/openMVS/build_openMVS`` folder
  * Install to ``software/openMVS/build_openMVS/install`` folder

* Install star_cats

  * Clone star_cats GitHub repository to ``software/star_cats/star_cats`` ``git clone https://github.com/Bill-Gray/star_cats.git``
  * Build to ``software/star_cats/build_star_cats`` ``make``

* Download UCAC4 star catalog to ``data/UCAC4``, use either:

  * the ``build/data/download_ucac4.sh`` script
  * download the folders ``u4b`` and ``u4i`` directly from e.g. `UCAC4 <http://casdc.china-vo.org/mirror/UCAC/UCAC4/>`_
