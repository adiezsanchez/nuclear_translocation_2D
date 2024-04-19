<h1></h1>

[![License](https://img.shields.io/pypi/l/napari-accelerated-pixel-and-object-classification.svg?color=green)](https://github.com/adiezsanchez/intestinal_organoid_yolov8/blob/main/LICENSE)
[![Development Status](https://img.shields.io/pypi/status/napari-accelerated-pixel-and-object-classification.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)

This repository provides tools in the form of interactive Jupyter notebooks to analyze intensity of marker proteins within the nuclear envelope of cells in 2D. Then it offers some data and segmentation visualization tools to define cells as marker + or - based on mean intensity levels.

<h2>Raw Data Download</h2>

1. [Contact Me](mailto:alberto.d.sanchez@ntnu.no) to obtain a fresh working S3 bucket pre-signed link.

2. Paste the link inside <code>0_data_download.ipynb</code> notebook after <code>presigned_url</code>.

3. Run the notebook to download and extract the data.

<h2>Image Analysis Instructions</h2>

#TODO:

<h2>Environment setup instructions</h2>

1. In order to run these Jupyter notebooks and .py scripts you will need to familiarize yourself with the use of Python virtual environments using Mamba. See instructions [here](https://biapol.github.io/blog/mara_lampert/getting_started_with_mambaforge_and_python/readme.html).

2. Then you will need to create a virtual environment using the command below or from the .yml file in the envs folder (recommended, see step 3):

   <code>mamba create -n microglia python=3.9 devbio-napari cellpose pytorch torchvision plotly pyqt -c conda-forge -c pytorch</code>

3. To recreate the venv from the environment.yml file stored in the envs folder (recommended) navigate into the envs folder using <code>cd</code> in your console and then execute:

   <code>mamba env create -f environment.yml</code>

4. Activate it by typing in the console:

   <code>mamba activate microglia</code>

5. Then launch Jupyter lab to interact with the code by typing:

   <code>jupyter lab</code>
