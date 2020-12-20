# T-REX
This repository contains source code to generate the T-REX visualizations described in Maloca et al., 
*Unraveling the deep learning gearbox in optical coherence tomography image segmentation towards explainable artificial intelligence*.
This is, Multidimensional Scaling (MDS) plots and heatmap plots based on Hamming distances that measure the distance between two pixel-wise labellings. 

## Setup
* Create a virtual environment with Python 3.8 64-bit with `venv`.
  See https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
* Install dependencies with `pip`
    ```
    pip install -r requirements.txt
    ```
## Visualizations
* Set `PLOTS_OUT_DIR` in `definitions.py`
* Create `PLOTS_OUT_DIR`
* Run unit tests in `plots/test_make_plots.py` to generate T-REX visualizations
* T-REX visualizations are stored in `PLOTS_OUT_DIR`

## U-Net
The U-Net model used in this study can be found in `u_net/model.py`.
