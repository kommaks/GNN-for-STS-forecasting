# GNN-for-STS-forecasting

This repository includes our works on Graph representation learning and its application on Cash Demand Prediction

## Requirements

Recommended version of OS & Python:

* **OS**: Ubuntu 18.04.2 LTS
* **Python**: python3.7 ([instructions to install python3.7](https://linuxize.com/post/how-to-install-python-3-7-on-ubuntu-18-04/)).

To install python dependencies, virtualenv is recommended, `sudo apt install python3.7-venv` to install virtualenv for python3.7. All the python dependencies are verified for `pip==20.1.1` and `setuptools==41.2.0`. Run the following commands to create a venv and install python dependencies:

```setup
python3.7 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

The file structure is listed as follows:

1 T-GCN is the source codes for the paper named “T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction” published at IEEE Transactions on Intelligent Transportation Systems (T-ITS) which forged the T-GCN model model the spatial and temporal dependence simultaneously.

2 A3T-GCN is the source codes for the paper named “A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting” published at ISPRS International Journal of Geo-Information which strengthen the T-GCN model model with attention structure.

.
