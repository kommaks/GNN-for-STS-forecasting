# GNN-for-STS-forecasting

This repository includes our works on Graph representation learning and its application on Cash Demand Prediction

## Requirements
**Create enviroment in Conda**

Recommended version of Python:

* **Python**: python3.10.12

Run the following commands to create a venv and install python dependencies:
```setup
conda create -n gnn_for_sts python=3.10.12
conda activate gnn_for_sts
pip install -r requirements.txt
```

## Dataset
We can get the raw data through the links above. We evaluate the performance of tansaction frequency forecasting for ATMs.

The file structure is listed as follows:

1 T-GCN is the source codes for the paper named “T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction” published at IEEE Transactions on Intelligent Transportation Systems (T-ITS) which forged the T-GCN model model the spatial and temporal dependence simultaneously.

2 A3T-GCN is the source codes for the paper named “A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting” published at ISPRS International Journal of Geo-Information which strengthen the T-GCN model model with attention structure.

.
