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

## Danish ATM Transactions Data Set
This dataset comprises around 2.5 million records of withdrawal data along with weather information at the time of the transactions from around 113 ATMs from the year 2017.     
[Data dictionary](/src/assets/RDS+Data+dictionary.pdf)  
[Source of the Data Set in Kaggle](https://www.kaggle.com/sparnord/danish-atm-transactions)  

> This data set contains various types of transactional data as well as the weather data at the time of the transaction, such as:  
**Transaction Date and Time:** Year, month, day, weekday, hour  
**Status of the ATM:** Active or inactive  
**Details of the ATM:** ATM ID, manufacturer name along with location details such as longitude, latitude, street name, street number and zip code  
**The weather of the area near the ATM during the transaction:** Location of measurement such as longitude, latitude, city name along with the type of weather, temperature, pressure, wind speed, cloud and so on  
**Transaction details:** Card type, currency, transaction/service type, transaction amount and error message (if any) 



The file structure is listed as follows:

1) T-GCN is the source codes for the paper named “T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction” published at IEEE Transactions on Intelligent Transportation Systems (T-ITS) which forged the T-GCN model model the spatial and temporal dependence simultaneously.

The manuscript can be visited at https://ieeexplore.ieee.org/document/8809901 or https://arxiv.org/abs/1811.05320
[The code](https://github.com/lehaifeng/T-GCN/tree/master/T-GCN)

2) A3T-GCN is the source codes for the paper named “A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting” published at ISPRS International Journal of Geo-Information which strengthen the T-GCN model model with attention structure.

The manuscript can be visited at https://www.mdpi.com/2220-9964/10/7/485/html or arxiv https://arxiv.org/abs/2006.11583.
[The code](https://github.com/lehaifeng/T-GCN/tree/master/A3T-GCN)

3) StemGNN is a Graph-based multivariate time-series forecasting model. StemGNN jointly learns temporal dependencies and inter-series correlations in the spectral domain, by combining Graph Fourier Transform (GFT) and Discrete Fourier Transform (DFT).

The manuscript can be visited at [https://www.mdpi.com/2220-9964/10/7/485/html](https://www.semanticscholar.org/paper/Spectral-Temporal-Graph-Neural-Network-for-Cao-Wang/645054d31fa26b29bbfb0cf73b75f8906c359415)https://www.semanticscholar.org/paper/Spectral-Temporal-Graph-Neural-Network-for-Cao-Wang/645054d31fa26b29bbfb0cf73b75f8906c359415 or arxiv https://arxiv.org/abs/2103.07719






.
