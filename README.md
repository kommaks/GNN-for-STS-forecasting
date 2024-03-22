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

2) A3T-GCN is the source codes for the paper named “A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting” published at ISPRS International Journal of Geo-Information which strengthen the T-GCN model model with attention structure.

3) StemGNN is a Graph-based multivariate time-series forecasting model. StemGNN jointly learns temporal dependencies and inter-series correlations in the spectral domain, by combining Graph Fourier Transform (GFT) and Discrete Fourier Transform (DFT).

## 1. T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction
Accurate and real-time traffic forecasting plays an important role in the Intelligent Traffic System and is of great significance for urban traffic planning, traffic management, and traffic control. However, traffic forecasting has always been considered an open scientific issue, owing to the constraints of urban road network topological structure and the law of dynamic change with time, namely, spatial dependence and temporal dependence. To capture the spatial and temporal dependence simultaneously, we propose a novel neural network-based traffic forecasting method, the temporal graph convolutional network (T-GCN) model, which is in combination with the graph convolutional network (GCN) and gated recurrent unit (GRU). Specifically, the GCN is used to learn complex topological structures to capture spatial dependence and the gated recurrent unit is used to learn dynamic changes of traffic data to capture temporal dependence. Then, the T-GCN model is employed to traffic forecasting based on the urban road network. Experiments demonstrate that our T-GCN model can obtain the spatio-temporal correlation from traffic data and the predictions outperform state-of-art baselines on real-world traffic datasets.

The manuscript can be visited at https://ieeexplore.ieee.org/document/8809901 or https://arxiv.org/abs/1811.05320

[The code](https://github.com/lehaifeng/T-GCN/tree/master/T-GCN)

## 2. A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting
Accurate real-time traffic forecasting is a core technological problem against the implementation of the intelligent transportation system. However, it remains challenging considering the complex spatial and temporal dependencies among traffic flows. In the spatial dimension, due to the connectivity of the road network, the traffic flows between linked roads are closely related. In terms of the temporal factor, although there exists a tendency among adjacent time points in general, the importance of distant past points is not necessarily smaller than that of recent past points since traffic flows are also affected by external factors. In this study, an attention temporal graph convolutional network (A3T-GCN) traffic forecasting method was proposed to simultaneously capture global temporal dynamics and spatial correlations. The A3T-GCN model learns the short-time trend in time series by using the gated recurrent units and learns the spatial dependence based on the topology of the road network through the graph convolutional network. Moreover, the attention mechanism was introduced to adjust the importance of different time points and assemble global temporal information to improve prediction accuracy. Experimental results in real-world datasets demonstrate the effectiveness and robustness of proposed A3T-GCN.

The manuscript can be visited at https://www.mdpi.com/2220-9964/10/7/485/html or arxiv https://arxiv.org/abs/2006.11583.

[The code](https://github.com/lehaifeng/T-GCN/tree/master/A3T-GCN)


.
