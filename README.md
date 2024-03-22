# GNN-for-STS-forecasting

This repository includes our works on Graph representation learning and its application on Cash Demand Prediction.

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

The manuscript can be visited at https://www.semanticscholar.org/paper/Spectral-Temporal-Graph-Neural-Network-for-Cao-Wang/645054d31fa26b29bbfb0cf73b75f8906c359415 or arxiv https://arxiv.org/abs/2103.07719
[The code](https://github.com/microsoft/StemGNN)

## Results

We compared muptivariate GNN - based models with baseline. We calculated RMSE and WMAPE metrics for them.
| Models | RMSE | MAPE(%) |
| -----   | ---- | ---- |
| GCN | 0.027 | 43.18 % |
| GRU | 0.023 | 36.3 % |
| T-GCN | 0.025 | 37.79 % |
| A3T-GCN | 0.01973 | 31.11 % |
| StemGNN | - | 32.2 $\pm$ 1.4 % |
| Prophet | 0.023 | 30.01 % |
|  |  | (Best: 29.1 %) |


## Authors

1. Maksim Komiakov, [@kommaks](https://github.com/kommaks)
2. Michil Trofimov, [@michtrofimov](https://github.com/michtrofimov)
3. Khasan Akhmadiev, [@hasaki77](https://github.com/hasaki77)
4. Andrei Volodichev, [@ahdr3w](https://github.com/ahdr3w)


## License

Copyright 2024 Komiakov Maksim

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.






.
