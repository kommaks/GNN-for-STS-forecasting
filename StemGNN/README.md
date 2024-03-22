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
```
## Training and Evaluation

The training procedure and evaluation procedure are all included in the `main.py`. To train and evaluate on some dataset, run the following command:

```train & evaluate
python main.py --train True --evaluate True --cross_val False --n_retrains 30 --dataset <name of csv file> --output_dir <path to output directory> --window_size <length of sliding window> --horizon <predict horizon>
```

The detailed descriptions about the parameters are as following:

| Parameter name | Description of parameter |
| --- | --- |
| train | whether to enable training, default True |
| evaluate | whether to enable evaluation, default True |
| cross_val | whether to run hyperparameters search, default False |
| n_retrains | Number of retrains of the model with fixed hyperparameters, default 30 |
| dataset | file name of input csv, default atm_transactions |
| window_size | length of sliding window, default 30 |
| horizon | predict horizon, default 5 |
| epoch | epoch size during training, default 80 |
| lr | learning rate, default 1.09e-3 |
| multi_layer | hyper parameter of STemGNN which controls the parameter number of hidden layers, default 5 |
| device | device that the code works on, 'cpu' or 'cuda:x' | 
| validate_freq | frequency of validation, default 10 |
| batch_size | batch size, default 64 |
| optimizer | Adam / AdamW / RMSprop, default AdamW |





