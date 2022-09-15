
The code of our paper "SARW: Similarity-Aware Random Walk for GCN" is rewritten according to the paper "Graphsaint: Graph sampling based inductive learning
method".


The following are some of the similar code environments as paper Graphsaint:


## Dependencies


* python >= 3.6.8
* tensorflow >=1.12.0  / pytorch >= 1.1.0
* cython >=0.29.2
* numpy >= 1.14.3
* scipy >= 1.1.0
* scikit-learn >= 0.19.1
* pyyaml >= 3.12
* g++ >= 5.4.0
* openmp >= 4.0


## Datasets


All datasets used in our papers are available for download:


* PPI
* PPI-large (a larger version of PPI)
* Reddit
* Flickr
* Yelp
* Amazon


They are available on [Google Drive link](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) (alternatively, [BaiduYun link (code: f1ao)](https://pan.baidu.com/s/1SOb0SiSAXavwAcNqkttwcg)). Rename the folder to `data` at the root directory.  The directory structure should be as below:


```
GraphSAINT/
│   README.md
│   run_graphsaint.sh
│   ...
│
└───graphsaint/
│   │   globals.py
│   │   cython_sampler.pyx
│   │   ...
│   │
│   └───tensorflow_version/
│   │   │    train.py
│   │   │    model.py
│   │   │    ...
│   │
│   └───pytorch_version/
│       │    train.py
│       │    model.py
│       │    ...
│
└───data/
│   └───ppi/
│   │   │    adj_train.npz
│   │   │    adj_full.npz
│   │   │    ...
│   │
│   └───reddit/
│   │   │    ...
│   │
│   └───...
│
```


## Training Configuration


The hyperparameters needed in training can be set via the configuration file: `./train_config/<name>.yml`.


The configuration files to reproduce the Table 2 results are packed in `./train_config/table2/`.


For detailed description of the configuration file format, please see `./train_config/README.md`


## Run Training


First of all, please compile cython samplers (see above).


We suggest looking through the available command line arguments defined in `./graphsaint/globals.py` (shared by both the Tensorflow and PyTorch versions). By properly setting the flags, you can maximize CPU utilization in the sampling step (by telling the number of available cores), select the directory to place log files, and turn on / off loggers (Tensorboard, Timeline, ...), etc.



To run the code on CPU


```
python -m graphsaint.<tensorflow/pytorch>_version.train --data_prefix ./data/<dataset_name> --train_config <path to train_config yml> --gpu -1
```


To run the code on GPU


```
python -m graphsaint.<tensorflow/pytorch>_version.train --data_prefix ./data/<dataset_name> --train_config <path to train_config yml> --gpu <GPU number>
```


For example `--gpu 0` will run on the first GPU. Also, use `--gpu <GPU number> --cpu_eval` to make GPU perform the minibatch training and CPU to perform the validation / test evaluation.




