# **APMGR: Adaptive Prototype Learning with Meta-Generated Residuals for Cross-Domain Recommendation**

## Dataset

We utilized the Amazon Reviews 5-score dataset.: http://jmcauley.ucsd.edu/data/amazon/

Put the data files [CDs and Vinyl](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz), [Movies and TV](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz), [Books](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz) (5-scores) in `./data/raw`.

Data process via:

```python
python entry.py --process_data_mid 1 --process_data_ready 1
```

```

└── data
    ├── mid                 # Mid data
    │   ├── Books.csv
    │   ├── CDs_and_Vinyl.csv
    │   └── Movies_and_TV.csv
    ├── raw                 # Raw data
    │   ├── reviews_Books_5.json.gz
    │   ├── reviews_CDs_and_Vinyl_5.json.gz
    │   └── reviews_Movies_and_TV_5.json.gz
    └── ready               # Ready data
        ├── _2_8
        ├── _5_5
        └── _8_2
└──pretrained_models        # base models
```


## run

```
python entry.py --task 1 --ratio [0.8,0.2] --epoch 10 --lr 0.01 --num_prototypes 20
```

More hyper-parameter settings can be made in `config.json`

## env

Python 3.9.22

torch

numpy

json

tqdm

tensorflow 2.20.0

## Dependent repositories

PTUPCDR repository:
[https://github.com/easezyc/WSDM2022-PTUPCDR](https://github.com/easezyc/WSDM2022-PTUPCDR)

The implementation of data processing and our code are all based on this repository.
