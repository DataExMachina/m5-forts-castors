# m5-forts-castors

## /data
- raw: CSV files from Kaggle.
- interim: raw data merged and formatted (long/grid format). No feature enginnering.
- refined: interim data enhanced. It could be a lot of different feature engineering for different algorithms.
- exernal: it could be dictionaries containing the hyper-parameters of the algorithms obtained with a Bayesian optimization in the cloud.
- submission: outputs files. 

## lightgbm process
- Generate data for **validation** of **evaluation** horizon.
```
python lightgbm_pipeline/raw_data_prep.py --horizon="validation"
python lightgbm_pipeline/raw_data_prep.py --horizon="evaluation"
```
