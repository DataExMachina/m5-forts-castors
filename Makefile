
lgb_data_prep:
	python lightgbm_pipeline/raw_data_prep.py --horizon="validation"
	python lightgbm_pipeline/raw_data_prep.py --horizon="evaluation"

lgb_fe:
	python lightgbm_pipeline/feature_engineering.py --horizon="validation" --task="volume"
	python lightgbm_pipeline/feature_engineering.py --horizon="validation" --task="share"
	python lightgbm_pipeline/feature_engineering.py --horizon="evaluation" --task="volume"
	python lightgbm_pipeline/feature_engineering.py --horizon="evaluation" --task="share"

lgb_train_and_pred_validation:
	python lightgbm_pipeline/machine_learning.py --horizon="validation" --task="volume"
	python lightgbm_pipeline/machine_learning.py --horizon="validation" --task="share"

lgb_train_and_pred_evaluation:
	python lightgbm_pipeline/machine_learning.py --horizon="evaluation" --task="volume"
	python lightgbm_pipeline/machine_learning.py --horizon="evaluation" --task="share"

lgb_predict_validation:
	python lightgbm_pipeline/machine_learning.py --horizon="validation" --task="volume" --ml="predict"
	python lightgbm_pipeline/machine_learning.py --horizon="validation" --task="share" --ml="predict"

lgb_predict_evaluation:
	python lightgbm_pipeline/machine_learning.py --horizon="evaluation" --task="volume" --ml="predict"
	python lightgbm_pipeline/machine_learning.py --horizon="evaluation" --task="share" --ml="predict"