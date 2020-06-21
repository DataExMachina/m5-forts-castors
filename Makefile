
lgb_data_prep:
	python lightgbm_pipeline/raw_data_prep.py --horizon="validation"
	python lightgbm_pipeline/raw_data_prep.py --horizon="evaluation"

lgb_fe:
	python lightgbm_pipeline/feature_engineering.py --horizon="validation" --task="volume"
	python lightgbm_pipeline/feature_engineering.py --horizon="validation" --task="share"
	python lightgbm_pipeline/feature_engineering.py --horizon="evaluation" --task="volume"
	python lightgbm_pipeline/feature_engineering.py --horizon="evaluation" --task="share"