import lightgbm as lgb
from glob import glob
import pickle

for model_path in glob("./data/models/*_lgb.txt"):
    model_new_path = model_path.replace("txt", "pickle")
    m_lgb = lgb.Booster(model_file=model_path)

    pickle_out = open(model_new_path, "wb")
    pickle.dump(m_lgb, pickle_out)
