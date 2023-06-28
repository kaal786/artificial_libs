from xgboost import XGBRegressor,XGBClassifier
from lightgbm import LGBMRegressor,LGBMClassifier
from catboost import CatBoostRegressor,CatBoostClassifier


groupbased_estimators=['QuantileRegressor',
                        'RegressorChain',
                        'MultiOutputRegressor',
                        'StackingRegressor',
                        'VotingRegressor',
                        'RadiusNeighborsRegressor',
                        'PLSRegression',
                        'ClassifierChain',
                        'MultiOutputClassifier',
                        'OneVsOneClassifier',
                        'OneVsRestClassifier',
                        'OutputCodeClassifier',
                        'RadiusNeighborsClassifier',
                        'StackingClassifier',
                        'VotingClassifier',
                        'GaussianProcessClassifier',
                        'BernoulliNB',
                       'CalibratedClassifierCV',
                       'LabelPropagation',
                       'LabelSpreading',
                       
                        ]

extrareg_estimators=[(XGBRegressor().__class__.__name__,XGBRegressor),
(LGBMRegressor().__class__.__name__,LGBMRegressor),
(CatBoostRegressor().__class__.__name__,CatBoostRegressor)]


extraclf_estimators=[(XGBClassifier().__class__.__name__,XGBClassifier),
                    (LGBMClassifier().__class__.__name__,LGBMClassifier),
                    (CatBoostClassifier().__class__.__name__,CatBoostClassifier)   ]