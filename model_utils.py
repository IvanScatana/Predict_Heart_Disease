import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

class CholesterolCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['Cholesterol'] = X['Cholesterol'].replace(0, np.nan)
        return X
    
def create_preprocessor():

    cholesterol_cleaner = CholesterolCleaner()
    imputer = ColumnTransformer(
        transformers=[('num_imputer', SimpleImputer(strategy='mean'), ['Age', 'Cholesterol'])],
        verbose_feature_names_out=False,
        remainder='passthrough'
    )

    ordinal_encoding_columns = ['RestingECG', 'ST_Slope']
    resting_ecg_order = ['Normal', 'ST', 'LVH']
    st_slope_order = ['Up', 'Flat', 'Down']
    one_hot_encoding_columns = ['Sex', 'ChestPainType', 'ExerciseAngina']

    standard_scaler_columns = ['Age', 'MaxHR']
    robust_scaler_columns = ['RestingBP', 'Cholesterol']
    minmax_scaler_columns = ['Oldpeak']

    scaler_and_encoder = ColumnTransformer(
        [
            ('ordinal_encoding', OrdinalEncoder(categories=[resting_ecg_order, st_slope_order]), ordinal_encoding_columns),
            ('one_hot_encoding_columns', OneHotEncoder(sparse_output=False), one_hot_encoding_columns),
            
            ('robust', RobustScaler(), robust_scaler_columns),
            ('standard', StandardScaler(), standard_scaler_columns),
            ('minmax', MinMaxScaler(), minmax_scaler_columns)
        ],
        verbose_feature_names_out = False,
        remainder = 'passthrough' 
    )

    preprocessor = Pipeline(
    [
        ('cleaner', cholesterol_cleaner),
        ('imputer', imputer),
        ('scaler_and_encoder', scaler_and_encoder)
    ]
    )

    return preprocessor


def create_final_pipeline():

    preprocessor = create_preprocessor()

    lr_params = {
        'solver': 'saga',
        'C': 15.632429914797136,
        'max_iter': 2688,
        'penalty': 'elasticnet',
        'l1_ratio': 0.13128978495943688,
        'random_state': 42
    }
    
    knn_params = {
        'n_neighbors': 18,
        'p': 1,
        'weights': 'distance'
    }
    
    tree_params = {
        'max_depth': 4,
        'criterion': 'entropy',
        'min_samples_split': 4,
        'min_samples_leaf': 6,
        'random_state': 42
    }
    
    rf_params = {
    'n_estimators': 80,          # уменьшили с 160
    'max_depth': 5,              # уменьшили с 8
    'min_samples_split': 10,     # увеличили с 3
    'min_samples_leaf': 5,       # увеличили с 1
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
    }
    
    cat_params = {
    'iterations': 250,           # уменьшили с 362
    'learning_rate': 0.05,       # увеличили с 0.0165
    'depth': 4,                  # уменьшили с 6
    'l2_leaf_reg': 5,            # увеличили с 0.696
    'bagging_temperature': 1,
    'random_strength': 2,        # увеличили
    'subsample': 0.6,            # уменьшили с 0.606
    'border_count': 127,
    'eval_metric': 'Accuracy',
    'verbose': 0,
    'random_seed': 42
    }
    
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'max_depth': 6,
        'num_leaves': 41,
        'learning_rate': 0.04051838615041596,
        'n_estimators': 150,
        'random_state': 42,
        'subsample': 0.9686428640355428,
        'colsample_bytree': 0.6998085542823399,
        'reg_alpha': 0.00031974101337827544,
        'reg_lambda': 2.4980671670119188,
        'min_child_samples': 8,
        'verbose': -1
    }
    
    xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 3,              # уменьшили с 5
    'learning_rate': 0.03,       # уменьшили с 0.0647
    'n_estimators': 100,         # уменьшили с 265
    'subsample': 0.5,            # уменьшили с 0.712
    'colsample_bytree': 0.5,     # уменьшили с 0.513
    'reg_alpha': 5,              # увеличили с 1.026
    'reg_lambda': 10,            # уменьшили с 17.27
    'gamma': 1,                  # увеличили с 0.131
    'min_child_weight': 5,       # увеличили с 1.518
    'random_state': 42,
    'use_label_encoder': False,
    'verbosity': 0,
    'n_jobs': -1
}

    lr = LogisticRegression(**lr_params)
    knn = KNeighborsClassifier(**knn_params)
    tree = DecisionTreeClassifier(**tree_params)
    rf = RandomForestClassifier(**rf_params)
    cat = CatBoostClassifier(**cat_params)
    lgbm = lgb.LGBMClassifier(**lgb_params)
    xgb_model = xgb.XGBClassifier(**xgb_params)

    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('cat', cat),
            ('xgb', xgb_model),
            ('knn', knn)
        ],
        voting='soft',
        weights=[1, 2, 1, 1]
    )
    
    # Финальный пайплайн
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', voting_clf)
    ])
    
    return final_pipeline