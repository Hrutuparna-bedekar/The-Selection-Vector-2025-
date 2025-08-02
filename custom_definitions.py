import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import FunctionTransformer , LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline as ImbPipeline   
from imblearn.over_sampling import SMOTE   
from sklearn.linear_model import LogisticRegression   
from splitter import split_columns

import re


#==============Your Name======================
# Your code
#==============Your Name=====================
#Dont remove the following snippet and follow the same

#==========Arjun E Naik==============
class YesNoConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        mapping = {'yes': 1, 'y': 1, 'no': 0, 'n': 0, '1': 1, '0': 0}

        return X.iloc[:, 0].apply(lambda x: mapping.get(str(x).lower(), 0)).values.reshape(-1, 1)

class ReplaceInfWithNan(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        return X.replace([np.inf, -np.inf], np.nan)

class ConvertToString(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.astype(str)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X_future = X.copy()
        X_future['new_feature_1'] = X_future['portland_cement_kg'] * X_future['mixing_water_kg']
        X_future['new_feature_2'] = X_future['ground_slag_kg'] + X_future['coal_ash_kg']
        return X_future
#==========Arjun E Naik==============

#=========Rudra======================
import re

from word2number import w2n
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Define custom column dropper class
class ColumnDropper(BaseEstimator, TransformerMixin):
    def _init_(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop, errors='ignore')

# Define classes for feature engineering
class CementGradeParser:
    def _call_(self, X):
        def parse(val):
            try:
                return float(val)
            except:
                val = str(val).lower().strip()
                num_match = re.findall(r"\d+\.?\d*", val)
                if num_match:
                    return float(num_match[0])
                try:
                    return w2n.word_to_num(val)
                except:
                    return None
        X['cement_grade'] = X['cement_grade'].apply(parse)
        return X

class DatePreprocessor:
    def _call_(self, X):
        X['last_modified'] = pd.to_datetime(X['last_modified'], errors='coerce')
        X['last_modified_year'] = X['last_modified'].dt.year.astype('int64')
        X['last_modified_month'] = X['last_modified'].dt.month.astype('int64')
        X['last_modified_day'] = X['last_modified'].dt.day.astype('int64')
        X['last_modified_weekday'] = X['last_modified'].dt.dayofweek.astype('int64')
        X.drop(columns=['last_modified'], inplace=True)

        X['inspection_timestamp'] = pd.to_datetime(X['inspection_timestamp'], errors='coerce')
        X['inspection_year'] = X['inspection_timestamp'].dt.year.fillna(-1).astype('int64')
        X['inspection_month'] = X['inspection_timestamp'].dt.month.fillna(-1).astype('int64')
        X['inspection_day'] = X['inspection_timestamp'].dt.day.fillna(-1).astype('int64')
        X['inspection_weekday'] = X['inspection_timestamp'].dt.dayofweek.fillna(-1).astype('int64')
        X.drop(columns=['inspection_timestamp'], inplace=True)
        return X

class CategoryConverter:
    def _call_(self, X):
        approval_mapping = {'yes': True, 'y': True, '1': True, 'no': False, 'n': False, '0': False}
        X['is_approved'] = X['is_approved'].astype(str).str.strip().str.lower().map(approval_mapping).astype(int)
        rating_map = {'A ++': 6, 'A+': 5, 'AA': 4.5, 'A': 4, 'A-': 3.5, 'B': 3, 'C': 2}
        X['supplier_rating'] = X['supplier_rating'].map(rating_map)
        X['is_valid_strength'] = X['is_valid_strength'].astype(int)
        X['static_col'] = X['static_col'].astype(int)
        return X

class TimeDeltaAndRatios:
    def _call_(self, X):
        X['time_since_casting'] = pd.to_timedelta(X['time_since_casting'], errors='coerce')
        X['time_since_casting_days'] = X['time_since_casting'].dt.total_seconds() / (24 * 3600)
        X.drop('time_since_casting', axis=1, inplace=True)
        X['water_binder_ratio'] = X['mixing_water_kg'] / (X['total_binder_kg'] + 1e-6)
        X['admixture_binder_ratio'] = X['chemical_admixture_kg'] / (X['total_binder_kg'] + 1e-6)
        return X

class RedundantDropper:
    def _call_(self, X):
        X.drop(columns=[
            'time_since_casting_days', 'days_since_last_modified',
            'days_since_inspection', 'random_noise'
        ], inplace=True, errors='ignore')
        return X

class FinalCleaner:
    def _call_(self, X):
        num_cols = X.select_dtypes(include=['float64', 'int64']).columns
        imputer = SimpleImputer(strategy='mean')
        X[num_cols] = imputer.fit_transform(X[num_cols])
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        return X

#===============Rudra=============

#==================================shrihari=====================================================================

class FinalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self.columns_)

        cement = df['Cement_component_1kg_in_a_m3_mixture'].replace(0, 0.0001)
        df['total_binder'] = (
            df['Cement_component_1kg_in_a_m3_mixture'] +
            df['Blast_Furnace_Slag_component_2kg_in_a_m3_mixture'] +
            df['Fly_Ash_component_3kg_in_a_m3_mixture']
        )
        total_binder_for_ratio = df['total_binder'].replace(0, 0.0001)
        df['water_cement_ratio'] = df['Water_component_4kg_in_a_m3_mixture'] / cement
        total_aggregate = (
            df['Coarse_Aggregate_component_6kg_in_a_m3_mixture'] +
            df['Fine_Aggregate_component_7kg_in_a_m3_mixture']
        )
        df['agg_binder_ratio'] = total_aggregate / total_binder_for_ratio

        safe_age = df['Age_day'].clip(lower=0)
        df['log_Age_day'] = np.log1p(safe_age)

        df = df.drop(columns=[
            'Cement_component_1kg_in_a_m3_mixture',
            'Blast_Furnace_Slag_component_2kg_in_a_m3_mixture',
            'Fly_Ash_component_3kg_in_a_m3_mixture',
            'Water_component_4kg_in_a_m3_mixture',
            'Coarse_Aggregate_component_6kg_in_a_m3_mixture',
            'Fine_Aggregate_component_7kg_in_a_m3_mixture'
        ])
        
        return df
    
column_mapping = {
    'portland_cement_kg': 'Cement_component_1kg_in_a_m3_mixture',
    'ground_slag_kg': 'Blast_Furnace_Slag_component_2kg_in_a_m3_mixture',
    'coal_ash_kg': 'Fly_Ash_component_3kg_in_a_m3_mixture',
    'mixing_water_kg': 'Water_component_4kg_in_a_m3_mixture',
    'chemical_admixture_kg': 'Superplasticizer_component_5kg_in_a_m3_mixture',
    'gravel_aggregate_kg': 'Coarse_Aggregate_component_6kg_in_a_m3_mixture',
    'sand_aggregate_kg': 'Fine_Aggregate_component_7kg_in_a_m3_mixture',
    'specimen_age_days': 'Age_day'
}
df = df.rename(columns=column_mapping)

TARGET = 'compressive_strength_mpa'
CORE_FEATURES = list(column_mapping.values())
df_clean = df[CORE_FEATURES + [TARGET]]

df_clean.dropna(subset=[TARGET], inplace=True)

# 2. Handle Outliers
for column in df_clean.columns:
    if df_clean[column].dtype in ['int64', 'float64']:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean[column] = np.where(df_clean[column] > upper_bound, upper_bound, df_clean[column])
        df_clean[column] = np.where(df_clean[column] < lower_bound, lower_bound, df_clean[column])

# 3. Prepare Final Data and Pipeline
X = df_clean.drop(TARGET, axis=1)
y = df_clean[TARGET]

# This pipeline uses the best configuration we found
submission_pipeline = Pipeline([
    ('feature_engineer', FinalFeatureEngineer()),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=10.0, random_state=42)) # Using the best alpha
])

#===============Shrihari=============================================

#===============Hrutuparna_bedekar=============================================
def dropper(X):
    col_drop=['Row_ID','mixing_water_kg_mislabeled','json_notes','compressive_strength_mpas','compreesive_strength_mpa','nan_col','static_col','random_noise','batch_id','compressive_strength_mpa_str','compressive_strength_duplicate','last_modified','inspection_timestamp','supplier_rating','strength_category','weird_col','cement_grade','is_valid_strength','is_approved','pseudo_target','time_since_casting','leaky_strength_log','dangerous_ratio','Unnamed: 0']
    X=X.replace([-np.inf,np.inf],np.NaN)
        
    
      
    X=X.drop(columns=[col for col in col_drop if col in X.columns],errors='ignore')
    X['specimen_age_days']=np.where(X['specimen_age_days']<0,np.NaN,X['specimen_age_days'])
   
    
    return X

def imputation(X):
    colum=X.columns         
    knn=KNNImputer(n_neighbors=3)
    X=knn.fit_transform(X)
    X=pd.DataFrame(X,columns=colum)
    for col in X.columns:
        lower=X[col].quantile(0.01)
        upper=X[col].quantile(0.99)
        X[col]=X[col].clip(lower,upper)
        
    
   
    return X

def agree(X):
    
   
    X['Total_aggregate'] = X['gravel_aggregate_kg'] + X['sand_aggregate_kg']
    X['binder_to_cement'] = (X['ground_slag_kg'] + X['coal_ash_kg']) / (X['portland_cement_kg']+1)
    X['water_cement_cal_ratio']=X['mixing_water_kg']/(X['portland_cement_kg']+1)
    X=X.drop(columns=['gravel_aggregate_kg','sand_aggregate_kg','ground_slag_kg','coal_ash_kg','portland_cement_kg','mixing_water_kg','water_cement_ratio'])
    
    
    return X




dropper_transform=FunctionTransformer(dropper,validate=False)
impute_transform=FunctionTransformer(imputation,validate=False)
agree_transform=FunctionTransformer(agree,validate=False)

#===============Hrutuparna_bedekar=============================================
