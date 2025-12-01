from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def missing_indicators(X: pd.DataFrame, missing_cols: List[str]) -> pd.DataFrame:
    X = X.copy()
    for col in missing_cols:
        if col in X.columns:
            missing_indicator_name = f'{col}_MISSING'
            X[missing_indicator_name] = X[col].isnull().astype(float)
    return X

class PreprocessingPipeline:
    def __init__(
        self,
        categorical_features: List[str] = None,
        missing_indicator_cols: List[str] = None,
        imputation_strategy: str = 'median'
    ):

        self.categorical_features = categorical_features or ['Surgical_Extraction_Type', 'Tooth_Angulation']
        self.missing_indicator_cols = missing_indicator_cols or ['Tooth_Mobility', 'Bone_Density']
        self.imputation_strategy = imputation_strategy
        
        self.ohe: Optional[OneHotEncoder] = None
        self.imputer: Optional[SimpleImputer] = None
        self.ohe_feature_names: Optional[List[str]] = None
        self.feature_columns: Optional[List[str]] = None
        self._is_fitted = False
    
    def fit(self, X_train: pd.DataFrame) -> 'PreprocessingPipeline':
        X_train = X_train.copy()
        
        X_train = missing_indicators(X_train, self.missing_indicator_cols)
        
        categorical_cols = [c for c in self.categorical_features if c in X_train.columns]
        
        if categorical_cols:
            self.ohe = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore', dtype=np.float32)
            self.ohe.fit(X_train[categorical_cols])
            self.ohe_feature_names = self.ohe.get_feature_names_out(categorical_cols)
        else:
            self.ohe = None
            self.ohe_feature_names = []
        
        if self.ohe is not None and categorical_cols:
            X_ohe = pd.DataFrame(self.ohe.transform(X_train[categorical_cols]), columns=self.ohe_feature_names, index=X_train.index)
            X_train = pd.concat([X_train.drop(columns=categorical_cols), X_ohe], axis=1)
        
        self.imputer = SimpleImputer(strategy=self.imputation_strategy)
        X_train_imputed = pd.DataFrame(self.imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)

        self.feature_columns = list(X_train_imputed.columns)
        self._is_fitted = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise ValueError("Fit pipeline before transforming")
        
        X = X.copy()
        
        X = missing_indicators(X, self.missing_indicator_cols)
        
        categorical_cols = [c for c in self.categorical_features if c in X.columns]
        
        if self.ohe is not None and categorical_cols:
            X_ohe = pd.DataFrame(self.ohe.transform(X[categorical_cols]), columns=self.ohe_feature_names, index=X.index)
            X = pd.concat([X.drop(columns=categorical_cols), X_ohe], axis=1)
        
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                for col in missing_cols:
                    X[col] = 0.0
            X = X.reindex(columns=self.feature_columns, fill_value=0.0)
        
        X_imputed = pd.DataFrame(self.imputer.transform(X), columns=X.columns, index=X.index)
        
        return X_imputed