from ...core.paths import root_path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def load_raw_data():
    df = pd.read_csv(root_path('data', 'raw', 'fed_recommenders_synthetic_dataset.csv'))
    
    target_classification = ["Risk_AlveolarOsteitis", "Risk_SecondaryInfection", "Risk_NerveDysesthesia", "Risk_Bleeding"]
    leakage_cols = ["Patient", "Client", "Removal_Prob", "Score_1", "Score_2", "Score_3", "Prob_1", "Prob_2", "Prob_3"]
    
    X = df.drop(columns=leakage_cols + target_classification)
    y_classification = df[target_classification]

    # missing indicator # TODO
    missing_cols = ['Tooth_Mobility', 'PartialBony_GingivalCoverage', 'Bone_Density', 'Surg_2_Subtype']
    for col in missing_cols:
        if col in X.columns:
            missing_indicator_name = f'{col}_MISSING'
            X[missing_indicator_name] = X[col].isnull().astype(float)
    
    return {'X': X, 'y_classification': y_classification}


def load_data_with_split(test_size=0.2, val_size=0.2, random_state=42):
    data = load_raw_data()

    X_temp, X_test, y_temp, y_test = train_test_split(data['X'],data['y_classification'],test_size=test_size,random_state=random_state,)

    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp,y_temp,test_size=val_size_adjusted,random_state=random_state,)

    features_to_onehot = ['Surgical_Extraction_Type', 'Tooth_Angulation']
    categorical_cols = [c for c in features_to_onehot if c in X_train.columns]

    if categorical_cols:
        ohe = OneHotEncoder(drop=None,sparse_output=False,handle_unknown='ignore',dtype=np.float32,)
        ohe.fit(X_train[categorical_cols])
        ohe_feature_names = ohe.get_feature_names_out(categorical_cols)

        def add_ohe(X):
            X_ohe = pd.DataFrame(ohe.transform(X[categorical_cols]),columns=ohe_feature_names,index=X.index,)
            X_new = X.drop(columns=categorical_cols)
            return pd.concat([X_new, X_ohe], axis=1)

        X_train = add_ohe(X_train)
        X_val = add_ohe(X_val)
        X_test = add_ohe(X_test)

    X_val  = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train),columns=X_train.columns,index=X_train.index,)
    X_val = pd.DataFrame(imputer.transform(X_val),columns=X_val.columns,index=X_val.index)
    X_test = pd.DataFrame(imputer.transform(X_test),columns=X_test.columns,index=X_test.index,)

    return {
        'train': {'X': X_train, 'y_classification': y_train},
        'val': {'X': X_val, 'y_classification': y_val},
        'test': {'X': X_test, 'y_classification': y_test},
    }