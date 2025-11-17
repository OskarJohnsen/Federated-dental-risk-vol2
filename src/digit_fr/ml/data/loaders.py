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
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        data['X'], 
        data['y_classification'], 
        test_size=test_size, 
        random_state=random_state
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_adjusted,
        random_state=random_state
    )
    
    features_to_onehot = ['Surgical_Extraction_Type', 'Tooth_Angulation']
    
    categorical_cols = [col for col in features_to_onehot if col in X_train.columns]
    numerical_cols = [col for col in X_train.columns if col not in categorical_cols]
    
    X_train_encoded = X_train.copy()
    X_val_encoded = X_val.copy()
    X_test_encoded = X_test.copy()
    
    if categorical_cols:
        ohe = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore', dtype=np.float32)
        ohe.fit(X_train[categorical_cols])
        
        ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
        
        X_train_ohe = pd.DataFrame(
            ohe.transform(X_train[categorical_cols]),
            columns=ohe_feature_names,
            index=X_train.index
        )
        X_val_ohe = pd.DataFrame(
            ohe.transform(X_val[categorical_cols]),
            columns=ohe_feature_names,
            index=X_val.index
        )
        X_test_ohe = pd.DataFrame(
            ohe.transform(X_test[categorical_cols]),
            columns=ohe_feature_names,
            index=X_test.index
        )
        
        X_train_encoded = X_train_encoded.drop(columns=categorical_cols)
        X_val_encoded = X_val_encoded.drop(columns=categorical_cols)
        X_test_encoded = X_test_encoded.drop(columns=categorical_cols)
        
        X_train_encoded = pd.concat([X_train_encoded, X_train_ohe], axis=1)
        X_val_encoded = pd.concat([X_val_encoded, X_val_ohe], axis=1)
        X_test_encoded = pd.concat([X_test_encoded, X_test_ohe], axis=1)
    
    X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train_encoded), 
        columns=X_train_encoded.columns, 
        index=X_train_encoded.index
    )
    X_val_imputed = pd.DataFrame(
        imputer.transform(X_val_encoded), 
        columns=X_val_encoded.columns, 
        index=X_val_encoded.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test_encoded), 
        columns=X_test_encoded.columns, 
        index=X_test_encoded.index
    )
    
    return {
        'train': {'X': X_train_imputed, 'y_classification': y_train},
        'val': {'X': X_val_imputed, 'y_classification': y_val},
        'test': {'X': X_test_imputed, 'y_classification': y_test}
    }