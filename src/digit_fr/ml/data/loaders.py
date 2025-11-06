from ...core.paths import root_path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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
    
    # onehot only nominal features
    features_to_onehot = ['Surgical_Extraction_Type', 'Tooth_Angulation']
    
    for col in features_to_onehot:
        if col in X.columns:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=False, dtype=float)
            X = X.drop(columns=[col])
            X = pd.concat([X, dummies], axis=1)
    
    # impute missing values # TODO
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    
    return {'X': X, 'y_classification': y_classification}


def load_data_with_split(test_size=0.2, random_state=42):
    data = load_raw_data()
    
    X_train, X_test, y_classification_train, y_classification_test = train_test_split(data['X'], data['y_classification'], test_size=test_size, random_state=random_state)
    
    return {
        'train': {'X': X_train, 'y_classification': y_classification_train},
        'test': {'X': X_test, 'y_classification': y_classification_test}
    }