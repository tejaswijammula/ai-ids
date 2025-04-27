
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(train_df, test_df):
    non_numeric_cols = train_df.select_dtypes(include=['object']).columns
    train_df = train_df.drop(columns=non_numeric_cols)
    test_df = test_df.drop(columns=non_numeric_cols)

    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    return X_train_balanced, y_train_balanced, X_test_scaled, y_test
