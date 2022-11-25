# For splitting up and processing dataframe
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
RANDOM_STATE_SEED = 420

def process_data(df, split=True):

    if split:
        train, test = train_test_split(df, test_size=0.3, random_state=RANDOM_STATE_SEED)
        numerical_cols = train.columns[:-3] #assumes last 3 are protocols and labels
        min_max_scaler = MinMaxScaler().fit(train[numerical_cols])
        train[numerical_cols] = min_max_scaler.transform(train[numerical_cols])
        test[numerical_cols] = min_max_scaler.transform(test[numerical_cols])
        y_train = np.array(train.pop("Label")) # pop removes "Label" from the dataframe
        X_train = train.values
        y_test = np.array(test.pop("Label")) # pop removes "Label" from the dataframe
        X_test = test.values
        return {'X_train': X_train,
                'y_train': y_train,
                'X_test':  X_test,
                'y_test': y_test}
    else:
        test = df
        numerical_cols = test.columns[:-3] #assumes last 3 are protocols and labels
        min_max_scaler = MinMaxScaler().fit(test[numerical_cols])
        test[numerical_cols] = min_max_scaler.transform(test[numerical_cols])
        y_test = np.array(test.pop("Label")) # pop removes "Label" from the dataframe
        X_test = test.values
        return {'X_test':  X_test,
                'y_test': y_test}

def save(model, filepath):
    joblib.dump(model, filepath)

def load(filepath):
    return joblib.load(filepath)