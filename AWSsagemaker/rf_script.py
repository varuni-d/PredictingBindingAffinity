
import argparse
import joblib
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE


# inference functions ---------------
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf



if __name__ =='__main__':

    print('extracting arguments')
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument('--n-estimators', type=int, default=10)
    parser.add_argument('--min-samples-leaf', type=int, default=3)

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='train.csv')
    parser.add_argument('--test-file', type=str, default='validation.csv')
#    parser.add_argument('--features', type=str)  # in this script we ask user to explicitly name features
#    parser.add_argument('--target', type=str) # in this script we ask user to explicitly name the target

    args, _ = parser.parse_known_args()

    print('reading data')
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    print('building training and testing datasets')
    X_train=train_df.drop('target',axis=1)
    y_train=train_df['target']
    
    X_test=test_df.drop('target',axis=1)
    y_test=test_df['target']
    
#    X_train = train_df[args.features.split()]
#    X_test = test_df[args.features.split()]
#    y_train = train_df[args.target]
#    y_test = test_df[args.target]

    # train
    print('training model')
    model = RandomForestRegressor(n_estimators=args.n_estimators,min_samples_leaf=args.min_samples_leaf,n_jobs=-1)
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)

#    rf_model = Pipeline([('scaler', StandardScaler()),('rf_model',RandomForestRegressor(random_state=1))])
#    params_rf = {'rf_model__n_estimators': [100, 200, 300],'rf_model__max_features': ['log2', 'auto', 'sqrt'],'rf_model__min_samples_leaf': [10, 30, 50]}
#    grid_rf=GridSearchCV(estimator=rf_model, param_grid=params_rf, scoring='neg_mean_absolute_error', cv=3, verbose=1, n_jobs=-1)

    #Fit and predict
#    grid_rf.fit(X_train, y_train)
#    test_pred = grid_rf.predict(X_test)
    
    mae_test = MAE(y_test, test_pred)
    mse_test = MSE(y_test, test_pred)   
    rmse_test=np.sqrt(mse_test)
    print('MAE:', mae_test)
    print('RMSE:',rmse_test)
    
    # print abs error
#    print('validating model')
#    abs_err = np.abs(model.predict(X_test) - y_test)
#    print(abs_err)

    # print couple perf metrics
 #   for q in [10, 50, 90]:
 #       print('AE-at-' + str(q) + 'th-percentile: '
 #             + str(np.percentile(a=abs_err, q=q)))
        
    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print('model persisted at ' + path)
    print(args.min_samples_leaf)
