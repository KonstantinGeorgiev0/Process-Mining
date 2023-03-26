import copy
import time
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
import splitter, constants
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from datetime import datetime
from baseline import classification_performance, regression_performance
from sklearn.linear_model import LinearRegression

def importance(model, X_val, y_val):
    r = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=0)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{X_val.columns[i]:<8}"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")

def tune_rf_activity(train_data, columns):
    X = train_data[columns]
    y = train_data[constants.NEXT_EVENT]

    
    ######## Parameters #######
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [1, 2, 5, 10, 15, 50]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 6, 8]
    # Method of selecting samples for training each tree
    bootstrap = [True, False] # Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    

    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, 
                                   param_distributions = random_grid, 
                                   n_iter = 10, 
                                   cv = 3, 
                                   verbose=0, 
                                   random_state=42, 
                                   n_jobs = -1)
    rf_random.fit(X, y)

    preds = rf_random.best_estimator_.predict(X)
    
    
    train_data[constants.NEXT_EVENT_PREDICTION] = preds

    print(rf_random.best_estimator_)
    print(accuracy_score(preds,y))

    return rf_random.best_estimator_

def train_activity_model(train_data, clf, columns, normal=True):
    X = train_data[columns]

    if not normal:
        y = train_data['next_activity_id']
    else:
        y = train_data[constants.NEXT_EVENT]

    clf.fit(X,y)
    preds = clf.predict(X)
    
    train_data[constants.NEXT_EVENT_PREDICTION] = preds
    

    classification_performance(train_data, "Confusion_Matrices/conf_matrix_random_forest_train.png")
    
    print(accuracy_score(preds,y))
    return clf

def train_time_model(train_data, clf, columns):
    X = train_data[columns]

    y = train_data[constants.TIME_DIFFERENCE]

    clf.fit(X,y)
    preds = clf.predict(X)

    train_data[constants.TIME_DIFFERENCE_PREDICTION] = preds

    regression_performance(train_data)

    print(mean_absolute_error(preds,y))
    return clf

def test_activity_model(test_data, clf, columns, normal=True):
    X = test_data[columns]

    if not normal:
        y = test_data['next_activity_id']
    else:
        y = test_data[constants.NEXT_EVENT]
    
    preds = clf.predict(X)

    test_data[constants.NEXT_EVENT_PREDICTION] = preds
    
    classification_performance(test_data, "Confusion_Matrices/conf_matrix_random_forest_test.png")

    acc = accuracy_score(preds, y)
    print(acc)
    return acc

def test_time_model(test_data, clf, columns):
    X = test_data[columns]

    y = test_data[constants.TIME_DIFFERENCE]
    preds = clf.predict(X)

    test_data[constants.TIME_DIFFERENCE_PREDICTION] = preds

    regression_performance(test_data)
    
    return 1

def compare_all_models(train_data, test_data, timer):
    cols = ['activity number in case', 'case end count',
            'days_until_next_holiday', 'is_weekend', 
            'seconds_since_week_start', 'is_work_time', 
            'seconds_to_work_hours', 'is_holiday',
            'workrate', 'active cases']

    # ----------------- TRAIN DATASET ---------------------------------
    # copy so we don't modify the original training set
    train_data = copy.deepcopy(train_data).rename(columns={constants.CASE_POSITION_COLUMN: 'name'})
    names_ohe = pd.get_dummies(train_data['name'])
    first_lag_ohe = pd.get_dummies(train_data['first_lag_event']).add_prefix('first_lag_')
    second_lag_ohe = pd.get_dummies(train_data['second_lag_event']).add_prefix('second_lag_')

    cols_train = list(names_ohe.columns) + list(first_lag_ohe.columns) + list(second_lag_ohe.columns)

    train_data = train_data.drop('name', axis=1).join(names_ohe).join(first_lag_ohe).join(second_lag_ohe).dropna()
    train_data['next_activity_id'] = pd.factorize(train_data[constants.NEXT_EVENT])[0]
   
    # --------------------- TEST DATASET -------------------------------
    # copy test dataset
    test_data = copy.deepcopy(test_data).rename(columns={constants.CASE_POSITION_COLUMN: 'name'})
    names_ohe = pd.get_dummies(test_data['name'])
    first_lag_ohe = pd.get_dummies(test_data['first_lag_event']).add_prefix('first_lag_')
    second_lag_ohe = pd.get_dummies(test_data['second_lag_event']).add_prefix('second_lag_')
    
    cols_test = list(names_ohe.columns) + list(first_lag_ohe.columns) + list(second_lag_ohe.columns)

    test_data = test_data.drop('name', axis=1).join(names_ohe).dropna().join(first_lag_ohe).join(second_lag_ohe).dropna()
    test_data['next_activity_id'] = pd.factorize(test_data[constants.NEXT_EVENT])[0]
    
    cols.extend([el for el in cols_train if el in cols_test])

    timer.send("Time to prepare columns (in seconds): ")
    
    print("Decision Tree:")
    print("-----------------------------")
    print("Next activity:")
    clf = DecisionTreeClassifier()
    dec_tree_clas = train_activity_model(train_data, clf, cols)

    timer.send("Time to train decision tree classifier (in seconds): ")

    test_activity_model(test_data, dec_tree_clas, cols)
    
    timer.send("Time to evaluate decision tree classifier (in seconds): ")

    print("Random Forest:")
    print("-----------------------------")
    print("Next activity:")
    clf = RandomForestClassifier()
    rand_forest_class = train_activity_model(train_data, clf, cols)

    timer.send("Time to train random forest classifier (in seconds): ")

    test_activity_model(test_data, rand_forest_class, cols)
    
    timer.send("Time to evaluation random forest classifier (in seconds): ")

    print("Linear Regression:")
    print("-----------------------------")
    print("Time to next activity:")
    reg = LinearRegression()
    lin_regr = train_time_model(train_data, reg, cols)

    timer.send("Time to train linear regression (in seconds): ")

    test_time_model(test_data, lin_regr, cols)

    timer.send("Time to evaluate linear regression (in seconds): ")

    print("Random Forest Regression:")
    print("-----------------------------")
    print("Time to next activity:")
    reg = RandomForestRegressor()
    rand_forest_regr = train_time_model(train_data, reg, cols)

    timer.send("Time to train random forest regression (in seconds): ")

    test_time_model(test_data, rand_forest_regr, cols)

    timer.send("Time to evaluate random forest regression (in seconds): ")
