from feature_utils import pred_emotion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

#create both logistic regression models
def create_logistic_regressions(train_df, y_train):
    #create dataframes for training
    X_train_with_clusters = train_df.copy()
    X_train_without_clusters = train_df.copy().drop(columns= [column for column in train_df.columns if 'cluster_by_hierarcal_5_features' in column] + [column for column in train_df.columns if 'kmeans_wolang_elbow' in column]+ [column for column in train_df.columns if 'kmeans_wlang_elbow' in column])

    #set up scoring function
    scorer = make_scorer(average_precision_score, needs_proba=True)

    model = LogisticRegression(penalty='l1', random_state=42)
    # Define the parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    }
    X_train_without_clusters.to_csv('data/X_train_without_clusters.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    #Find best params
    grid_search_with_clusters = GridSearchCV(model, param_grid, cv=5, scoring=scorer, verbose=5)
    grid_search_with_clusters.fit(X_train_with_clusters, y_train)

    grid_search_without_clusters = GridSearchCV(model, param_grid, cv=5, scoring=scorer, verbose=5)
    grid_search_without_clusters.fit(X_train_without_clusters, y_train)

    #get best models
    best_model_with_clusters = grid_search_with_clusters.best_estimator_
    best_model_without_clusters = grid_search_without_clusters.best_estimator_

    return best_model_with_clusters, best_model_without_clusters

def create_xgboost(train_df, y_train):
    #create 2 dataframes for training
    X_train_with_clusters = train_df.copy()
    X_train_without_clusters = train_df.copy().drop(columns= [column for column in train_df.columns if 'cluster_by_hierarcal_5_features' in column] + [column for column in train_df.columns if 'kmeans_wolang_elbow' in column]+ [column for column in train_df.columns if 'kmeans_wlang_elbow' in column])

    #set up scoring function
    scorer = make_scorer(average_precision_score, needs_proba=True)

    model = XGBClassifier(random_state=42)

    # Define the parameter grid
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [ 0.03, 0.05, 0.1,],
        'colsample_bytree': [0.9, 0.95, 1.0],
        'n_estimators': [100, 200, 300],
    }
    #Find the estimators
    grid_search_with_clusters = GridSearchCV(model, param_grid, cv=5, scoring=scorer, verbose=5)
    grid_search_with_clusters.fit(X_train_with_clusters, y_train)

    grid_search_without_clusters = GridSearchCV(model, param_grid, cv=5, scoring=scorer, verbose=5)
    grid_search_without_clusters.fit(X_train_without_clusters, y_train)

    #get best models
    best_model_with_clusters = grid_search_with_clusters.best_estimator_
    best_model_without_clusters = grid_search_without_clusters.best_estimator_

    return best_model_with_clusters, best_model_without_clusters