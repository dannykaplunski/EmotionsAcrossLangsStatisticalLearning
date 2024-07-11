from dataframe_utils import create_dataframe,  create_categorical_columns, df_split, create_scaler, get_all_dummies, pred_emotion
from clusters import create_hierarchical_clustering_model, create_kmeans_model
from model import create_logistic_regressions, create_xgboost
import pickle
import pandas as pd 
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#create directory
Path("./data").mkdir(parents=True, exist_ok=True)

#create the dataframe and save it
print('Creating the dataframe')
df = create_dataframe()
df.to_csv('data/full_df.csv', index=False)
df = pd.read_csv('data/full_df.csv')

print('creating clustering algorithms')
df, centroids_5_clusters = create_hierarchical_clustering_model(df)
with open('data/centroids_5_clusters.pkl', 'wb') as f:  # open a text file
    pickle.dump(centroids_5_clusters, f)

df.to_csv('data/new_full_df.csv', index=False)

df, elbowclusters, elbowclusters2 = create_kmeans_model(df)

with open('data/elbowclusters.pkl', 'wb') as f:  # open a text file
    pickle.dump(elbowclusters, f)

with open('data/elbowclusters2.pkl', 'wb') as f:  # open a text file
    pickle.dump(elbowclusters2, f)

#df split and save it
print('splitting the data')
train_df,test_df = df_split(df)
#create dummies dictionary to generate the categorical columns
print('creating dummies')
dummies_dictionary = get_all_dummies(train_df)
with open('data/dummies_dictionary.pkl', 'wb') as f:  
    pickle.dump(dummies_dictionary, f)

train_df = create_categorical_columns(train_df, dummies_dictionary)
test_df = create_categorical_columns(test_df, dummies_dictionary)
train_df['emotion_' + pred_emotion] = (train_df['emotion'] ==  pred_emotion).astype(int)
test_df['emotion_' + pred_emotion] = (test_df['emotion'] ==  pred_emotion).astype(int)

train_df.to_csv('data/train_df_with_filename.csv', index=False)
test_df.to_csv('data/test_df_with_filename.csv', index=False)

# create the scaler and save it
drop_cols = ['file_name', 'emotion', 'emotion_' + pred_emotion]
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['emotion_' + pred_emotion]
X_test = test_df.drop(columns=drop_cols)
y_test = test_df['emotion_' + pred_emotion]

scaler, X_train_scalled = create_scaler(X_train)

with open('data/scaler.pkl', 'wb') as f:  # open a text file
    pickle.dump(scaler, f)

with open('data/columns.pkl', 'wb') as f:  # open a text file
    pickle.dump(X_test.columns, f)

X_test_array = scaler.transform(X_test)
X_test_scalled = pd.DataFrame(X_test_array, columns=X_test.columns)

#save train and test
X_train_scalled.to_csv('data/train_df.csv', index=False)
X_test_scalled.to_csv('data/test_df.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

#create logistic models and save them
print('creating logistic regression models')
best_model_with_clusters, best_model_without_clusters = create_logistic_regressions(X_train_scalled, y_train)

with open('data/logistic_model_with_clusters.pkl', 'wb') as f:  # open a text file
    pickle.dump(best_model_with_clusters, f)

with open('data/logistic_model_without_clusters.pkl', 'wb') as f: 
    pickle.dump(best_model_without_clusters, f)

print('creating xgboost models')
#create xgboost models and save them
best_model_with_clusters, best_model_without_clusters = create_xgboost(X_train_scalled, y_train)

with open('data/xgboost_model_with_clusters.pkl', 'wb') as f:  
    pickle.dump(best_model_with_clusters, f)

with open('data/xgboost_model_without_clusters.pkl', 'wb') as f:
    pickle.dump(best_model_without_clusters, f)

