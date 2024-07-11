from dataframe_utils import create_categorical_columns, create_dataframe_from_test
import pickle
import pandas as pd 
import warnings
import sys
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
def main():
    
    if len(sys.argv) < 2:
        print("Usage: python predict_audio_file.py <filename>")
        sys.exit(1) 

    test_file = sys.argv[1]
    #create the dataframe and save it
    df = create_dataframe_from_test(test_file)

    with open('data/dummies_dictionary.pkl', 'rb') as f:  
        dummies_dictionary = pickle.load(f)
    features = []
    for col_first in ['kmeans_wolang_elbow', 'kmeans_wlang_elbow', 'cluster_by_hierarcal_5_features']:
        for col_second in dummies_dictionary[col_first]:
            features.append(col_first + '_' + str(col_second))
    
    df[features] = [0] * len(features)
    del(dummies_dictionary['kmeans_wolang_elbow'], dummies_dictionary['kmeans_wlang_elbow'],dummies_dictionary['cluster_by_hierarcal_5_features'])
    df = create_categorical_columns(df, dummies_dictionary)
    df = df.drop(columns=['file_name'])
    #get the columns order
    with open('data/columns.pkl', 'rb') as f:  # open a text file
        cols = pickle.load( f)

    #read scaler  and apply it
    with open('data/scaler.pkl', 'rb') as f:  
        scaler = pickle.load(f)

    df_scalled_array = scaler.transform(df[cols])
    df_scalled = pd.DataFrame(df_scalled_array, columns=cols)
    df_scalled = df_scalled.drop(features, axis=1)
    df_scalled.to_csv('data/df_scalled.csv', index=False)
    #read xgboost model and apply
    with open('data/xgboost_model_without_clusters.pkl', 'rb') as f:
        model = pickle.load(f)

    #get prediction and print it
    prediction = model.predict(df_scalled)[0]
    if prediction == 1:
        pred = 'Sad'
    else:
        pred = 'Not Sad'

    print(f'The prediction for the file {test_file} is {pred}')

if __name__ == '__main__':
    main()  