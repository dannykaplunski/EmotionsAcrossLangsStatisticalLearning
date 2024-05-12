import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
df = pd.read_csv('110524.csv')
df = df.drop('Unnamed: 0',axis = 1)
df_encoded = pd.get_dummies(df, columns=['language',	'emotion',	'gender'])
columns = ['bwfourier_short_3', 'chroma01_1', 'chroma03_1_diff', 'magmax_short_1', 'zcr_mean', 'mfccs_mean_5', 'chroma11_1', 'chroma10_1', 'chroma12_0', 'maxf_items_short_2', 'chroma12_1', 'chroma12_1_diff', 'chroma05_1_diff', 'chroma07_1_diff', 'maxpf_short_3', 'maxpf', 'chroma09_1_grad', 'chroma09_0_diff', 'chroma06_2', 'maxpf_short_4', 'minf_items', 'chroma02_1_diff', 'chroma05_3', 'chroma04_4', 'chroma02_1', 'chroma06_1_grad', 'chroma05_4', 'mfccs_std_2', 'bwfourier_short_1', 'f0_short_3', 'magmax_short_4', 'chroma10_2', 'mfccs_std_3', 'chroma09_2', 'chroma06_3', 'max_amp', 'mfccs_std_12', 'mfccs_mean_6', 'mfccs_mean_4', 'chroma08_3', 'mfccs_mean_2', 'chroma08_4', 'chroma09_4', 'n2_short_1_diff', 'f0_short_2', 'mfccs_std_11', 'maxf_items_short_1', 'mfccs_mean_0', 'bwfourier', 'minf_items_short_1', 'meancent_short_1', 'mfccs_std_5', 'chroma01_2_diff', 'minf_items_short_3', 'minf_items_short_4', 'chroma07_4', 'maxf_items_short_3', 'mfccs_std_8', 'chroma08_1', 'median_spectral_envelope', 'time_from_peak_to_end', 'chroma07_0_grad', 'chroma09_3', 'chroma08_1_diff', 'chroma06_1_diff', 'maxpf_short_1', 'chroma08_2', 'chroma02_1_grad', 'chroma07_3', 'min_spectral_envelope', 'f0_short_1', 'min_amp_short_2', 'chroma09_1', 'maxf_items', 'maxpf_short_0', 'maxptf_short_4', 'chroma07_1_grad', 'min_amp', 'f0_short_0', 'f0', 'bwfourier_short_2', 'maxpf_short_2', 'minf_items_short_2', 'chroma06_4', 'chroma12_4', 'chroma07_2', 'chroma10_0_diff', 'mfccs_mean_3']
pred_columns = [column for column in df_encoded.columns if 'emotion_' in column]

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, make_scorer

dictionary_for_pred_columns = {}
return_dictionary = {}
scorer = make_scorer(average_precision_score, needs_proba=True)

def func():
  for pred_column in pred_columns:
    dataframe_number_of_features_and_their_roc = pd.DataFrame(columns = ['score', 'features_in_store'])
    best_score, best_last_round_score = 0, 0.0001
    best_feature = ""
    features_in_store = []
    print(pred_column)

    while best_last_round_score != best_score:
      best_last_round_score = best_score
      p_values = {}

      for feature in [column for column in columns if column not in features_in_store]:
          model = LogisticRegression(max_iter=10000)
          cv_scores = cross_val_score(model, df_encoded[[feature] + features_in_store], df_encoded[pred_column], cv=5, scoring=scorer)

          score = cv_scores.mean()
          if score > best_score:
            best_score = score
            best_feature = feature

      if best_last_round_score != best_score:
        dataframe_number_of_features_and_their_roc.loc[ len(features_in_store)] = [best_score, features_in_store]
        features_in_store.append(best_feature)
        print(best_score, features_in_store)

    print("Best feature for " + str(pred_column) + " on precision_score: " + str(features_in_store) + " with the accuracy of " + str(best_score) )
    dictionary_for_pred_columns[pred_column] = (dataframe_number_of_features_and_their_roc, features_in_store, best_score)

  return dictionary_for_pred_columns
