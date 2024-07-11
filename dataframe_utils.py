import os 
import pandas as pd
import numpy as np
import librosa
from scipy.io import wavfile
from feature_utils import other_emotions, pred_emotion, create_general_statistics, get_time_to_peak_amplitude, estimate_syllabic_rate, extract_mfcc_features, get_spectral_envelope, split_audio, get_mMa, get_fourier, get_spec, get_fpwr
import random
from sklearn.preprocessing import StandardScaler

# Get all files and filter them by categories
all_files = os.listdir('wav-files/')
all_categories = other_emotions + [pred_emotion]
filtered_files = [ x for x in all_files if 'deleted' not in x and x.split('_')[1] in all_categories ]

#create row for the dataframe from the audio file that was given in the path
def get_row(file_path):
    n = 5
    df = pd.DataFrame()
    try: #getting "clean" audio data with trimming, sample rate, and bit depth
        audio_data, sr = librosa.load(file_path, sr=None)
        audio_data, index =  librosa.effects.trim(audio_data, top_db=30)
        sample_rate, new = wavfile.read(file_path)
        bit_depth = new.dtype.itemsize * 8
    except:
        raise TypeError('Error reading file')
    df['sample_rate'] = [sr]
    df['bit_depth'] = [bit_depth]

    num_samples = len(audio_data)
    duration = num_samples / sr
    df['duration'] = [duration]

    syllabic_rate = estimate_syllabic_rate(audio_data, sr, duration )
    df['syllabic_rate'] = syllabic_rate

    #getting the mean and std of zero crossing rate of all the data
    zcr = librosa.feature.zero_crossing_rate(audio_data) 
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    df[['zcr_mean', 'zcr_std', 'zcr_frequncy']] = [zcr_mean, zcr_std, len(zcr) / duration]

    #get_spectral_envelope values
    mean_spectral_envelope, median_spectral_envelope, min_spectral_envelope, max_spectral_envelope, std_spectral_envelope, iqr_spectral_envelope, mean_diff_spectral_envelope, std_diff_spectral_envelope, skew_spectral_envelope, kurtosis_spectral_envelope = get_spectral_envelope(audio_data, sr )
    df[['mean_spectral_envelope', 'median_spectral_envelope', 'min_spectral_envelope', 'max_spectral_envelope', 'std_spectral_envelope', 'iqr_spectral_envelope', 'mean_diff_spectral_envelopeæ', 'std_diff_spectral_envelope', 'skew_spectral_envelope', 'kurtosis_spectral_envelope']] =   [mean_spectral_envelope, median_spectral_envelope, min_spectral_envelope, max_spectral_envelope, std_spectral_envelope, iqr_spectral_envelope, mean_diff_spectral_envelope, std_diff_spectral_envelope, skew_spectral_envelope, kurtosis_spectral_envelope]
    
    # get_time_to_peak_amplitude values
    time_of_peak, time_of_peak_precentage, time_from_peak_to_end = get_time_to_peak_amplitude(audio_data, sr, duration)
    df[['time_of_peak', 'time_of_peak_precentage', 'time_from_peak_to_end']] = [time_of_peak, time_of_peak_precentage, time_from_peak_to_end]
    
    # extract_mfcc_features values getting all the means and stds of each mfccs
    mfccs_mean, mfccs_std = extract_mfcc_features(audio_data, sr)
    df[['mfccs_mean_' + str(x) for x in range(len(mfccs_mean))] + ['mfccs_std_' + str(x) for x in range(len(mfccs_std))]] = np.concatenate((mfccs_mean, mfccs_std))
    
    #split the data into n parts and get the features of each part
    splited_data = split_audio(audio_data , n)   

    #get zcr features for each part
    zcr_means = [ np.mean(librosa.feature.zero_crossing_rate(x))  for x in splited_data]
    zcr_stds = [ np.std(librosa.feature.zero_crossing_rate(x))  for x in splited_data]
    zcr_frequncys = [ len(librosa.feature.zero_crossing_rate(x)) / (duration / (n))  for x in splited_data]
    df[['zcr_means_' + str(x) for x in range(len(splited_data))]] = zcr_means
    df[['zcr_stds_' + str(x) for x in range(len(splited_data))]] = zcr_stds
    df[['zcr_frequncys_' + str(x) for x in range(len(splited_data))]] = zcr_frequncys

    #get the differences between the features and see how they differ from the mean
    df[['zcr_diffs_' + str(x) + '_' + str(x+1) for x in range(len(splited_data) - 1)]] = np.diff(zcr_means)
    df[['mean_zcr', 'mean_zcr_diff']] =    [np.mean(zcr_means), np.mean(np.diff(zcr_means))]
    df[['zcr_from_average' + str(x)  for x in range(len(splited_data) - 1)]] =  df[['zcr_means_' + str(x)  for x in range(len(splited_data) - 1)]] - list(df['mean_zcr']) * (len(splited_data) - 1)
    df[['zcr_diffs_from_average' + str(x)  + '_' + str(x+1) for x in range(len(splited_data) - 1)]] =  df[['zcr_diffs_' + str(x)  + '_' + str(x+1) for x in range(len(splited_data) - 1)]]- list(df['mean_zcr_diff']) * (len(splited_data) - 1) 
    df[['zcr_stds_diffs_' + str(x) + '_' + str(x+1) for x in range(len(splited_data) - 1)]] = np.diff(zcr_stds)
    df[['std_zcr', 'stds_zcr_diff']] =    [np.mean(zcr_stds), np.mean(np.diff(zcr_stds))]
    df[['zcr_stds_from_average' + str(x)  for x in range(len(splited_data) - 1)]] =  df[['zcr_stds_' + str(x)  for x in range(len(splited_data) - 1)]] - list(df['std_zcr']) * (len(splited_data) - 1)
    df[['zcr_stds_diffs_from_average' + str(x)  + '_' + str(x+1) for x in range(len(splited_data) - 1)]] =  df[['zcr_stds_diffs_' + str(x)  + '_' + str(x+1) for x in range(len(splited_data) - 1)]]- list(df['stds_zcr_diff']) * (len(splited_data) - 1) 
    df[['zcr_frequncys_diffs_' + str(x) + '_' + str(x+1) for x in range(len(splited_data) - 1)]] = np.diff(zcr_frequncys)
    df[['frequncys_zcr', 'frequncys_zcr_diff']] =    [np.mean(zcr_frequncys), np.mean(np.diff(zcr_frequncys))]
    df[['zcr_frequncys_from_average' + str(x)  for x in range(len(splited_data) - 1)]] =  df[['zcr_frequncys_' + str(x)  for x in range(len(splited_data) - 1)]] - list(df['frequncys_zcr']) * (len(splited_data) - 1)
    df[['zcr_frequncys_diffs_from_average' + str(x)  + '_' + str(x+1) for x in range(len(splited_data) - 1)]] =  df[['zcr_frequncys_diffs_' + str(x)  + '_' + str(x+1) for x in range(len(splited_data) - 1)]]- list(df['frequncys_zcr_diff']) * (len(splited_data) - 1) 

    #get syllabic_rates  for each part and their diffs
    syllabic_rates = [estimate_syllabic_rate(x, sr, duration / (n)) for x in splited_data]
    df[['syllabic_rate_' + str(x) for x in range(len(splited_data))]] = syllabic_rates
    df[['syllabic_rate_diffs_' + str(x)  + '_' + str(x+1) for x in range(len(splited_data) - 1)]] = np.diff(syllabic_rates)
    df[['mean_syllabic_rate', 'mean_syllabic_rate_diff']] =    [np.mean(syllabic_rates), np.mean(np.diff(syllabic_rates))]
    df[['syllabic_rate_from_average' + str(x) for x in range(len(splited_data) - 1)]] =  df[['syllabic_rate_' + str(x)  for x in range(len(splited_data) - 1)]] - list(df['mean_syllabic_rate']) * (len(splited_data) - 1) 
    df[['syllabic_rate_diffs_from_average' + str(x) + '_' + str(x+1) for x in range(len(splited_data) - 1)]] =  df[['syllabic_rate_diffs_' + str(x)  + '_' + str(x+1) for x in range(len(splited_data) - 1)]] - list(df['mean_syllabic_rate_diff']) * (len(splited_data) - 1) 
    
    #get amp features
    max_amp, min_amp, mean_amp, med_amp = get_mMa(audio_data)
    df[['max_amp', 'min_amp', 'mean_amp', 'med_amp']] = [max_amp, min_amp, mean_amp, med_amp]
    #0) max signal amplitude, 1) min signal amplitude, 2) mean signal amplitude, 3) median signal amplitude
    #get fourier stats
    magmax, maxpf, maxf_items, minf_items, bwfourier = get_fourier(audio_data, sr)
    df[['magmax', 'maxpf', 'maxf_items', 'minf_items', 'bwfourier']] = magmax, maxpf, maxf_items, minf_items, bwfourier
    #0) max magnitude of fourier, 1) f for max magnitude, 2) highest relevant f, 3) lowest relevant f, 4) range (max-min) relevant f
    #setupfor spectral features:
    nfft = round(512*sr/22050) #512 recommended for voice at 22050 Hz
    hoplwhole = sr*3 #when we want data for the whole thing
    hopl = round(sr/5) #number of audio samples between adjacent STFT columns, we want 200ms chunks for the clips
    #frequency resolution
    fres = sr/nfft
    #max frequency
    maxf = maxf_items
    #get spectral stats
    bw, specroll, f0 = get_spec(audio_data, hoplwhole, sr, nfft, maxf)
    df[['bw', 'specroll', 'f0']] = [bw, specroll, f0]
    #0) spect energy BW, 1) spectral rolloff 85% energy, 2) fundamental freq (f0)
    
    #get normalized range/f0
    rangef0 = minf_items/maxpf
    df['rangef0'] = [rangef0]
    #get spectrogram stats
    maxptf, maxsumf, meancent = get_fpwr(audio_data, nfft, int(hoplwhole/2), fres, sr)
    df[['maxptf', 'maxsumf', 'meancent']] = [maxptf, maxsumf, meancent]
    # 0) freq with max power at any point, 1) freq with max overall power, 2) spectral centroid
    #and normalize them
    df[['maxptf/meancent', 'maxsumf/meancent']] =[maxptf/meancent, maxsumf/meancent]
    
    #get chroma data
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=hoplwhole, n_fft=nfft)
    chroma = np.mean(chroma, axis=1) #in case it gives 2 columns
    #can give multiple columns, so mean them.
    df[['chroma01', 'chroma02', 'chroma03', 'chroma04', 'chroma05', 'chroma06', 'chroma07', 'chroma08', 'chroma09', 'chroma10', 'chroma11', 'chroma12']]= chroma[0:12]
    
    #get data for the different segments of the file-----------------------
    #Makes a setup vector for segmenting the audio file into subclips
    #number of sections to split into +1
    #the shortest filess are 1 second and we don't want clips shorter than 200 ms since that's what the human ear picks up
    pps = np.round(num_samples/n) #points per section, with the last section getting the extras or shorted
    cutoffs = [int(pps * i) for i in range(0, n)]
    cutoffs.append(int(num_samples))
    #Run all of the subroutines to get the data for the segmented audio clips
    segdata = []
    new_df = pd.DataFrame()
    for i in range(0, n):
        #make the shortened clip
        shortclip = audio_data[cutoffs[i]:cutoffs[i+1]]
        #run it through the subroutines in order to get the data for each
        max_amp_short, min_amp_short, mean_amp_short, med_amp_short = get_mMa(shortclip) #4 columns
        #add wave normalization: max_amp, min_amp, mean_amp, med_amp
        n1 = max_amp_short/max(max_amp, np.abs(min_amp)) #to overall peak
        n2 = min_amp_short/max(max_amp, np.abs(min_amp)) #to overall peak
        n3 = mean_amp_short/mean_amp #to overall avg mag
        n4 = med_amp_short/med_amp #to overall median mag
        magmax_short, maxpf_short, maxf_items_short, minf_items_short, bwfourier_short = get_fourier(shortclip, sr) #5 columns
        #add fourier normalization 
        bw_short, specroll_short, f0_short  = get_spec(shortclip, hopl, sr, nfft, maxf) #3 columns
        cliprangef0 = bwfourier_short/maxpf_short #single value
        maxptf_short, maxsumf_short, meancent_short = get_fpwr(shortclip, nfft, hopl, fres, sr) #3 columns
        #add spectral normalization: maxptf/meancent, maxsumf/meancent
        #get chroma data
        chroma = librosa.feature.chroma_stft(y=shortclip, sr=sr, hop_length=hopl, n_fft=nfft)
        chroma_short = np.mean(chroma, axis=1) #in case it gives 2 columns
        #Outputs are all lists. Combine into 1 long row list and append as new row.
        df[['max_amp_short_' + str(i), 'min_amp_short_' + str(i), 'mean_amp_short_' + str(i), 'med_amp_short_' + str(i)]] = [max_amp_short, min_amp_short, mean_amp_short, med_amp_short]
        df[['magmax_short_' + str(i), 'maxpf_short_' + str(i), 'maxf_items_short_' + str(i), 'minf_items_short_' + str(i), 'bwfourier_short_' + str(i)]] = [magmax_short, maxpf_short, maxf_items_short, minf_items_short, bwfourier_short]
        df[['bw_short_' + str(i), 'specroll_short_' + str(i), 'f0_short_' + str(i), 'cliprangef0_short_' + str(i)]] = [bw_short, specroll_short, f0_short, cliprangef0]
        df[['maxptf_short_' + str(i), 'maxsumf_short_' + str(i), 'meancent_short_' + str(i)]] = [maxptf_short, maxsumf_short, meancent_short]
        df[['maxptf_short/meancent_short_' + str(i), 'maxsumf_short/meancent_short_' + str(i)]] = [maxptf_short/meancent_short, maxsumf_short/meancent_short]
        df[['n1_short_' + str(i), 'n2_short_' + str(i), 'n3_short_' + str(i), 'n4_short_' + str(i)]] = [n1, n2, n3, n4]
        df[['maxpf_short/maxpf_short_' + str(i) , 'maxf_items_short/maxf_items_short_' + str(i)]] = [maxpf_short/maxpf , maxf_items_short/maxf_items]
        df[['chroma01_short_' + str(i) , 'chroma02_short_' + str(i), 'chroma03_short_' + str(i), 'chroma04_short_' + str(i), 'chroma05_short_' + str(i), 'chroma06_short_' + str(i), 'chroma07_short_' + str(i), 'chroma08_short_' + str(i), 'chroma09_short_' + str(i), 'chroma10_short_' + str(i), 'chroma11_short_' + str(i), 'chroma12_short_' + str(i)]] = chroma_short[0:12]
    columns = ['max_amp_short', 'min_amp_short', 'mean_amp_short', 'med_amp_short', 'magmax_short', 'maxpf_short', 'maxf_items_short', 'minf_items_short', 'bwfourier_short', 'bw_short', 'specroll_short', 'f0_short', 'cliprangef0_short', 'maxptf_short', 'maxsumf_short', 'meancent_short', 'maxptf_short/meancent_short', 'maxsumf_short/meancent_short', 'n1_short', 'n2_short', 'n3_short', 'n4_short', 'maxpf_short/maxpf_short', 'maxf_items_short/maxf_items_short', 'chroma01_short', 'chroma02_short', 'chroma03_short', 'chroma04_short', 'chroma05_short', 'chroma06_short', 'chroma07_short', 'chroma08_short', 'chroma09_short', 'chroma10_short', 'chroma11_short', 'chroma12_short']

 # Compute differences
    for col in columns:
        for i in range(0, n - 1):  # up to 4 because we compute difference with the next section
            df[f'{col}_{i}_diff'] = df[f'{col}_{i+1}']- df[f'{col}_{i}'] #corrected order to new-old
            df[f'{col}_{i}_grad'] = df[f'{col}_{i}'] / df[f'{col}_{i+1}'].where(df[f'{col}_{i+1}'] != 0, other=0) #corrected order to new/old

    return df

# creates the df, if you want to do that you need to import this function 
# returns the completed dataframe and the files that got an unexpected error
def create_dataframe():
    #find all the files that didn't work
    not_ok_array = [] 
    df = pd.DataFrame()
    i = 0
    for file in filtered_files[0:]:
        try:
            #create new row from file
            new_df = get_row("wav-files/" + file) 
            new_df['file_name'] = [file]
            df = df.append( new_df )
        except:
            not_ok_array.append(file) 
        #print out progress
        if i % 100 == 0:
            print('done with ', i, ' files out of ', len(filtered_files), ' files')
        i += 1
    #find all the files that didn't work
    df = create_general_statistics(df)
    print("the files that got an unexpected error are: ", not_ok_array)
    return df

def create_dataframe_from_test(file_path):
    try:
        #create new row from file
        df = get_row(file_path) 
        df['file_name'] = [file_path]
        df['language'] = file_path.split('/')[1].split('_')[0]
        df['gender'] = file_path.split('/')[1].split('_')[1]
        return df
    except:
        raise TypeError('Error reading file')

#split the data to test and train data
def df_split(df):
    df_shuffled = df.sample(frac=1, random_state=random.seed())
    counts = df.groupby(['language', 'emotion']).size().reset_index(name='count')
    #define what percent of the data will be pulled out as test data
    pcttest = .1
    #define the distribution of files that we want in the final set
    counts['dset'] = (counts['count']*pcttest).astype(int)
    counts['movelist'] = 0 #set up row to store how many got moved
    testrows = [] #list of rows we want to remove
    #Make a list of the first rows that meet the criteria above
    for i in range(len(df_shuffled)):
        lang = df_shuffled['language'].loc[i]
        emot = df_shuffled['emotion'].loc[i]
        #check to see if we need more of that combo
        currentpull = counts.loc[(counts['language'] == lang) & (counts['emotion'] == emot), 'movelist'].item()
        needed = counts.loc[(counts['language'] == lang) & (counts['emotion'] == emot), 'dset'].item()
        if currentpull<needed:
            #put this row in the list for the test set
            testrows.append(i)
            #update the the counts df
            counts.loc[(counts['language'] == lang) & (counts['emotion'] == emot), 'movelist'] += 1
    testset = df_shuffled.loc[testrows]

    # Remove the moved rows from the original DataFrame
    trainset = df_shuffled.drop(testrows)
    #check the test set distribution
    return trainset, testset


def get_all_dummies(df):
    dummies_dictionary = {}
    dummies_dictionary['language'] = df['language'].unique()
    dummies_dictionary['gender'] = df['gender'].unique()
    dummies_dictionary['kmeans_wolang_elbow'] = df['kmeans_wolang_elbow'].unique()
    dummies_dictionary['kmeans_wlang_elbow'] = df['kmeans_wlang_elbow'].unique()
    dummies_dictionary['cluster_by_hierarcal_5_features'] = df['cluster_by_hierarcal_5_features'].unique()
    return dummies_dictionary

def contains_any(string, substrings):
    for substring in substrings:
        if substring in string:
            return True
    return False

def create_scaler(df):
    scaler = StandardScaler()
    df_scaled_array = scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled_array, columns=df.columns)
    return scaler, df

def create_categorical_columns(df, dummies_dictionary):
    keys = dummies_dictionary.keys()
    for column in keys:
        for value in dummies_dictionary[column]:
            df[column + '_' + str(value)] = [1 if x == value else 0 for x in df[column]]
        df = df.drop(columns=[column])
    return df