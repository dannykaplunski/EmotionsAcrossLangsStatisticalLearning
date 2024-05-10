import os 
import pandas as pd
import scipy
import numpy as np
import librosa
import torch
import scipy.signal
from scipy.io import wavfile

# Get all files and filter them by categories
all_files = os.listdir('wav-files/')
all_categories = ['ANG', 'HAP', 'SAD']
filtered_files = [ x for x in all_files if x.split('_')[1] in all_categories ]

# Create categories
create_country_name = lambda x: x.split('_')[0]
create_emotion_name = lambda x: x.split('_')[1]
create_gender_name = lambda x: x.split('_')[2]

def create_general_statistics(df):
    df['language'] = df['file_name'].apply(create_country_name)
    df['emotion'] = df['file_name'].apply(create_emotion_name)
    df['gender'] = df['file_name'].apply(create_gender_name)
    return df

# get the time to peak of audio data in seconds, precentage and from in seconds from the end
def get_time_to_peak_amplitude(audio_data, sr, duration):
    peak_amplitude_index = np.argmax(np.abs(audio_data))
    time_of_peak = peak_amplitude_index / sr
    time_of_peak_precentage = time_of_peak / sr
    time_from_peak_to_end = duration - time_of_peak
    return time_of_peak, time_of_peak_precentage, time_from_peak_to_end

# Estimate the syllabic rate of the audio data
def estimate_syllabic_rate(audio_data, sr, duration_in_seconds):
    # Pre-emphasize the audio data to increase the accuracy of energy estimation
    audio_preemphasized = np.append(audio_data[0], audio_data[1:] - 0.97 * audio_data[:-1])

    frame_length = int(0.025 * sr)  # 25ms window 
    hop_length = int(0.010 * sr) #10ms hop length
    energy = np.array([
        np.sum(np.abs(audio_preemphasized[i:i + frame_length]**2))
        for i in range(0, len(audio_preemphasized), hop_length)
    ]) # Energy is calculated as the sum of squares of the amplitudes in the frame.
    peaks, _ = scipy.signal.find_peaks(energy, height=np.mean(energy), distance=frame_length) # Find number of peaks in the energy signal it is the number of syllables
    num_syllables = len(peaks)
    syllabic_rate = num_syllables / duration_in_seconds # change to rate
    return syllabic_rate

# The spectral centroid is a measure used in digital signal processing to characterize a spectrum. It indicates the center of mass of the spectrum and is often associated with the perceived brightness of a sound.
# librosa.feature.spectral_centroid returns array so we need to calculate the mean, median, min, max, std, iqr, mean diff, std diff, skew and kurtosis of this array
def get_spectral_envelope(y, sr ):
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0] 
    return np.mean(spectral_centroids), np.median(spectral_centroids), np.min(spectral_centroids), np.max(spectral_centroids), np.std(spectral_centroids), np.percentile(spectral_centroids, 75) - np.percentile(spectral_centroids, 25), np.mean(np.diff(spectral_centroids)), np.std(np.diff(spectral_centroids)), scipy.stats.skew(spectral_centroids), scipy.stats.kurtosis(spectral_centroids)
    # mean, median, min, max, std, iqr, mean diff, std diff, skew and kurtosis of spectral_centroids

#Mel-frequency cepstral coefficients MFCCs - effectively represent the short-term power spectrum of sound, and are used in automatic speech and speaker recognition.
# Mel Filterbank: The audio signal is first converted to the Mel scale, which approximates the human auditory system's response more closely than the linearly-spaced frequency bands used in the Fourier transform.
# Logarithmic Scaling: Taking the logarithm of the powers at each of the mel frequencies, this operation mimics the human ear's perception of loudness.
# Discrete Cosine Transform (DCT): This is applied to the log mel powers, which decorrelates the energy distribution across filter bands and compresses the most relevant frequency components into fewer coefficients.
# for every mfcc (13) we calculate the mean and std of it in the audio data
def extract_mfcc_features(y, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = mfccs.mean(axis=1)
    mfccs_std   = mfccs.std(axis=1)
    return mfccs_mean, mfccs_std
    # mfccs_mean: the mean of each mfcc (n_mfcc the number of mfccs, 
    # mfccs_std: the std of each mfcc

#splitting the audio data into n parts ****what is y?**** answer : you right, changed it to the audio_data
def split_audio(audio_data, num_parts):
    total_samples = len(audio_data)
    part_length = total_samples // num_parts
    parts = [audio_data[i * part_length:(i + 1) * part_length] for i in range(num_parts)]
    if total_samples % num_parts != 0:
        parts[-1] = np.concatenate((parts[-1], audio_data[num_parts * part_length:]))
    return parts
    #audio_data array splitted to num_parts parts

#Gets basic stats of the audio wave
def get_mMa(clip):
    max_amp = np.max(clip)
    min_amp = np.min(clip)
    mean_amp = np.mean(clip) 
    med_amp = np.median(clip)
    return max_amp, min_amp, mean_amp, med_amp 
    #max signal amplitude, min signal amplitude, mean signal amplitude, median signal amplitude

#Gets the fourier transform information
def get_fourier(clip, samp_rate):
    #Get the Fourier transform
    fourier = np.fft.fft(clip)
    #Get the spectrum
    magspectrum = np.abs(fourier)
    #get the frequencies that correspond to the fourier magnitudes
    frequencies = abs(np.fft.fftfreq(len(fourier), d=1/samp_rate)) 
    #get the f with max magnitude
    magmax = np.max(magspectrum) 
    #find the index
    fmaxindex = np.where(magspectrum == magmax)
    #get the corresponding frequency of maximum power at any point during the clip. If there are 2 points, pick one.
    maxpf = frequencies[int(np.max(fmaxindex[0]))] 
    #define max f as highest f with 1/3 of the peak frequency
    isrelmax = torch.tensor(magspectrum > magmax/3).to(torch.float)
    relfs = torch.tensor(frequencies)*isrelmax #frequencies over 1/3 max mult by 1, others zeroed out
    #maximum relevant frequency
    maxf = max(relfs).numpy() 
    #define min f as lowest f with 1/3 of the peak (may often be 0)
    minf = min(relfs).numpy()
    #the BW of the magnitudes is max-min
    bwfourier = maxf-minf
    return magmax, maxpf, maxf.item(), minf.item(), bwfourier
    #max magnitude of fourier, f for max magnitude, highest relevant f, lowest relevant f, range (max-min) relevant f

#get basic spectral stats
def get_spec(clip, hoplen, samp_rate, nfft, maxf):
    #Get BW
    bw = np.max(librosa.feature.spectral_bandwidth(y=clip, sr=samp_rate, hop_length=hoplen)[0]) #sometimes gives 2 items in vector, one can be 0
    #Get the spectral rolloff
    specroll = np.max(librosa.feature.spectral_rolloff(y= clip, sr=samp_rate, n_fft=nfft, hop_length=hoplen, roll_percent=0.85)[0])#sometimes gives 2 items in vector, one can be 0
    #get fundamental frequency
    try:
        f0 = np.nanmean(librosa.pyin(y=clip, fmin = 60, fmax = maxf, sr=samp_rate)[0])
    except RuntimeWarning:
    #sometimes it doesn't give a fundamental frequency and thinks it's nothing
        f0 = 0
    return bw, specroll, f0
    #spect energy BW, spectral rolloff 85% energy, fundamental freq (f0)

#get the frequencies with the max power (spectrogram) and spectral centroid
def get_fpwr(clip, nfft, hoplen, f_res, samp_rate):
    fourier = librosa.stft(clip, n_fft=nfft, hop_length=hoplen)
    #convert it from raw values to dB (log)
    findb = librosa.amplitude_to_db(abs(fourier))
    #0 is the frequency bins, 1 is the time stamp
    #frequency corresponding to each row
    f_bins = [i * f_res for i in range(fourier.shape[0])]
    #get the max energy value
    overallmaxdb = np.max(findb) 
    #find the row it's in
    overallindex = np.where(findb == overallmaxdb)
    #get the corresponding frequency for that row
    maxptf=f_bins[int(overallindex[0])] #frequency of maximum power at any point during the clip
    #sum up the energy for all time periods for each row
    rowsums = np.sum(fourier, axis=1)
    #get the max energy value
    rowmax = np.max(rowsums)
    #find the row it's in
    sumindex = np.where(rowsums == rowmax)
    #get the corresponding frequency for that row
    maxsumf=f_bins[int(sumindex[0])] #frequency of maximum power throughout the clip
    centroidvect = librosa.feature.spectral_centroid(y=clip, sr=samp_rate)
    meancent = np.mean(centroidvect)
    return maxptf, maxsumf, meancent
    #freq with max power at any point, freq with max overall power, spectral centroid

def get_features(file_path):
    n = 5
    df = pd.DataFrame()
    try: #getting "clean" audio data with trimming, sample rate, and bit depth
        audio_data, sr = librosa.load(file_path)
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
    df[['magmax', 'maxpf', 'maxf_items', 'minf_items', 'bwfourier']] = [magmax, maxpf, maxf_items, minf_items, bwfourier]
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
    maxptf, maxsumf, meancent = get_fpwr(audio_data, nfft, hoplwhole, fres, sr)
    df[['maxptf', 'maxsumf', 'meancent']] = [maxptf, maxsumf, meancent]
    # 0) freq with max power at any point, 1) freq with max overall power, 2) spectral centroid
    #and normalize them
    df[['maxptf/meancent', 'maxsumf/meancent']] =[maxptf/meancent, maxsumf/meancent]
    
    #get chroma data
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=hoplwhole, n_fft=nfft)
    #can give 1 or 2 columns. need to either max or avg them, not sure which makes sense yet
    
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
        #Outputs are all lists. Combine into 1 long row list and append as new row.
        df[['max_amp_short_' + str(i), 'min_amp_short_' + str(i), 'mean_amp_short_' + str(i), 'med_amp_short_' + str(i)]] = [max_amp_short, min_amp_short, mean_amp_short, med_amp_short]
        df[['magmax_short_' + str(i), 'maxpf_short_' + str(i), 'maxf_items_short_' + str(i), 'minf_items_short_' + str(i), 'bwfourier_short_' + str(i)]] = [magmax_short, maxpf_short, maxf_items_short, minf_items_short, bwfourier_short]
        df[['bw_short_' + str(i), 'specroll_short_' + str(i), 'f0_short_' + str(i), 'cliprangef0_short_' + str(i)]] = [bw_short, specroll_short, f0_short, cliprangef0]
        df[['maxptf_short_' + str(i), 'maxsumf_short_' + str(i), 'meancent_short_' + str(i)]] = [maxptf_short, maxsumf_short, meancent_short]
        df[['maxptf_short/meancent_short_' + str(i), 'maxsumf_short/meancent_short_' + str(i)]] = [maxptf_short/meancent_short, maxsumf_short/meancent_short]
        df[['n1_short_' + str(i), 'n2_short_' + str(i), 'n3_short_' + str(i), 'n4_short_' + str(i)]] = [n1, n2, n3, n4]
        df[['maxpf_short/maxpf_short_' + str(i) , 'maxf_items_short/maxf_items_short_' + str(i)]] = [maxpf_short/maxpf , maxf_items_short/maxf_items]
    columns = ['max_amp_short', 'min_amp_short', 'mean_amp_short', 'med_amp_short', 'magmax_short', 'maxpf_short', 'maxf_items_short', 'minf_items_short', 'bwfourier_short', 'bw_short', 'specroll_short', 'f0_short', 'cliprangef0_short', 'maxptf_short', 'maxsumf_short', 'meancent_short', 'maxptf_short/meancent_short', 'maxsumf_short/meancent_short', 'n1_short', 'n2_short', 'n3_short', 'n4_short', 'maxpf_short/maxpf_short', 'maxf_items_short/maxf_items_short']

    # Compute differences
    for col in columns:
        for i in range(0, n - 1):  # up to 4 because we compute difference with the next section
            df[f'{col}_{i}_diff'] = df[f'{col}_{i}'] - df[f'{col}_{i+1}']
            df[f'{col}_{i}_grad'] = df[f'{col}_{i}'] / df[f'{col}_{i+1}']
    #****need to tell it what to do when dividing by 0****

# creates the df, if you want to do that you need to import this function 
def df_func():
    not_ok_array = []
    df = pd.DataFrame()
    for file in filtered_files[0:]:
        new_df = pd.DataFrame()
        try:
            new_df = get_features("wav-files/" + file)
            new_df['file_name'] = [file]
            df = df.append( new_df )
        except TypeError:
            continue
        except librosa.util.exceptions.ParameterError:
            continue
        except:
            not_ok_array.append(file)
            break
    df = create_general_statistics(df)
    return df, not_ok_array
#returns the completed dataframe and the files that got an unexpected error

#for debbuging, if you need to see what happens with a specific file, if it gives an exception or just to see its results you import it and get the file
def specific_df_func(name):
    new_df = pd.DataFrame()
    new_df = get_features("wav-files/" + name)
    new_df['file_name'] = [name]
    return new_df

#'URD_ANG_M_(. URDU-Dataset Angry SM4_F15_A067.wav).wav'
#Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/Users/danny/learnings/EmotionsAcrossLangsStatisticalLearning/script.py", line 287, in specific_df_func
#     new_df = get_features("wav-files/" + name)
#   File "/Users/danny/learnings/EmotionsAcrossLangsStatisticalLearning/script.py", line 202, in get_features
#     maxptf, maxsumf, meancent = get_fpwr(audio_data, nfft, hoplwhole, fres, sr)
#   File "/Users/danny/learnings/EmotionsAcrossLangsStatisticalLearning/script.py", line 103, in get_fpwr
#     maxptf=f_bins[int(overallindex[0])] 

# 'URD_ANG_M_(. URDU-Dataset Angry SM1_F1_A01.wav).wav'
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/Users/danny/learnings/EmotionsAcrossLangsStatisticalLearning/script.py", line 287, in specific_df_func
#     new_df = get_features("wav-files/" + name)
#   File "/Users/danny/learnings/EmotionsAcrossLangsStatisticalLearning/script.py", line 194, in get_features
#     bw, specroll, f0 = get_spec(audio_data, hoplwhole, sr, nfft, maxf)
#   File "/Users/danny/learnings/EmotionsAcrossLangsStatisticalLearning/script.py", line 92, in get_spec
#     f0 = np.nanmean(librosa.pyin(y=clip, fmin = 60, fmax = maxf, sr=samp_rate)[0])
#   File "/Users/danny/Library/Python/3.9/lib/python/site-packages/librosa/core/pitch.py", line 779, in pyin
#     __check_yin_params(
#   File "/Users/danny/Library/Python/3.9/lib/python/site-packages/librosa/core/pitch.py", line 957, in __check_yin_params
#     raise ParameterError(f"fmin={fmin:.3f} must be less than fmax={fmax:.3f}")
# librosa.util.exceptions.ParameterError: fmin=60.000 must be less than fmax=0.000
