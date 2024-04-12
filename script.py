import os 
import pandas as pd
import scipy
from scipy.io import wavfile
import numpy as np
from scipy.signal import find_peaks, convolve
from scipy.signal.windows import gaussian
import librosa
import torch

all_files = os.listdir('wav-files/')
all_categories = ['ANG', 'HAP', 'SAD']
filtered_files = [ x for x in all_files if x.split('_')[1] in all_categories ]

create_country_name = lambda x: x.split('_')[0]
create_emotion_name = lambda x: x.split('_')[1]
create_gender_name = lambda x: x.split('_')[2]

def create_general_statistics(df):
    df['language'] = df['file_name'].apply(create_country_name)
    df['emotion'] = df['file_name'].apply(create_emotion_name)
    df['gender'] = df['file_name'].apply(create_gender_name)
    return df

def get_zcr(data, duration):
    zero_crossings = np.sum(np.abs(np.diff(np.sign(data)))) / 2
    zcr = zero_crossings / duration
    return zcr

def get_time_to_peak_amplitude(data, sample_rate, duration):
    peak_amplitude_index = np.argmax(np.abs(data))
    time_of_peak = peak_amplitude_index / sample_rate
    time_of_peak_precentage = time_of_peak / duration
    time_from_peak_to_end = duration - time_of_peak
    return time_of_peak, time_of_peak_precentage, time_from_peak_to_end

def estimate_syllabic_rate(sample_rate, data):
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    data = data / np.max(np.abs(data))
    frame_size = int(sample_rate * 0.05)  
    frame_step = int(frame_size / 2)  
    energy = np.array([
        np.sum(np.abs(data[i:i+frame_size])**2)
        for i in range(0, len(data) - frame_size + 1, frame_step)
    ])
    energy = energy / np.max(energy)
    peaks, _ = find_peaks(energy, height=0.1)  
    num_syllables = len(peaks)
    duration_in_seconds = len(data) / sample_rate
    syllabic_rate = num_syllables / duration_in_seconds
    return syllabic_rate

def get_spectral_envelope(y, sr ):
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0] 
    return np.mean(spectral_centroids), np.median(spectral_centroids), np.min(spectral_centroids), np.max(spectral_centroids), np.std(spectral_centroids), np.percentile(spectral_centroids, 75) - np.percentile(spectral_centroids, 25), np.mean(np.diff(spectral_centroids)), np.std(np.diff(spectral_centroids)), scipy.stats.skew(spectral_centroids), scipy.stats.kurtosis(spectral_centroids)


def extract_mfcc_features(y, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = mfccs.mean(axis=1)
    mfccs_std   = mfccs.std(axis=1)
    return mfccs_mean, mfccs_std

def split_audio(y, num_parts):
    total_samples = len(y)
    part_length = total_samples // num_parts
    parts = [y[i * part_length:(i + 1) * part_length] for i in range(num_parts)]
    if total_samples % num_parts != 0:
        parts[-1] = np.concatenate((parts[-1], y[num_parts * part_length:]))
    return parts

def get_mMa(clip):
    max_amp = np.max(clip)
    min_amp = np.min(clip)
    mean_amp = np.mean(clip) 
    med_amp = np.median(clip)
    return [max_amp, min_amp, mean_amp, med_amp]

def get_fourier(clip, samp_rate):
    fourier = np.fft.fft(clip)
    magspectrum = np.abs(fourier)
    frequencies = abs(np.fft.fftfreq(len(fourier), d=1/samp_rate)) 
    magmax = np.max(magspectrum) 
    fmaxindex = np.where(magspectrum == magmax)
    maxpf = frequencies[int(np.max(fmaxindex[0]))] 
    isrelmax = torch.tensor(magspectrum > magmax/3).to(torch.float)
    relfs = torch.tensor(frequencies)*isrelmax 
    maxf = max(relfs).numpy() 
    minf = min(relfs).numpy()
    bwfourier = maxf-minf
    fourierout = [magmax, maxpf, maxf.item(), minf.item(), bwfourier]
    return fourierout

def get_spec(clip, hoplen, samp_rate, nfft, maxf):
    bw = np.max(librosa.feature.spectral_bandwidth(y=clip, sr=samp_rate, hop_length=hoplen)[0]) #sometimes gives 2 items in vector, one can be 0
    specroll = np.max(librosa.feature.spectral_rolloff(y= clip, sr=samp_rate, n_fft=nfft, hop_length=hoplen, roll_percent=0.85)[0])#sometimes gives 2 items in vector, one can be 0
    try:
        f0 = np.nanmean(librosa.pyin(y=clip, fmin = 60, fmax = maxf, sr=samp_rate)[0])
    except RuntimeWarning:
        f0 = 0
    spect = [bw, specroll, f0]
    return spect

def get_fpwr(clip, nfft, hoplen, f_res, samp_rate):
    fourier = librosa.stft(clip, n_fft=nfft, hop_length=hoplen)
    findb = librosa.amplitude_to_db(abs(fourier))
    f_bins = [i * f_res for i in range(fourier.shape[0])]
    overallmaxdb = np.max(findb) 
    overallindex = np.where(findb == overallmaxdb)
    maxptf=f_bins[int(overallindex[0])] 
    rowsums = np.sum(fourier, axis=1)
    rowmax = np.max(rowsums)
    sumindex = np.where(rowsums == rowmax)
    maxsumf=f_bins[int(sumindex[0])] 
    centroidvect = librosa.feature.spectral_centroid(y=clip, sr=samp_rate)
    meancent = np.mean(centroidvect)
    return [maxptf, maxsumf, meancent]

def diffgrad(intensor):
    cols = intensor.shape[1]
    rows = intensor.shape[0]
    zerost = torch.zeros(1, cols)
    lastt = torch.cat((intensor, zerost), dim=0)
    nextt = torch.cat((zerost, intensor), dim=0)
    diffs = nextt-lastt
    grads = nextt/lastt 
    tensoroutput = torch.cat((diffs[0:rows], grads[0:rows]), dim=1)
    return tensoroutput

def get_features(file_path):
    df = pd.DataFrame()
    try:
        sample_rate, data = wavfile.read(file_path)
        audio_data, sr = librosa.load(file_path)
    except:
        raise TypeError('Error reading file')
    df['sample_rate'] = [sample_rate]
    bit_depth = data.dtype.itemsize * 8  
    df['bit_depth'] = [bit_depth]
    num_samples = data.shape[0] 
    duration = num_samples / sample_rate
    df['duration'] = [duration]
    syllabic_rate = estimate_syllabic_rate(sample_rate, data)
    df['syllabic_rate'] = syllabic_rate
    if len(data.shape) > 1:
        data = data[:,0] 
    zcr = get_zcr( data, duration)
    df['zcr'] = [zcr]
    mean_spectral_envelope, median_spectral_envelope, min_spectral_envelope, max_spectral_envelope, std_spectral_envelope, iqr_spectral_envelope, mean_diff_spectral_envelope, std_diff_spectral_envelope, skew_spectral_envelope, kurtosis_spectral_envelope= get_spectral_envelope(audio_data, sr )
    df[['mean_spectral_envelope', 'median_spectral_envelope', 'min_spectral_envelope', 'max_spectral_envelope', 'std_spectral_envelope', 'iqr_spectral_envelope', 'mean_diff_spectral_envelopeæ', 'std_diff_spectral_envelope', 'skew_spectral_envelope', 'kurtosis_spectral_envelope']] =   [mean_spectral_envelope, median_spectral_envelope, min_spectral_envelope, max_spectral_envelope, std_spectral_envelope, iqr_spectral_envelope, mean_diff_spectral_envelope, std_diff_spectral_envelope, skew_spectral_envelope, kurtosis_spectral_envelope]
    time_of_peak, time_of_peak_precentage, time_from_peak_to_end = get_time_to_peak_amplitude(data, sample_rate, duration)
    df[['time_of_peak', 'time_of_peak_precentage', 'time_from_peak_to_end']] = [time_of_peak, time_of_peak_precentage, time_from_peak_to_end]
    mfccs_mean, mfccs_std = extract_mfcc_features(audio_data, sr)
    df[['mfccs_mean_' + str(x) for x in range(len(mfccs_mean))] + ['mfccs_std_' + str(x) for x in range(len(mfccs_std))]] = np.concatenate((mfccs_mean, mfccs_std))
    splited_data = split_audio(audio_data , 6)   
    zcrs = [get_zcr(x, duration / 6) for x in splited_data]
    syllabic_rates = [estimate_syllabic_rate(sample_rate, x) for x in splited_data]
    df[['zcr_' + str(x) for x in range(len(splited_data))]] = zcrs
    df[['zcr_diffs_' + str(x) for x in range(len(splited_data) - 1)]] = np.diff(zcrs)
    df[['mean_zcr', 'mean_zcr_diff']] =    [np.mean(zcrs), np.mean(np.diff(zcrs))]
    df[['zcr_from_average' + str(x) for x in range(len(splited_data) - 1)]] =  df[['zcr_' + str(x) for x in range(len(splited_data) - 1)]] - list(df['mean_zcr']) * (len(splited_data) - 1)
    df[['zcr_diffs_from_average' + str(x) for x in range(len(splited_data) - 1)]] =  df[['zcr_diffs_' + str(x) for x in range(len(splited_data) - 1)]]- list(df['mean_zcr_diff']) * (len(splited_data) - 1) 
    df[['syllabic_rate_' + str(x) for x in range(len(splited_data))]] = syllabic_rates
    df[['syllabic_rate_diffs_' + str(x) for x in range(len(splited_data) - 1)]] = np.diff(syllabic_rates)
    df[['mean_syllabic_rate', 'mean_syllabic_rate_diff']] =    [np.mean(syllabic_rates), np.mean(np.diff(syllabic_rates))]
    df[['syllabic_rate_from_average' + str(x) for x in range(len(splited_data) - 1)]] =  df[['syllabic_rate_' + str(x) for x in range(len(splited_data) - 1)]] - list(df['mean_syllabic_rate']) * (len(splited_data) - 1) 
    df[['syllabic_rate_diffs_from_average' + str(x) for x in range(len(splited_data) - 1)]] =  df[['syllabic_rate_diffs_' + str(x) for x in range(len(splited_data) - 1)]] - list(df['mean_syllabic_rate_diff']) * (len(splited_data) - 1) 
    pts = len(audio_data)
    secs = pts/sample_rate
    #get data for the whole audio file************************
    #get the wave stats
    wavestats = get_mMa(audio_data)
    #0) max signal amplitude, 1) min signal amplitude, 2) mean signal amplitude, 3) median signal amplitude
    #get fourier stats
    fourierstats = get_fourier(audio_data, sample_rate)
    #0) max magnitude of fourier, 1) f for max magnitude, 2) highest relevant f, 3) lowest relevant f, 4) range (max-min) relevant f
    #setupfor spectral features:
    nfft = round(512*sample_rate/22050) #512 recommended for voice at 22050 Hz
    hoplwhole = sample_rate*3 #when we want data for the whole thing
    hopl = round(sample_rate/5) #number of audio samples between adjacent STFT columns, we want 200ms chunks for the clips
    #frequency resolution
    fres = sample_rate/nfft
    #max frequency
    maxf = fourierstats[2]
    #get spectral stats
    specstats = get_spec(audio_data, hoplwhole, sample_rate, nfft, maxf)
    #0) spect energy BW, 1) spectral rolloff 85% energy, 2) fundamental freq (f0)
    #get normalized range/f0
    rangef0 = fourierstats[4]/specstats[2]
    #get spectrogram stats
    spectrostats = get_fpwr(audio_data, nfft, hoplwhole, fres, sample_rate)
    # 0) freq with max power at any point, 1) freq with max overall power, 2) spectral centroid
    #and normalize them
    nspectrostats = [spectrostats[0]/spectrostats[2], spectrostats[1]/spectrostats[2]]
    #get chroma data
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate, hop_length=hoplwhole, n_fft=nfft)
    #can give 1 or 2 columns. need to either max or avg them, not sure which makes sense yet
    #get data for the different segments of the file************************
    #Makes a setup vector for segmenting the audio file into subclips
    n = 6 #number of sections to split into +1
    #the shortest filess are 1 second and we don't want clips shorter than 200 ms since that's what the human ear picks up
    pps = np.round(pts/n) #points per section, with the last section getting the extras or shorted
    cutoffs = [int(pps * i) for i in range(0, n-1)]
    cutoffs.append(int(pts))
    #Run all of the subroutines to get the data for the segmented audio clips
    segdata = []
    for i in range(0, n-1):
        #make the shortened clip
        shortclip = audio_data[cutoffs[i]:cutoffs[i+1]]
        #run it through the subroutines in order to get the data for each
        clipdata= get_mMa(shortclip) #4 columns
        #add wave normalization:
        n1 = clipdata[0]/max(wavestats[0], np.abs(wavestats[1])) #to overall peak
        n2 = clipdata[1]/max(wavestats[0], np.abs(wavestats[1])) #to overall peak
        n3 = clipdata[2]/wavestats[2] #to overall avg mag
        n4 = clipdata[3]/wavestats[3] #to overall median mag
        clipndata = [n1, n2, n3, n4] #4 columns
        clipfourier = get_fourier(shortclip, sample_rate) #5 columns
        #add fourier normalization
        clipnfourier = [clipfourier[1]/fourierstats[1] , clipfourier[2]/fourierstats[2]] #2 columns, normalized to overall
        clipspec = get_spec(shortclip, hopl, sample_rate, nfft, maxf) #3 columns
        cliprangef0 = clipfourier[4]/clipspec[2] #single value
        clipspectro = get_fpwr(shortclip, nfft, hopl, fres, sample_rate) #3 columns
        #add spectral normalization:
        clipnspectro = [clipspectro[0]/clipspectro[2], clipspectro[1]/clipspectro[2]] #2 columns, normalized to f0 of clip
        #Outputs are all lists. Combine into 1 long row list and append as new row.
        segdata.append(clipdata + clipfourier + clipspec + [cliprangef0] + clipspectro + clipnspectro + clipndata + clipnfourier) #18 columns
        #0-3: wave stats, 4-8: fourier stats, 9-11: spectrum stats, 12: norm range, 
        #13-15: spectrogram stats, 16-17: normalized spectrogram, 
        #18-21: wave stats norm to overall, 22-23: fourier norm to overall
    segmenttensor = torch.tensor(segdata)    
    #get data for changes between segments************************
    dg = diffgrad(segmenttensor[:,0:17]) #taking the diff of grad between data normalized to the overall would just be a linear combo of the data
    #make one big output for the whole file*************************
    #we have all of the data from the whole audio file, the data for the segments, and the data for the change between segments
    #we want to flatten it into a single row
    wholedata = torch.tensor(wavestats + fourierstats + specstats + [rangef0] + spectrostats + nspectrostats)
    segmentdata = torch.reshape(torch.cat((segmenttensor, dg), dim=1), (1,290))[0]
    final_output = torch.cat((wholedata, segmentdata), dim=0)
    numpy_array = final_output.cpu().numpy()
    df[[f"Tensor_Col_{x+1}" for x in range(len(numpy_array))]] = numpy_array

    return df

df = pd.DataFrame()
no_good_files = []
for file in filtered_files[2648:]:
    new_df = pd.DataFrame()
    try:
        new_df = get_features("wav-files/" + file)
        new_df['file_name'] = [file]
    except TypeError as e:
        print(f'Error reading file {file}')
        continue
    except librosa.ParameterError as e:
        no_good_files.append(file)
        continue
    df = df.append( new_df )
df = create_general_statistics(df)