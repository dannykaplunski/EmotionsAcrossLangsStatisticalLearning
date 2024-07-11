import numpy as np
import scipy
import scipy.signal
import librosa
import torch

other_emotions = ['ANG', 'HAP']
pred_emotion = 'SAD'

# each lambda is for getting the right value from the file name
create_country_name = lambda x: x.split('_')[0]
create_emotion_name = lambda x: x.split('_')[1]
create_gender_name = lambda x: x.split('_')[2]

# creates language, emotion and gender features
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
    if med_amp == 0:
        med_amp = mean_amp
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
    if maxpf == 0:
        maxpf = 1
    #define max f as highest f with 1/3 of the peak frequency
    isrelmax = torch.tensor(magspectrum > magmax/3).to(torch.float)
    relfs = torch.tensor(frequencies)*isrelmax #frequencies over 1/3 max mult by 1, others zeroed out
    #maximum relevant frequency
    maxf = max(relfs).numpy() 
    #define min f as lowest f with 1/3 of the peak (may often be 0)
    minf = min(relfs[relfs.nonzero()]).numpy() #excludes 0
    #the BW of the magnitudes is max-min
    bwfourier = maxf-minf
    if bwfourier[0] == 0:
        bwfourier[0] = 1
    return magmax, maxpf, maxf.item(), minf.item(), bwfourier.item()
    #max magnitude of fourier, f for max magnitude, highest relevant f, lowest relevant f, range (max-min) relevant f

#get basic spectral stats
def get_spec(clip, hoplen, samp_rate, nfft, maxf):
    #Get BW
    bw = np.max(librosa.feature.spectral_bandwidth(y=clip, sr=samp_rate, hop_length=hoplen)[0]) #sometimes gives 2 items in vector, one can be 0
    #Get the spectral rolloff
    specroll = np.max(librosa.feature.spectral_rolloff(y= clip, sr=samp_rate, n_fft=nfft, hop_length=hoplen, roll_percent=0.85)[0])#sometimes gives 2 items in vector, one can be 0
    #get fundamental frequency
    f0 = np.nanmean(librosa.pyin(y=clip, fmin = 10, fmax = maxf, sr=samp_rate)[0])
    f0 = 1 if np.isnan(f0) else f0
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
    if maxptf == None or maxptf == 0:
        maxptf = 1
    #sum up the energy for all time periods for each row
    rowsums = np.sum(fourier, axis=1)
    #get the max energy value
    rowmax = np.max(rowsums)
    #find the row it's in
    sumindex = np.where(rowsums == rowmax)
    #get the corresponding frequency for that row
    maxsumf=f_bins[int(sumindex[0])] #frequency of maximum power throughout the clip
    if maxsumf == 0:
        maxsumf = 1
    centroidvect = librosa.feature.spectral_centroid(y=clip, sr=samp_rate)
    meancent = np.mean(centroidvect)
    return maxptf, maxsumf, meancent
    #freq with max power at any point, freq with max overall power, spectral centroid