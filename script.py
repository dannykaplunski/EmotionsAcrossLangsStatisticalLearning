import os 
import pandas as pd
from scipy.io import wavfile
import numpy as np
from scipy.signal import find_peaks

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

def get_bit_depth(file_path):
    try:
        sample_rate, data = wavfile.read(file_path)
    except:
        raise TypeError('Error reading file')
    bit_depth = data.dtype.itemsize * 8  
    num_samples = data.shape[0]
    duration = num_samples / sample_rate
    syllabic_rate = estimate_syllabic_rate(sample_rate, data)
    if len(data.shape) > 1:
        data = data[:,0] 
    zcr = get_zcr( data, duration)
    return sample_rate, bit_depth, duration, zcr, syllabic_rate

df = pd.DataFrame()
for file in filtered_files:
    new_df = pd.DataFrame()
    new_df['file_name'] = [file]
    try:
        new_df[['sample_rate', 'bit_depth', 'duration', 'zcr', 'syllabic_rate']] = get_bit_depth('wav-files/' + file)
    except TypeError as e:
        print(f'Error reading file {file}')
        continue
    df = df.append( new_df )
    
df = create_general_statistics(df)