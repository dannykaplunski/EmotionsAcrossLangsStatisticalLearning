import os 
import pandas as pd
from scipy.io import wavfile

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

def get_bit_depth(file_path):
    try:
        sample_rate, data = wavfile.read(file_path)
    except:
        raise TypeError('Error reading file')
    bit_depth = data.dtype.itemsize * 8  # Convert bytes to bits
    num_samples = data.shape[0]
    duration = num_samples / sample_rate
    return sample_rate, bit_depth, duration

df = pd.DataFrame()
for file in filtered_files:
    new_df = pd.DataFrame()
    new_df['file_name'] = [file]
    try:
        new_df[['sample_rate', 'bit_depth', 'duration']] = get_bit_depth('wav-files/' + file)
    except TypeError as e:
        print(f'Error reading file {file}')
        continue
    df = df.append( new_df )


df = create_general_statistics(df)