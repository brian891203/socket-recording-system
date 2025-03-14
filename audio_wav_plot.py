import numpy as np
import json
import math
import os
import shutil
from tqdm import tqdm
import argparse
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('-root-path', type=str, default='./', help='root folder path')
    parser.add_argument('-min-freq', type=int, default=15000, help='valid minimum frequency')
    parser.add_argument('-max-freq', type=int, default=25000, help='valid maximum frequency')
    args = parser.parse_args()

    return args

def wav2spec(wav_path):
    sample_rate, audio_data = wavfile.read(wav_path)
    frequencies, times, Sxx = spectrogram(audio_data, fs=sample_rate)  # default value: nperseg=256, noverlap=32(256/8)

    freq_start_index = np.where(frequencies >= args.min_freq)[0][0]    #18000 idx => 8, 22000 idx => 19
    freq_end_index = np.where(frequencies <= args.max_freq)[0][-1]
    selected_spectrogram = Sxx[freq_start_index:freq_end_index + 1, :]

    output = {
        'spec': np.array(selected_spectrogram),
        'time': times
    }

    return output

if __name__ == '__main__':
    args = get_args()
    root_path = "./"

    sample_paths = [os.path.join(root_path, folder) for folder in os.listdir(root_path)]
    sample_paths = [path for path in sample_paths if os.path.isdir(path)]

    for file_idx, sample_path in enumerate(tqdm(sample_paths)):
        wav_paths = [os.path.join(sample_path, file) for file in os.listdir(sample_path) if file.endswith('.wav')] # ex. ['./00001\\audio_1694414756_c1.wav', './00001\\audio_1694414756_c3.wav', './00001\\audio_1694414757_c2a1.wav', './00001\\audio_1694414757_c2a2.wav']

        for file_idx, wav_path in enumerate(wav_paths):
            wav_file_name = os.path.basename(wav_path).split('.')[0]

            spec = wav2spec(
                wav_path=wav_path,
            )

            npy_output_path = os.path.join(sample_path, f"{wav_file_name}.npy")

            np.save(npy_output_path, spec)

    #18000 idx => 8, 22000 idx => 19
    for file_idx, sample_path in enumerate(tqdm(sample_paths)):
        record_info_files = [file for file in os.listdir(sample_path) if file.endswith('c1.json')]
        record_info_files = os.path.join(sample_path,record_info_files[0])
        
        with open(record_info_files, 'r') as file:
            info = json.load(file)

        spec_npy_paths = [os.path.join(sample_path, file) for file in os.listdir(sample_path) if file.endswith('.npy')]

        filename = os.path.basename(spec_npy_paths[0]).split('_')[1]
        plt.figure(figsize=(20, 10))
        text = '\n'.join([f'{key}: {value}' for key, value in info.items()])
        plt.text(0.02, 1.05, f'Time Stamp: {filename}', transform=plt.gca().transAxes, fontsize=20, verticalalignment='top')
        plt.text(0.02, 0.9, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

        for spec_path in spec_npy_paths:
            data = np.load(spec_path, allow_pickle=True)
            data_dict = data.item()
            # filename = os.path.basename(spec_path).split('_')[1]

            if info["transmitter"] == [18000]:
                spec_data_18 = data_dict['spec'][8]
                spec_data_18 = np.array(spec_data_18)
                spec_data_18 = 20*(np.log10(spec_data_18))

                time = list(range(len(spec_data_18)))
                # plt.plot(time, spec_data_18)
                plt.plot(time, spec_data_18, label=f'client {os.path.basename(spec_path).split("_")[2].split(".")[0]}')
                plt.xlabel('Time')
                plt.ylabel('Values')
                plt.title('Spectrom Curve')
                # text = '\n'.join([f'{key}: {value}' for key, value in info.items()])
                # plt.text(0.02, 1.05, f'Time Stamp: {filename}', transform=plt.gca().transAxes, fontsize=20, verticalalignment='top')
                # plt.text(0.02, 0.9, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

                os.makedirs("./Spectrom Curve", exist_ok=True)
                fig_output_path = os.path.join("./Spectrom Curve", f"{filename}_18k_Spectrom Curve.png")

            elif info["transmitter"] == [22000]:
                spec_data_22 = data_dict['spec'][19]
                spec_data_22 = np.array(spec_data_22)
                spec_data_22 = 20*(np.log10(spec_data_22))

                time = list(range(len(spec_data_22)))
                # plt.plot(time, spec_data_18)
                plt.plot(time, spec_data_22, label=f'client {os.path.basename(spec_path).split("_")[2].split(".")[0]}')
                plt.xlabel('Time')
                plt.ylabel('Values')
                plt.title('Spectrom Curve')
                # text = '\n'.join([f'{key}: {value}' for key, value in info.items()])
                # plt.text(0.02, 1.05, f'Time Stamp: {filename}', transform=plt.gca().transAxes, fontsize=20, verticalalignment='top')
                # plt.text(0.02, 0.9, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

                os.makedirs("./Spectrom Curve", exist_ok=True)
                fig_output_path = os.path.join("./Spectrom Curve", f"{filename}_22k_Spectrom Curve.png")


            else:
                spec_data_18 = data_dict['spec'][8]
                spec_data_22 = data_dict['spec'][19]
                spec_data_18 = np.array(spec_data_18)
                spec_data_22 = np.array(spec_data_22)

                spec_data_18 = 20*(np.log10(spec_data_18))
                spec_data_22 = 20*(np.log10(spec_data_22))

                time = list(range(len(spec_data_18)))
                # plt.plot(time, spec_data_18)
                plt.plot(time, spec_data_18, label=f'client {os.path.basename(spec_path).split("_")[2].split(".")[0]}')
                plt.plot(time, spec_data_22, label=f'client {os.path.basename(spec_path).split("_")[2].split(".")[0]}')
                plt.xlabel('Time')
                plt.ylabel('Values')
                plt.title('Spectrom Curve')
                # text = '\n'.join([f'{key}: {value}' for key, value in info.items()])
                # plt.text(0.02, 1.05, f'Time Stamp: {filename}', transform=plt.gca().transAxes, fontsize=20, verticalalignment='top')
                # plt.text(0.02, 0.9, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

                os.makedirs("./Spectrom Curve", exist_ok=True)
                fig_output_path = os.path.join("./Spectrom Curve", f"{filename}_18k_Spectrom Curve.png")


        plt.legend()
        plt.savefig(fig_output_path)
        plt.close()

        

        