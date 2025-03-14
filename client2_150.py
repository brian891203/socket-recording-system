import os
import time
import socket
import argparse
import threading
import numpy as np
import json

import cv2
import pyaudio
import wave
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


def get_args():
    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('-host', type=str, default='192.168.0.100', help='Server host')  #'192.168.0.100'
    parser.add_argument('-port', type=int, default=12345, help='Port for TCP')

    parser.add_argument('-video', action='store_true')
    parser.add_argument('-cam-id', type=int, default=0)
    parser.add_argument('-save-video', type=str, default='./video')

    parser.add_argument('-audio', action='store_true')
    parser.add_argument('-mic-name', type=str, default="麥克風 (Cotron EZM-001)")  # JLAB TALK GO MICROPHONE, Cotron EZM-001
    parser.add_argument('-save-audio', type=str, default='./audio_150')
    parser.add_argument('-amp-degree', type=int, default=15)
    parser.add_argument('-window-size', type=int, default=100)
    parser.add_argument('-hop-size', type=int, default=25)
    parser.add_argument('-start-video-display', action='store_true', help='Start video display thread')
    parser.add_argument('-start-audio-display', action='store_true', help='Start audio display thread')

    parser.add_argument('-sample-rate', type=int, default=96000)
    parser.add_argument('-d', type=int, default=300, help='Distance(cm)')
    parser.add_argument('-a', type=int, default=0, help='Angle(+45,0,-45)')
    parser.add_argument('-trans-freq', type=int, default=18000, nargs='+', help='Transmitter frequency(Hz)')
    parser.add_argument('-situation', type=str, default=" ", nargs='+', help='situation like movement or env settings etc.')
    parser.add_argument('-d-fixed', type=str, default=" ", nargs='+', help='user face to that mic')
    # parser.add_argument('-x', type=int, default=0)
    # parser.add_argument('-x', type=int, default=0)
    # parser.add_argument('-y', type=int, default=0)
    parser.add_argument('-z', type=int, default=150)
    parser.add_argument('-grid', type=int, nargs='+')


    # parser.add_argument('-audio2', action='store_true')

    args = parser.parse_args()

    return args


def list_avalible_cameras():
    num_cameras = 10
    camera_names = []

    for i in range(num_cameras):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                camera_names.append(f"Camera {i}")
            cap.release()
        else:
            break

    return camera_names


class WebcamClient:
    def __init__(self, host, port, amp_degree=15, window_size=100, hop_size=25):
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.isvis_video = threading.Event()
        self.isvis_video.clear()
        self.isvis_audio = threading.Event()
        self.isvis_audio.clear()

        self.isrecord_video = threading.Event()
        self.isrecord_video.clear()
        self.isrecord_audio = threading.Event()
        self.isrecord_audio.clear()

        self.amp_degree = amp_degree
        self.amp_maximum = int(2 ** amp_degree)
        self.window_size = window_size
        self.hop_size = hop_size

        self.start_client()

    def display_video(self):
        while self.isvis_video.is_set():  # check if isvis_video is True
            ret, frame = self.cap.read()
            if ret:
                cv2.circle(frame, (320, 240), 3, (0, 255, 0), 1)
                cv2.circle(frame, (260, 240), 3, (0, 255, 0), 1)
                cv2.circle(frame, (380, 240), 3, (0, 255, 0), 1)

                cv2.imshow('Webcam', frame)
                cv2.waitKey(1)
            else:
                print('Cannot display images from webcam')
                break

        print('Stop for displaying images, release caps')
        self.cap.release()
        cv2.destroyAllWindows()

    def display_audio(self):
        plt_size = int(self.window_size * self.default_sample_rate / 1000)
        chunk_size = int(self.hop_size * self.default_sample_rate / 1000)
        record_data = np.zeros(plt_size) + 1e-14

        plt.ion()
        self.fig = plt.figure()

        ax_spectrum = self.fig.add_subplot(111)

        spectrogram_image = ax_spectrum.imshow(
            np.zeros((0, 0)),
            aspect='auto', cmap='viridis',
            extent=[0, plt_size, 0, self.default_sample_rate / 2],
        )

        ax_spectrum.set_xlabel('Time (milliseconds)')
        ax_spectrum.set_ylabel('Frequency (Hz)')
        ax_spectrum.set_title('Spectrum')
        ax_spectrum.text(
            0.95, 0.95,
            f'Mic: {args.mic_name}\nTransmitter freq: {args.trans_freq} Hz\nSample Rate: {args.sample_rate}Hz\nDistance: {args.d} cm\nAngle: {args.a}°',
            verticalalignment='top', horizontalalignment='right',
            transform=ax_spectrum.transAxes,
            color='white', fontsize=10, bbox=dict(facecolor='none', alpha=0.0)
        )

        self.fig.subplots_adjust(hspace=0.5)

        colorbar = plt.colorbar(spectrogram_image, ax=ax_spectrum)
        colorbar.set_label('Magnitude (dB)')

        plt.show(block=False)

        while self.isvis_audio.is_set():
            stream_data = self.stream.read(chunk_size, exception_on_overflow=False)
            stream_array = np.frombuffer(stream_data, dtype=np.int16).copy()
            # stream_array = stream_array / self.amp_maximum  # amplitude_max: 2^15
            record_data = np.append(record_data, stream_array)
            record_data = record_data[chunk_size:chunk_size + plt_size]

            f, t, Sxx = spectrogram(record_data, fs=self.default_sample_rate)
            target_freq = 10000  # 10 kHz
            index_cutoff = np.argmax(f >= target_freq)
            filtered_Sxx = Sxx[index_cutoff:, :]
            filtered_f = f[index_cutoff:]

            db_values = 20 * np.log10(np.abs(filtered_Sxx[::-1]))
            vmin = np.min(db_values)
            vmax = np.max(db_values)

            spectrogram_image.set_data(db_values)
            spectrogram_image.set_extent([0, plt_size, filtered_f[0], self.default_sample_rate / 2])
            spectrogram_image.set_clim(vmin, vmax)

            # print("vmax : ", vmax)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        plt.close('all')
        cv2.destroyAllWindows()

    def displays(self):
        if args.video:
            self.cap = cv2.VideoCapture(args.cam_id, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                print("Device {} is not opened".format(args.cam_id))
                exception_command = 'Fail to open cam {}'.format(args.cam_id)
                self.client_socket.sendall(exception_command.encode())
                return False

            self.video_display = threading.Thread(target=self.display_video)
            if args.start_video_display:
                self.video_display.daemon = True
                self.isvis_video.set()  # set up isvis_video as True
                self.video_display.start()

            # self.isvis_video.set()  # set up isvis_video as True
            # self.video_display.start()

        if args.audio:
            self.audio = pyaudio.PyAudio()

            for idx in range(self.audio.get_device_count()):
                if args.mic_name == self.audio.get_device_info_by_index(idx).get('name'):
                    print('Device found: {}'.format(self.audio.get_device_info_by_index(idx).get('name')))
                    MIC_INDEX = idx
                    break
            else:
                print("No device found: {}".format(args.mic_name))
                exception_command = 'Fail to open mic {}'.format(args.mic_name)
                self.client_socket.sendall(exception_command.encode())
                return False

            # self.default_sample_rate = float(self.audio.get_device_info_by_index(MIC_INDEX).get('defaultSampleRate'))
            self.default_sample_rate = args.sample_rate
            assert self.default_sample_rate != 0
            self.plt_size = int(self.window_size * self.default_sample_rate / 1000)
            self.chunk_size = int(self.hop_size * self.default_sample_rate / 1000)

            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                input_device_index=MIC_INDEX,
                rate=int(self.default_sample_rate),
                input=True,
                frames_per_buffer=self.chunk_size
            )

            self.audio_display = threading.Thread(target=self.display_audio)
            if args.start_audio_display:
                self.audio_display.daemon = True
                self.isvis_audio.set()
                self.audio_display.start()
            # self.isvis_audio.set()
            # self.audio_display.start()

        self.client_socket.sendall("start".encode())

        return True

    def displays_close(self):
        if args.video:
            self.isvis_video.clear()
            # self.video_display.join()

        if args.audio:
            self.isvis_audio.clear()
            self.stream.stop_stream()
            self.audio.terminate()
            # self.audio_display.join()

        self.client_socket.sendall("end".encode())

        return False

    def record_video(self):
        root_folder = args.save_video

        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        save_path = os.path.join(root_folder, 'video_{}_c2.mp4'.format(int(time.time())))
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

        frame_count = 0
        while self.isrecord_video.is_set():
            ret, frame = self.cap.read()
            if ret:
                video_writer.write(frame)
                frame_count += 1
                print(f"Frame {frame_count} saved.")

        video_writer.release()

    def record_audio(self):
        root_folder = args.save_audio

        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        self.time = int(time.time())
        save_path = os.path.join(root_folder, 'audio_{}_c2a2.wav'.format(self.time))

        self.audio_frames_recorded = []
        while self.isrecord_audio.is_set():
            frame = self.stream.read(self.chunk_size)
            self.audio_frames_recorded.append(frame)
            print(f"Audio Length: {len(self.audio_frames_recorded)}")
            
            stream_array = np.frombuffer(frame, dtype=np.int16).copy()            
            _, _, Sxx = spectrogram(stream_array, fs=self.default_sample_rate)
            Sxx_mean = np.mean(Sxx, axis=1)
            # print(f"Sxx: {Sxx_mean}")
            if Sxx_mean.max() < 1e-6:
                self.client_socket.sendall("warning".encode())

        wavefile = wave.open(save_path, 'wb')
        wavefile.setnchannels(1)
        wavefile.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wavefile.setframerate(self.default_sample_rate)
        wavefile.writeframes(b''.join(self.audio_frames_recorded))
        wavefile.close()

    def records(self):
        if args.video:
            self.video_record = threading.Thread(target=self.record_video)
            self.video_record.daemon = True
            self.isrecord_video.set()
            self.video_record.start()

        if args.audio:
            self.audio_record = threading.Thread(target=self.record_audio)
            self.audio_record.daemon = True
            self.isrecord_audio.set()
            self.audio_record.start()

        self.client_socket.sendall("film".encode())

        return True

    def records_stop(self):
        if args.video:
            self.isrecord_video.clear()
            self.video_record.join()

        if args.audio:
            self.isrecord_audio.clear()  # --> event clear means event unsetting
            self.audio_record.join()

        self.client_socket.sendall("stop".encode())

        return False

    def start_client(self):
        self.client_socket.connect((self.host, self.port))
        self.counters = 0
        # self.direction_fixed = False
        self.transmitter_counter = 0

        data = self.client_socket.recv(1024).decode()
        if data == "ready":
            self.client_socket.sendall("ready".encode())
            print("Waiting for server command...")

        while True:
            data = self.client_socket.recv(1024).decode()
            if data == "start":
                print("Start the webcam view...")
                self.isstart = self.displays()
                if self.isstart:
                    print('Start sucessfully.')
                else:
                    print('Start unsucessfully, close the connection.')
                    self.client_socket.close()
            elif data == 'end':
                print("Close the webcam view...")
                self.isstart = self.displays_close()
                if not self.isstart:
                    print('End successfully.')
            elif data == 'film':
                print("Starts to record...")
                self.isrecord = self.records()
            elif data == 'stop':
                print('Stop for recording...')
                self.isrecord = self.records_stop()
                print('Save Experimental Data as json file...')
                
                self.save_experiment_data()  
                self.counters += 1

                self.client_socket.sendall(f"show;{args.trans_freq};{self.counters}".encode())

                # print('Save Experimental Data as Amp-Time Plot...')
                # self.amp_time_plot()
                if not self.isrecord:
                    print('End of recording.')
            elif data == 'exit':
                print('Exit for all operation...')
                break

        self.client_socket.sendall("exit".encode())
        self.client_socket.close()

        print('Client has closed.')
        exit()

    def save_experiment_data(self):
        # transmitter_list = [[18000], [22000], [18000, 22000]]
        for idx, frame in enumerate(self.audio_frames_recorded):
            self.audio_frames_recorded[idx] = np.frombuffer(frame, dtype=np.int16)
        self.audio_frames_recorded = np.concatenate(self.audio_frames_recorded, -1)

        # f, t, spec = spectrogram(self.audio_frames_recorded, fs=self.default_sample_rate)

        experiment_data = {
            "mic_name": "Cotron EZM-001",
            "mic_height": 150,
            "transmitter": args.trans_freq,
            "direction_fixed": False
        }

        root_folder = args.save_audio
        json_path = os.path.join(root_folder, 'data_{}_c2a2.json'.format(self.time))
        with open(json_path, 'w') as json_file:
            json.dump(experiment_data, json_file, indent=4)

    def amp_time_plot(self):
        freqs, times, spec = spectrogram(self.audio_frames_recorded, fs=self.default_sample_rate)

        for idx, f in enumerate(freqs):
            if f != 18000:
                continue

            amp_freq = spec[idx]
            amp_freq = 20 * np.log10(amp_freq)
            plt.plot(np.arange(spec.shape[-1]), amp_freq, label=f'Avg: {np.median(amp_freq)}')

        plt.xlabel('Frame')
        plt.ylabel('Amplitude(dB)')
        plt.title('Amplitude(dB) of Frequencies Over Time')
        plt.legend(loc='upper left')
        plt.grid()
        output_file = os.path.join(args.save_audio, 'plt_{}.png'.format(self.time))
        plt.savefig(output_file)
        plt.close()


if __name__ == "__main__":
    args = get_args()

    if not args.video and not args.audio:
        raise ValueError('Nothing is operating for video or audio.')

    # print(list_avalible_cameras())

    client = WebcamClient(
        host=args.host,
        port=args.port,
        amp_degree=args.amp_degree,
        window_size=args.window_size,
        hop_size=args.hop_size
    )
