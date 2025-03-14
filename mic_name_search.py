import pyaudio
audio = pyaudio.PyAudio()
if __name__ == '__main__':
    for idx in range(audio.get_device_count()):
        print('Device found: {}'.format(audio.get_device_info_by_index(idx).get('name')))
        MIC_INDEX = idx

# "Microsoft Sound Mapper - Input"