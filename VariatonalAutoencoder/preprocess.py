# 1 - load file
# 2 - pad signal (if necessary)
# 3 - extracting log spectrogram (librosa)
# 4 - normalize spectrogram
# 5 - save normalized spectrogram

#PreprocessingPipeline

import os
import pickle
import librosa
import numpy as np
from glob import glob
from tqdm.auto import tqdm

class Loader:
    #Loader responsible for loading  audio file

    def __init__(self, sample_rate, duration, mono): # for audio data receiving we use mono signal
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0]
        return signal


class Padder:
    #Padder  responsible to apply padding to array
    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items): # [1, 2, 3] --> 2 --> [0, 0, 1, 2, 3] "zero-padding"
        padded_array = np.pad(array,
                              (num_missing_items, 0),
                              mode=self.mode)
        return padded_array

    def right_pad(self, array, num_missing_items): # [1, 2, 3] --> 2 --> [1, 2, 3, 0, 0] "zero-padding"
        padded_array = np.pad(array,
                              (0, num_missing_items),
                              mode=self.mode)
        return padded_array


class LogSpectrogramExtractor:
    #LogSpectrogramExtractor extracts log spectrograms (in dB) from time-series signal
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal): # short-time fourier transform
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length)[:-1] # (1 + frame_size / 2, num_frames) 1024 --> 513 (not good - want even) --> 512
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram


class MinMaxNormaliser:
    #MinMaxNormaliser applies min max normalisation to array
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalise(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min()) # --> [0, 1]
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array


class Saver:
    #saver responsible to save features and  min max values.
    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)
        return save_path

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir,
                                 "min_max_values.pkl")
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path


class PreprocessingPipeline:

    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {} # dictionary
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_dir):
        for file_path in tqdm(glob(audio_files_dir+"/*")): # look through all files
            self._process_file(file_path)
            # print(f"Processed file {file_path}")
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        try:
            signal = self.loader.load(file_path)
            if self._is_padding_necessary(signal):
                signal = self._apply_padding(signal)
            feature = self.extractor.extract(signal)
            norm_feature = self.normaliser.normalise(feature)
            save_path = self.saver.save_feature(norm_feature, file_path)
            self._store_min_max_value(save_path, feature.min(), feature.max())
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
        

    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal

    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = { # create new key in dictionary equal to save_path - dict of dict
            "min": min_val,
            "max": max_val
        }

if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 345
    DURATION = 0.997 # in seconds
    SAMPLE_RATE = 22050 
    MONO = True

    SPECTROGRAMS_SAVE_DIR = "Dataset/Snares/Spectograms"
    MIN_MAX_VALUES_SAVE_DIR = "Dataset/Snares/"
    FILES_DIR = "Dataset/Snares/Audio"

    # instantiate all objects
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normaliser = MinMaxNormaliser(0, 1)
    saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normaliser = min_max_normaliser
    preprocessing_pipeline.saver = saver
    preprocessing_pipeline.process(FILES_DIR)
