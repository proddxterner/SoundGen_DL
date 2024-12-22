import os
import pickle
from glob import glob
from tqdm.auto import tqdm

import numpy as np
import soundfile as sf

from soundgenerator import SoundGenerator
from autoencoder import VAE
from train import SPECTROGRAMS_PATH

HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "SoundGen_DL/VariatonalAutoencoder/samples/original"
SAVE_DIR_GENERATED = "SoundGen_DL/VariatonalAutoencoder/samples/generated"
MIN_MAX_VALUES_PATH = "SoundGen_DL/VariatonalAutoencoder/Dataset/Snares"

def load_dss(spectrograms_path):
    x_train = []
    file_paths = []
    print(os.getcwd())
    for file_path in tqdm(glob(spectrograms_path+"/*.wav.npy")):
        spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
        x_train.append(spectrogram)
        file_paths.append(file_path)
    x_train = np.array(x_train)
    print(x_train.shape)
    if (len(x_train) > 0):
        x_train = x_train[..., np.newaxis] # -> (10006/num_samples, 256, 64, 1)
    return x_train, file_paths


def select_spectrograms(spectrograms, file_paths, min_max_values, num_spectrograms = 2): # samples spectrograms from train set
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrograms = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in file_paths]

    print(file_paths)
    print(sampled_min_max_values)
    return sampled_spectrograms, sampled_min_max_values

def save_signals(signals, save_dir, sample_rate = 22050): # saving to disk
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)

if __name__ == "__main__":
    # initialise sound gen
    vae = VAE.load("model")
    sound_generator = SoundGenerator(vae, HOP_LENGTH)
    # load spec & min_max
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)

    specs, file_paths = load_dss(SPECTROGRAMS_PATH)
    # sample spec & min_max
    sampled_specs, sampled_min_max_values = select_spectrograms(specs, file_paths, min_max_values, 5)
    # generate audio for sampled spec
    signals, _ = sound_generator.generate(sampled_specs, sampled_min_max_values)
    # convert spec samples to audio
    original_signals = sound_generator.convert_spectrograms_to_audio(sampled_specs, sampled_min_max_values)
    # save audio signals
    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(original_signals, SAVE_DIR_ORIGINAL)