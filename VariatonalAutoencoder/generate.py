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
SAVE_DIR_ORIGINAL = "samples/original"
SAVE_DIR_GENERATED = "samples/generated"
MIN_MAX_VALUES_PATH = "Dataset/Snares"

def load_dss(spectrograms_path):
    x_train = []
    file_paths = []
    print("Current Directory:", os.getcwd())
    for file_path in tqdm(glob(spectrograms_path + "/*.wav.npy")):
        spectrogram = np.load(file_path)  # (n_bins, n_frames, 1)
        x_train.append(spectrogram)
        file_paths.append(file_path)
    x_train = np.array(x_train)
    print("Loaded spectrograms shape:", x_train.shape)
    if len(x_train) > 0:
        x_train = x_train[..., np.newaxis]  # -> (num_samples, 256, 64, 1)
    return x_train, file_paths

def select_spectrograms(spectrograms, file_paths, min_max_values, num_spectrograms=2):
    assert len(spectrograms) >= num_spectrograms, "Not enough spectrograms to sample."
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms, replace=False)
    sampled_spectrograms = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in file_paths]

    print("Sampled file paths:", file_paths)
    print("Sampled min-max values:", sampled_min_max_values)
    return sampled_spectrograms, sampled_min_max_values

def save_signals(signals, save_dir, sample_rate=22050):
    os.makedirs(save_dir, exist_ok=True)
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, f"{i}.wav")
        sf.write(save_path, signal, sample_rate)

if __name__ == "__main__":
    # Check required paths
    assert os.path.exists(MIN_MAX_VALUES_PATH + '/min_max_values.pkl'), "Min-max values file not found!"
    assert os.path.exists(SPECTROGRAMS_PATH), "Spectrograms path not found!"
    
    # Initialize sound generator
    vae = VAE.load("model")
    sound_generator = SoundGenerator(vae, HOP_LENGTH)

    # Load spectrograms and min-max values
    with open(MIN_MAX_VALUES_PATH + '/min_max_values.pkl', "rb") as f:
        min_max_values = pickle.load(f)

    specs, file_paths = load_dss(SPECTROGRAMS_PATH)

    # Sample spectrograms and generate audio
    sampled_specs, sampled_min_max_values = select_spectrograms(specs, file_paths, min_max_values, num_spectrograms=5)
    signals, _ = sound_generator.generate(sampled_specs, sampled_min_max_values)

    # Convert and save signals
    original_signals = sound_generator.convert_spectrograms_to_audio(sampled_specs, sampled_min_max_values)
    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(original_signals, SAVE_DIR_ORIGINAL)
