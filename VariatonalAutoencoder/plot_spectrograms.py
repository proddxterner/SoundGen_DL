import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

def plot_spectrogram(file_path, title, ax):
    """
    Load a .wav file, generate a spectrogram, normalize it, and plot.
    """
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Generate the spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128)

    # Convert to log scale (dB)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Normalize to range [0, 1]
    normalized_spectrogram = (log_spectrogram - log_spectrogram.min()) / (log_spectrogram.max() - log_spectrogram.min())

    # Plot the spectrogram
    librosa.display.specshow(normalized_spectrogram, sr=sr, hop_length=512, x_axis="time", y_axis="mel", cmap="viridis", ax=ax)
    ax.set_title(title)
    ax.axis("off")

# Directories containing original and generated files
original_dir = "samples/original"
generated_dir = "samples/generated"

# Get file lists (assuming both directories have the same number of files)
original_files = sorted([os.path.join(original_dir, f) for f in os.listdir(original_dir) if f.endswith('.wav')])[:5]
generated_files = sorted([os.path.join(generated_dir, f) for f in os.listdir(generated_dir) if f.endswith('.wav')])[:5]

# Create a plot with 5 rows and 2 columns (side-by-side comparison)
fig, axes = plt.subplots(len(original_files), 2, figsize=(10, 15))

for i, (orig_file, gen_file) in enumerate(zip(original_files, generated_files)):
    # Plot the original spectrogram
    plot_spectrogram(orig_file, f"Original {os.path.basename(orig_file)}", axes[i, 0])
    
    # Plot the generated spectrogram
    plot_spectrogram(gen_file, f"Generated {os.path.basename(gen_file)}", axes[i, 1])

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
