# Audio

The `audio` class provides access to the `Audio` object which contains methods for reading and writing audio files, extracting features, generating spectrograms, and more.

## Class: `Audio`

The `Audio` class is designed for handling audio files and provides functionalities such as loading, trimming, resampling, filtering, and visualization.

### Attributes:
- `filepath`: Path to the audio file.
- `filename`: Name of the audio file.
- `audio`: Audio data as a NumPy array.
- `sample_rate`: Sampling rate of the audio.
- `duration`: Duration of the audio in seconds.
- `length`: Length of the audio data in samples.
- `metadata`: Metadata of the audio file.
- `extension`: File extension of the audio file.
- `data`: DataFrame containing audio signal and time-related information.
- `start`: Start time of the audio.
- `end`: End time of the audio.

### Methods:
- `__init__`: Initialize the `Audio` object with a file or raw audio data.
- `add_data(filepath)`: Append audio data from another file.
- `trim(start, end=None, length=None, time_format="datetime", restart=False)`: Trim the audio to a specified time range or length.
- `resample(sample_rate)`: Resample the audio to a new sample rate.
- `spectrogram(...)`: Generate a spectrogram of the audio.
- `plot_spectrogram(...)`: Plot the spectrogram of the audio.
- `plot_melspectrogram(...)`: Plot the mel spectrogram of the audio.
- `plot_waveform(time_format="datetime")`: Plot the waveform of the audio.
- `plot_envelope(time_format="datetime")`: Plot the envelope of the audio signal.
- `psd(window_size=4096)`: Compute the power spectral density (PSD) of the audio.
- `plot_psd(window_size=4096)`: Plot the PSD of the audio.
- `lowpass_filter(cutoff, order=4, overwrite=False, type="sos")`: Apply a lowpass Butterworth filter.
- `highpass_filter(cutoff, order=4, overwrite=False, type="sos")`: Apply a highpass Butterworth filter.
- `bandpass_filter(lowcut, highcut, order=4, type="sos", overwrite=False)`: Apply a bandpass Butterworth filter.
- `envelope(overwrite=False)`: Compute the envelope of the audio signal.
- `write_audio(filepath)`: Write the audio data to a file.
- `fade_in(fade_time=0.1, window="hann", overwrite=False)`: Apply a fade-in effect to the audio.
- `fade_out(fade_time=0.1, window="hann", overwrite=False)`: Apply a fade-out effect to the audio.

---

## Usage Examples

### Loading an Audio File
```python
from sonicdb.audio import Audio

# Load an audio file
audio = Audio(filepath="example.wav")
print(f"Duration: {audio.duration} seconds")
```

### Trimming Audio
```python
# Trim the audio from 5 to 10 seconds
trimmed_audio = audio.trim(start=5, end=10, time_format="seconds")
print(f"Trimmed Duration: {trimmed_audio.duration} seconds")
```

### Resampling Audio
```python
# Resample the audio to 16 kHz
audio.resample(sample_rate=16000)
print(f"New Sample Rate: {audio.sample_rate}")
```

### Generating and Plotting a Spectrogram
```python
# Generate a spectrogram
time, frequency, Pxx = audio.spectrogram()

# Plot the spectrogram
audio.plot_spectrogram()
```

### Applying Filters
```python
# Apply a lowpass filter with a cutoff frequency of 1000 Hz
filtered_audio = audio.lowpass_filter(cutoff=1000, overwrite=False)
```

### Writing Audio to a File
```python
# Save the audio to a new file
audio.write_audio(filepath="output.wav")
```

### Combining Multiple Audio Files
```python
from sonicdb.audio import combine_audio

# Combine multiple audio files
combined_audio = combine_audio(["file1.wav", "file2.wav", "file3.wav"])
print(f"Combined Duration: {combined_audio.duration} seconds")
```

### Adding Effects
```python
from sonicdb.audio import echo

# Add an echo effect
echoed_audio = echo(audio.audio, sample_rate=audio.sample_rate, delay=0.2, decay=0.6)
```

---

## Functions:
- `combine_audio(list_of_files)`: Combine multiple audio files into one.
- `butter_lowpass(cutoff, fs, order, type="sos")`: Design a lowpass Butterworth filter.
- `butter_lowpass_filter(data, cutoff, fs, order=5, type="sos")`: Apply a lowpass Butterworth filter.
- `butter_highpass(cutoff, fs, order=5, type="sos")`: Design a highpass Butterworth filter.
- `butter_highpass_filter(data, cutoff, fs, order=5, type="sos")`: Apply a highpass Butterworth filter.
- `butter_bandpass(lowcut, highcut, fs, order=5, type="sos")`: Design a bandpass Butterworth filter.
- `butter_bandpass_filter(data, lowcut, highcut, fs, order=5, type="sos")`: Apply a bandpass Butterworth filter.
- `spectrogram(...)`: Generate a spectrogram of audio data.
- `write_audio(data, filepath, sample_rate)`: Write audio data to a file.
- `mp3_to_wav(input, output, output_format="wav")`: Convert an MP3 file to WAV format.
- `psd(x, sample_rate, window_size=4096, ...)`: Compute the power spectral density of a signal.
- `peak_hold(data, window=8192, sample_rate=24000)`: Compute the peak hold spectrum.
- `average_hold(data, window=1024, sample_rate=24000)`: Compute the average hold spectrum.
- `fade_in(data, sample_rate, fade_time=0.1, window="hann")`: Apply a fade-in effect to audio data.
- `fade_out(data, sample_rate, fade_time=0.1, window="hann")`: Apply a fade-out effect to audio data.
- `echo(data, sample_rate, delay=0.1, decay=0.5)`: Add an echo effect to audio data.
- `spectral_subtraction(signal, noise, scaling_factor=1.0)`: Perform spectral subtraction for noise reduction.
- `wavelet_denoise(signal, wavelet='db4', level=4)`: Denoise audio using wavelet decomposition.
- `autocorrelation(signal)`: Compute the autocorrelation of an audio signal.

