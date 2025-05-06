# # type: ignore
import os
from copy import deepcopy
from datetime import datetime
from datetime import timedelta

import librosa
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from pydub import AudioSegment
from scipy import signal

from sonicdb import utilities
import pywt


np.seterr(divide="ignore")
valid_audio = ["wav", "flac", "mp3", "ogg", "aiff", "au"]


class Audio:  # pragma: no cover
    """
    A class for handling audio files, providing functionalities such as loading, trimming, resampling, filtering, and visualization.

    Attributes:
        filepath (str): Path to the audio file.
        filename (str): Name of the audio file.
        audio (np.ndarray): Audio data as a NumPy array.
        sample_rate (int): Sampling rate of the audio.
        duration (float): Duration of the audio in seconds.
        length (int): Length of the audio data in samples.
        metadata (dict): Metadata of the audio file.
        extension (str): File extension of the audio file.
        data (pd.DataFrame): DataFrame containing audio signal and time-related information.
        start (datetime): Start time of the audio.
        end (datetime): End time of the audio.
    """

    def __init__(self, filepath=None, audio=None, sample_rate=None, start=None):
        if filepath is not None:
            self.filepath = filepath
            self.filename = os.path.basename(filepath)
            self.audio, self.sample_rate = librosa.load(filepath)
            self.duration = librosa.get_duration(path=self.filepath)
            self.length = len(self.audio)

            self.metadata = utilities.metadata(self.filepath)
            self.extension = self.metadata["extension"]

        if audio is not None:
            self.audio = audio
            self.sample_rate = sample_rate
            self.duration = len(self.audio) / self.sample_rate
            self.length = len(self.audio)

        self.data = pd.DataFrame()

        if isinstance(start, datetime):
            self.start = start
        else:
            if start is None:
                try:
                    self.start = utilities.read_datetime(self.filename[:23])
                except Exception:
                    self.start = self.metadata["created"]
                    # self.start = utilities.read_datetime("00:00:00")
                # self.start = utilities.read_datetime(start)
            else:
                self.start = utilities.read_datetime(start)

        self.end = self.start + timedelta(seconds=len(self.audio) / self.sample_rate)

        self.data["signal"] = self.audio
        self.data["seconds"] = self.data.index / self.sample_rate
        self.data["ms"] = self.data["seconds"] * 1000
        self.data["datetime"] = pd.date_range(
            start=self.start,
            end=(self.start + timedelta(seconds=self.data.seconds.max())),
            periods=len(self.audio),
        )
        # rearrange columns
        self.data = self.data[["datetime", "seconds", "ms", "signal"]]

    def add_data(self, filepath):
        """
        Append audio data from another file to the current audio object.

        Args:
            filepath (str): Path to the audio file to append.
        """

        audio = Audio(filepath)

        # TODO Check sample rate of new file and convert if necessary to match
        self.audio = np.append(self.audio, audio.audio)

        self.end = self.start + timedelta(seconds=len(self.audio) / self.sample_rate)

        self.data = pd.DataFrame()
        self.data["datetime"] = pd.date_range(
            start=self.start, end=self.end, periods=len(self.audio)
        )
        self.data["signal"] = self.audio

        if isinstance(self.metadata, dict):
            self.metadata = [self.metadata]
            self.metadata.np.append(audio.metadata)

        self.metadata.np.append(audio.metadata)

    def trim(self, start, end=None, length=None, time_format="datetime", restart=False):
        """
        Trim the audio to a specified start and end time or length.

        Args:
            start (datetime | str | float | int): Start time of the trim.
            end (datetime | str | float | int, optional): End time of the trim. Defaults to None.
            length (float, optional): Length of the trimmed audio in seconds, milliseconds, or samples. Defaults to None.
            time_format (str, optional): Format of the time ('datetime', 'samples', 'seconds', 'ms'). Defaults to "datetime".
            restart (bool, optional): Whether to reset the time indices. Defaults to False.

        Returns:
            Audio: A new Audio object with the trimmed data.
        """
        sample = deepcopy(self)

        if time_format == "datetime":
            if not isinstance(start, datetime):
                try:
                    start = utilities.read_datetime(start)
                except ValueError:
                    start = start

            if end is None:
                if isinstance(start, datetime):
                    end = start + timedelta(seconds=length)
            else:
                if not isinstance(end, datetime):
                    end = utilities.read_datetime(end)

            if length is None:
                length = (end - start).total_seconds()

            sample.data = sample.data.loc[
                (sample.data.datetime >= start) & (sample.data.datetime <= end)
            ]

        if time_format == "samples":
            if end is None:
                end = start + length

            sample = deepcopy(self)
            sample.data = sample.data.loc[start:end]

        if time_format == "seconds":
            if end is None:
                end = start + length

            sample.data = sample.data.loc[
                (sample.data["seconds"] >= start) & (sample.data["seconds"] <= end)
            ]

        if time_format == "ms":
            if end is None:
                end = start + length

            sample.data = sample.data.loc[
                (sample.data["ms"] >= start) & (sample.data["ms"] <= end)
            ]

        if restart:
            sample.data = sample.data.reset_index(drop=True)
            sample.data["seconds"] = sample.data.index / sample.sample_rate
            sample.data["ms"] = sample.data["seconds"] * 1000

        sample.start = sample.data.datetime.iloc[0]
        sample.end = sample.data.datetime.iloc[-1]
        sample.audio = sample.data.signal.values
        sample.length = len(sample.audio)
        sample.duration = len(sample.audio) / sample.sample_rate

        return sample

    def resample(self, sample_rate: int) -> None:
        """
        Resample the audio to a new sample rate.

        Args:
            sample_rate (int): Target sample rate.
        """

        try:
            self.audio = librosa.resample(
                self.audio, orig_sr=self.sample_rate, target_sr=sample_rate
            )
        except Exception as e:
            print(f"Error: {e}")
            self.audio = [0] * int(self.duration) * int(sample_rate)
            print(f"An error occurred while resampling audio: {e}")

        self.sample_rate = sample_rate
        self.data = pd.DataFrame()
        self.end = self.start + timedelta(seconds=len(self.audio) / self.sample_rate)

        self.data["datetime"] = pd.date_range(
            start=self.start, end=self.end, periods=len(self.audio)
        )
        self.data["seconds"] = self.data.index / self.sample_rate
        self.data["ms"] = self.data["seconds"] * 1000
        self.data["signal"] = self.audio

    def spectrogram(self, window="hann", window_size=8192, nfft=8192, noverlap=4096, nperseg=8192, time_format="datetime") -> tuple:
        """
        Generate a spectrogram of the audio.

        Args:
            window (str, optional): Window function to use. Defaults to "hann".
            window_size (int, optional): Size of the window in samples. Defaults to 8192.
            nfft (int, optional): Number of FFT points. Defaults to 8192.
            noverlap (int, optional): Number of overlapping samples. Defaults to 4096.
            nperseg (int, optional): Number of samples per segment. Defaults to 8192.
            time_format (str, optional): Format of the time axis ('datetime', 'samples', 'seconds', 'ms'). Defaults to "datetime".

        Returns:
            tuple: Time, frequency, and power spectral density (Pxx).
        """

        time, frequency, Pxx = spectrogram(
            self.data.signal,
            self.sample_rate,
            window=window,
            window_size=window_size,
            nfft=nfft,
            noverlap=noverlap,
            nperseg=nperseg,
            start=self.start,
            end=self.end,
            time_format=time_format,
        )

        return time, frequency, Pxx

    def plot_spectrogram(self, window="hann", window_size: int = 8192, nfft: int = 8192, noverlap: int = 4096, nperseg: int = 8192, zmin: int = None, zmax: int = None, gain: int = 0, showscale: bool = False, cmap="jet", aspect="auto", time_format="datetime", fig=None, ax=None):
        """
        Plot the spectrogram of the audio.

        Args:
            window (str, optional): Window function to use. Defaults to "hann".
            window_size (int, optional): Size of the window in samples. Defaults to 8192.
            nfft (int, optional): Number of FFT points. Defaults to 8192.
            noverlap (int, optional): Number of overlapping samples. Defaults to 4096.
            nperseg (int, optional): Number of samples per segment. Defaults to 8192.
            zmin (int, optional): Minimum z-value for color scale. Defaults to None.
            zmax (int, optional): Maximum z-value for color scale. Defaults to None.
            gain (int, optional): Gain to apply to the spectrogram. Defaults to 0.
            showscale (bool, optional): Whether to show color scale. Defaults to False.
            cmap (str, optional): Colormap to use. Defaults to "jet".
            aspect (str, optional): Aspect ratio. Defaults to "auto".
            time_format (str, optional): Format of the time axis ('datetime', 'samples', 'seconds', 'ms'). Defaults to "datetime".
            fig (matplotlib.figure.Figure, optional): Figure object to plot on. Defaults to None.
            ax (matplotlib.axes.Axes, optional): Axis object to plot on. Defaults to None.

        Returns:
            tuple: Figure and axis objects if a figure is created.
        """
        if ax is None:
            fig, ax = plt.subplots()

        time, frequency, Pxx = self.spectrogram(
            window=window,
            window_size=window_size,
            nfft=nfft,
            noverlap=noverlap,
            nperseg=nperseg,
        )

        Pxx = 10 * np.log10(Pxx) + gain

        if zmin is None:
            zmin = Pxx.min()
        if zmax is None:
            zmax = Pxx.max()

        if time_format == "seconds":
            extents = [
                self.data["seconds"].min(),
                self.data["seconds"].max(),
                frequency.min(),
                frequency.max(),
            ]
        elif time_format == "ms":
            extents = [
                self.data["ms"].min(),
                self.data["ms"].max(),
                frequency.min(),
                frequency.max(),
            ]
        elif time_format == "samples":
            extents = [0, len(self.data), frequency.min(), frequency.max()]
        else:
            extents = [self.start, self.end, frequency.min(), frequency.max()]

        axi = ax.imshow(
            Pxx,
            cmap=cmap,
            aspect=aspect,
            extent=extents,
            origin="lower",
        )
        axi.set_clim([zmin, zmax])

        ax.set_ylabel("Frequency [Hz]")

        if time_format == "seconds":
            ax.set_xlabel("Time [s]")
            ax.set_xlim(self.data["seconds"].min(), round(self.data["seconds"].max()))
        elif time_format == "ms":
            ax.set_xlabel("Time [ms]")
            ax.set_xlim(self.data["ms"].min(), round(self.data["ms"].max()))
        elif time_format == "samples":
            ax.set_xlabel("Samples")
            ax.set_xlim(0, len(self.data))
        else:
            ax.set_xlim([self.data.datetime.iloc[0], self.data.datetime.iloc[-1]])
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            # ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))

        if showscale == "right":
            cbar = fig.colorbar(
                axi,
                location="right",
                orientation="vertical",
                ticks=[zmin, round(zmin + (zmax - zmin) / 2), zmax],
            )
            cbar.ax.set_ylabel("Power [dB]")
        elif showscale == "top":
            cbar = fig.colorbar(
                axi,
                location="top",
                orientation="horizontal",
                ticks=[zmin, round(zmin + (zmax - zmin) / 2), zmax],
                pad=0.1,
            )
            # cbar.ax.set_ylabel("Power [dB]", rotation="horizontal")
            # set the label to be on the top of the colorbar
            cbar.ax.xaxis.set_label_position("top")
            cbar.ax.set_xlabel("Power [dB]")

        if fig:
            return fig, ax

    def plot_melspectrogram(self, window="hann", nmels: int = 8192, window_size: int = 8192, nfft: int = 8192, noverlap: int = 4096, nperseg: int = 8192, zmin: int = None, zmax: int = None, gain: int = 0, showscale: bool = False, cmap="jet", aspect="auto", time_format="datetime", ax=None, fmin=0, fmax=None):
        """
        Plot the mel spectrogram of the audio.

        Args:
            window (str, optional): Window function to use. Defaults to "hann".
            nmels (int, optional): Number of mel bands to generate. Defaults to 8192.
            window_size (int, optional): Size of the window in samples. Defaults to 8192.
            nfft (int, optional): Number of FFT points. Defaults to 8192.
            noverlap (int, optional): Number of overlapping samples. Defaults to 4096.
            nperseg (int, optional): Number of samples per segment. Defaults to 8192.
            zmin (int, optional): Minimum z-value for color scale. Defaults to None.
            zmax (int, optional): Maximum z-value for color scale. Defaults to None.
            gain (int, optional): Gain to apply to the spectrogram. Defaults to 0.
            showscale (bool, optional): Whether to show color scale. Defaults to False.
            cmap (str, optional): Colormap to use. Defaults to "jet".
            aspect (str, optional): Aspect ratio. Defaults to "auto".
            time_format (str, optional): Format of the time axis ('datetime', 'samples', 'seconds', 'ms'). Defaults to "datetime".
            ax (matplotlib.axes.Axes, optional): Axis object to plot on. Defaults to None.
            fmin (int, optional): Minimum frequency. Defaults to 0.
            fmax (int, optional): Maximum frequency. Defaults to None.

        Returns:
            tuple: Figure and axis objects if a figure is created.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        if fmax is None:
            fmax = self.sample_rate / 2

        Pxx = librosa.feature.melspectrogram(
            y=self.audio,
            sr=self.sample_rate,
            n_fft=nfft,
            hop_length=noverlap,
            n_mels=128,
            fmin=fmin,
            fmax=fmax,
        )

        Pxx = 10 * np.log10(Pxx) + gain

        if zmin is None:
            zmin = Pxx.min()
        if zmax is None:
            zmax = Pxx.max()

        if time_format == "seconds":
            extents = [
                self.data["seconds"].min(),
                self.data["seconds"].max(),
                fmin,
                fmax,
            ]
        elif time_format == "ms":
            extents = [
                self.data["ms"].min(),
                self.data["ms"].max(),
                fmin(),
                fmax(),
            ]
        elif time_format == "samples":
            extents = [0, len(self.data), fmin, fmax]
        else:
            extents = [self.start, self.end, fmin, fmax]

        axi = ax.imshow(
            Pxx,
            cmap=cmap,
            aspect=aspect,
            extent=extents,
            origin="lower",
        )
        axi.set_clim([zmin, zmax])

        ax.set_ylabel("Frequency [Hz]")

        if time_format == "seconds":
            ax.set_xlabel("Time [s]")
            ax.set_xlim(self.data["seconds"].min(), round(self.data["seconds"].max()))
        elif time_format == "ms":
            ax.set_xlabel("Time [ms]")
            ax.set_xlim(self.data["ms"].min(), round(self.data["ms"].max()))
        elif time_format == "samples":
            ax.set_xlabel("Samples")
            ax.set_xlim(0, len(self.data))
        else:
            ax.set_xlim([self.data.datetime.iloc[0], self.data.datetime.iloc[-1]])
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            # ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))

        if showscale == "right":
            cbar = fig.colorbar(
                axi, location="right", orientation="vertical", ticks=[zmin, zmax]
            )
            cbar.ax.set_ylabel("Power [dB]")
        elif showscale == "top":
            cbar = fig.colorbar(
                axi, location="top", orientation="horizontal", ticks=[zmin, zmax]
            )
            cbar.ax.set_title("Power [dB]")

        if fig:
            return fig, ax

    def plot_waveform(self, time_format: str = "datetime", ax=None):
        """
        Plot the waveform of the audio.

        Args:
            time_format (str, optional): Format of the time axis ('datetime', 'samples', 'seconds', 'ms'). Defaults to "datetime".
            ax (matplotlib.axes.Axes, optional): Axis object to plot on. Defaults to None.

        Returns:
            tuple: Figure and axis objects if a figure is created.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        if time_format == "datetime":
            ax.plot(self.data.datetime, self.data.signal)
            ax.set_xlim(self.start, self.end)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))

        if time_format == "seconds":
            ax.plot(self.data["seconds"], self.data.signal)
            ax.set_xlabel("Time [s]")
            ax.set_xlim(self.data["seconds"].min(), self.data["seconds"].max())

        if time_format == "ms":
            ax.plot(self.data["ms"], self.data.signal)
            ax.set_xlabel("Time [ms]")
            ax.set_xlim(self.data["ms"].min(), self.data["ms"].max())

        if time_format == "samples":
            ax.plot(self.data.index, self.data.signal)
            ax.set_xlabel("Samples")
            ax.set_xlim(0, len(self.data))

        ax.set_ylabel("Amplitude")

        if fig:
            return fig, ax

    def plot_envelope(self, time_format: str = "datetime"):
        """
        Plot the envelope of the audio signal.

        Args:
            time_format (str, optional): Format of the time axis ('datetime', 'samples', 'seconds', 'ms'). Defaults to "datetime".

        Returns:
            tuple: Figure and axis objects.
        """
        fig, ax = plt.subplots()

        if time_format == "datetime":
            ax.plot(self.data.datetime, self.envelope())
            ax.set_xlim(self.start, self.end)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))

        if time_format == "seconds":
            ax.plot(self.data["seconds"], self.envelope())
            ax.set_xlabel("Time [s]")
            ax.set_xlim(self.data["seconds"].min(), self.data["seconds"].max())

        if time_format == "ms":
            ax.plot(self.data["ms"], self.envelope())
            ax.set_xlabel("Time [ms]")
            ax.set_xlim(self.data["ms"].min(), self.data["ms"].max())

        if time_format == "samples":
            ax.plot(self.data.index, self.envelope())
            ax.set_xlabel("Samples")
            ax.set_xlim(0, len(self.data))

        ax.set_ylabel("Amplitude")

        return fig, ax

    def psd(self, window_size: int = 4096) -> tuple:
        """
        Compute the power spectral density (PSD) of the audio.

        Args:
            window_size (int, optional): Size of the window in samples. Defaults to 4096.

        Returns:
            tuple: Frequency and power values.
        """
        frequency, power = psd(
            self.data.signal, self.sample_rate, window_size=window_size
        )

        return frequency, power

    def plot_psd(self, window_size: int = 4096) -> tuple:
        """
        Plot the power spectral density (PSD) of the audio.

        Args:
            window_size (int, optional): Size of the window in samples. Defaults to 4096.

        Returns:
            tuple: Figure and axis objects.
        """
        frequency, power = self.psd(window_size=window_size)

        fig, ax = plt.subplots()
        ax.plot(frequency, power)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power [dB]")

        return fig, ax

    def lowpass_filter(self, cutoff, order=4, overwrite=False, type="sos"):
        """
        Apply a lowpass Butterworth filter to the audio.

        Args:
            cutoff (int): Cutoff frequency in Hz.
            order (int, optional): Filter order. Defaults to 4.
            overwrite (bool, optional): Whether to overwrite the current audio. Defaults to False.
            type (str, optional): Filter type ('sos' or 'ab'). Defaults to "sos".

        Returns:
            list | None: Filtered audio if overwrite is False.
        """

        audio = butter_lowpass_filter(
            self.data.signal, cutoff, self.sample_rate, order, type=type
        )

        if overwrite is True:
            self.data.signal = audio
            self.audio = audio
        else:
            return list(audio)

    def highpass_filter(self, cutoff, order=4, overwrite=False, type="sos"):
        """
        Apply a highpass Butterworth filter to the audio.

        Args:
            cutoff (int): Cutoff frequency in Hz.
            order (int, optional): Filter order. Defaults to 4.
            overwrite (bool, optional): Whether to overwrite the current audio. Defaults to False.
            type (str, optional): Filter type ('sos' or 'ab'). Defaults to "sos".

        Returns:
            list | None: Filtered audio if overwrite is False.
        """

        audio = butter_highpass_filter(
            self.data.signal, cutoff, self.sample_rate, order, type=type
        )

        if overwrite is True:
            self.data.signal = audio
            self.audio = audio
        else:
            return list(audio)

    def bandpass_filter(self, lowcut, highcut, order=4, type="sos", overwrite=False):
        """
        Apply a bandpass Butterworth filter to the audio.

        Args:
            lowcut (int): Low cutoff frequency in Hz.
            highcut (int): High cutoff frequency in Hz.
            order (int, optional): Filter order. Defaults to 4.
            type (str, optional): Filter type ('sos' or 'ab'). Defaults to "sos".
            overwrite (bool, optional): Whether to overwrite the current audio. Defaults to False.

        Returns:
            list | None: Filtered audio if overwrite is False.
        """

        audio = butter_bandpass_filter(
            self.data.signal, lowcut, highcut, self.sample_rate, order, type=type
        )

        if overwrite is True:
            self.data.signal = audio
            self.audio = audio
        else:
            return list(audio)

    def envelope(self, overwrite=False):
        """
        Compute the envelope of the audio signal using the Hilbert transform.

        Args:
            overwrite (bool, optional): Whether to overwrite the current audio. Defaults to False.

        Returns:
            np.ndarray: Envelope of the audio signal.
        """
        envelope = np.abs(signal.hilbert(self.data.signal))

        if overwrite is True:
            self.data.signal = envelope
            self.audio = envelope

        if overwrite is False:
            return envelope

    def write_audio(self, filepath: str) -> None:
        """
        Write the audio data to a file.

        Args:
            filepath (str): Path to save the audio file. The file will be saved as a WAV file.
        """
        if ".wav" not in filepath:
            filepath = filepath + ".wav"

        sf.write(filepath, self.data.signal, self.sample_rate)

    def fade_in(self, fade_time=0.1, window="hann", overwrite=False):
        """
        Apply a fade-in effect to the audio.

        Args:
            fade_time (float, optional): Duration of the fade-in effect in seconds. Defaults to 0.1.
            window (str, optional): Window function to use. Defaults to "hann".
            overwrite (bool, optional): Whether to overwrite the current audio. Defaults to False.

        Returns:
            np.ndarray | None: Audio with fade-in effect if overwrite is False.
        """
        data = fade_in(self.data.signal.values, self.sample_rate, fade_time, window)

        if overwrite is True:
            self.data.signal = data
            self.audio = self.data.signal
        else:
            return data

    def fade_out(self, fade_time=0.1, window="hann", overwrite=False):
        """
        Apply a fade-out effect to the audio.

        Args:
            fade_time (float, optional): Duration of the fade-out effect in seconds. Defaults to 0.1.
            window (str, optional): Window function to use. Defaults to "hann".
            overwrite (bool, optional): Whether to overwrite the current audio. Defaults to False.

        Returns:
            np.ndarray | None: Audio with fade-out effect if overwrite is False.
        """
        data = fade_out(self.data.signal.values, self.sample_rate, fade_time, window)

        if overwrite is True:
            self.data.signal = data
            self.audio = self.data.signal
        else:
            return data


def combine_audio(list_of_files):  # pragma: no cover
    """
    Combine multiple audio files into one.

    Args:
        list_of_files (list): List of file paths to combine.

    Returns:
        Audio: Combined audio object.
    """

    combined = None

    for f in list_of_files:
        if combined is None:
            combined = Audio(f)
        else:
            combined.data.np.append(Audio(f).data)

    return combined


def butter_lowpass(cutoff, fs, order, type="sos"):  # pragma: no cover
    """
    Design a lowpass Butterworth filter.

    Args:
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling rate in Hz.
        order (int): Filter order.
        type (str, optional): Filter type ('sos' or 'ab'). Defaults to "sos".

    Returns:
        np.ndarray | tuple: Filter coefficients.
    """
    nyq = 0.5 * fs
    cutoff = cutoff / nyq

    if type == "ab":
        b, a = signal.butter(order, cutoff, btype="lowpass", analog=False)
        return b, a
    elif type == "sos":
        sos = signal.butter(order, cutoff, btype="lowpass", analog=False, output="sos")
        return sos


def butter_lowpass_filter(data, cutoff, fs, order=5, type="sos"):  # pragma: no cover
    """
    Apply a lowpass Butterworth filter to the data.

    Args:
        data (array-like): Input data to be filtered.
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling rate in Hz.
        order (int, optional): Order of the filter. Defaults to 5.
        type (str, optional): Type of the filter ('sos' or 'ab'). Defaults to "sos".

    Returns:
        array-like: Filtered data.
    """
    if type == "ab":
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
    elif type == "sos":
        sos = butter_lowpass(cutoff, fs, order=order, type="sos")
        y = signal.sosfiltfilt(sos, data)

    return y


def butter_highpass(cutoff, fs, order=5, type="sos"):  # pragma: no cover
    """
    Design a highpass Butterworth filter.

    Args:
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling rate in Hz.
        order (int): Order of the filter.
        type (str, optional): Type of the filter ('sos' or 'ab'). Defaults to "sos".

    Returns:
        np.ndarray | tuple: Filter coefficients.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if type == "ab":
        b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
        b, a
    elif type == "sos":
        sos = signal.butter(
            order, normal_cutoff, btype="high", analog=False, output="sos"
        )

        return sos


def butter_highpass_filter(data, cutoff, fs, order=5, type="sos"):  # pragma: no cover
    """
    Apply a highpass Butterworth filter to the data.

    Args:
        data (array-like): Input data to be filtered.
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling rate in Hz.
        order (int, optional): Order of the filter. Defaults to 5.
        type (str, optional): Type of the filter ('sos' or 'ab'). Defaults to "sos".

    Returns:
        array-like: Filtered data.
    """
    if type == "ab":
        b, a = butter_highpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
    elif type == "sos":
        sos = butter_highpass(cutoff, fs, order=order, type="sos")
        y = signal.sosfiltfilt(sos, data)

    return y


def butter_bandpass(lowcut, highcut, fs, order=5, type="sos"):  # pragma: no cover
    """
    Design a bandpass Butterworth filter.

    Args:
        lowcut (float): Low cutoff frequency in Hz.
        highcut (float): High cutoff frequency in Hz.
        fs (float): Sampling rate in Hz.
        order (int, optional): Order of the filter. Defaults to 5.
        type (str, optional): Type of the filter ('sos' or 'ab'). Defaults to "sos".

    Returns:
        np.ndarray | tuple: Filter coefficients.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if type == "ab":
        b, a = signal.butter(order, [low, high], btype="band", analog=False)
        return b, a
    elif type == "sos":
        sos = signal.butter(
            order, [low, high], analog=False, btype="band", output="sos"
        )
        return sos


def butter_bandpass_filter(
    data, lowcut, highcut, fs, order=5, type="sos"
):  # pragma: no cover
    """
    Apply a bandpass Butterworth filter to the data.

    Args:
        data (array-like): Input data to be filtered.
        lowcut (float): Low cutoff frequency in Hz.
        highcut (float): High cutoff frequency in Hz.
        fs (float): Sampling rate in Hz.
        order (int, optional): Order of the filter. Defaults to 5.
        type (str, optional): Type of the filter ('sos' or 'ab'). Defaults to "sos".

    Returns:
        array-like: Filtered data.
    """
    if type == "ab":
        b, a = butter_bandpass(lowcut, highcut, fs, order=order, type="ab")
        y = signal.filtfilt(b, a, data)
    elif type == "sos":
        sos = butter_bandpass(lowcut, highcut, fs, order=order, type="sos")
        y = signal.sosfiltfilt(sos, data)

    return y


def spectrogram(
    data: list or pd.Series,
    sample_rate: int,
    window_size: int = 8192,
    window="hann",
    nfft: int = 4096,
    noverlap: int = 4096,
    nperseg: int = 8192,
    time_format: str = "datetime",
    start: datetime = None,
    end: datetime = None,
) -> tuple:  # pragma: no cover
    """
    Generates spectrogram of audio

    Args:
        data (list or pd.Series): Data to generate spectrogram of
        sample_rate (int): Sample rate of data
        window_size (int, optional): Window size in samples. Defaults to 8192.
        nfft (int, optional): FFT number. Defaults to 4096.
        noverlap (int, optional): Overlap amount in samples. Defaults to 4096.
        nperseg (int, optional): Number of samples per segment. Defaults to 8192.
        start (datetime, optional): Start time. Defaults to None.
        end (datetime, optional): End time. Defaults to None.

    Returns:
        tuple: time, frequency, Pxx
    """
    if window == "hann":
        window = signal.windows.hann(window_size)
    elif window == "hamming":
        window = signal.windows.hamming(window_size)
    elif window == "blackman":
        window = signal.windows.blackman(window_size)
    elif window == "bartlett":
        window = signal.windows.bartlett(window_size)

    frequency, time, Pxx = signal.spectrogram(
        data,
        sample_rate,
        window=window,
        nfft=nfft,
        noverlap=noverlap,
        nperseg=nperseg,
        mode="psd",
    )

    if time_format == "datetime":
        if start:
            if end is None:
                end = start + timedelta(seconds=len(data) / sample_rate)
            datetime = pd.date_range(start, end, periods=len(time))

            time = datetime
    elif time_format == "samples":
        time = time * sample_rate
    elif time_format == "ms":
        time = time * 1000
    elif time_format == "seconds":
        time = time

    return time, frequency, Pxx


def write_audio(
    data: list or pd.Series, filepath: str, sample_rate: int
) -> None:  # pragma: no cover
    """
    Writes audiofile of data with set samplerate. Omit extension, will output only wav.

    Args:
        data (list or pd.Series): data to output
        filepath (str): filepath of output
        sample_rate (int): desired file sample rate
    """

    sf.write(filepath + ".wav", data, sample_rate)


def mp3_to_wav(
    input: str, output: str, output_format: str = "wav"
) -> None:  # pragma: no cover
    """
    Converts mp3 file to wav file.

    Args:
        file (file): filepath of input
        output (str): filepath of output
        output_format (str, optional): Output format. Defaults to "wav".
    """
    sound = AudioSegment.from_mp3(input)
    sound.export(output, format=output_format)


def psd2(
    x: list or pd.Series, sample_rate: int, window_size: int = 4096
) -> tuple:  # pragma: no cover
    """
    Compute the power spectral density of a signal.

    Args:
        x (array): signal
        sample_rate (int): sample rate of the signal
        sample_window (int, optional): length of the window to use for the FFT. Defaults to 4096.

    Returns:
        tuple: power spectral density
    """

    f = np.fft.rfft(x)
    f1 = f[0 : int(window_size / 2)]
    pf1 = 2 * np.abs(f1 * np.conj(f1)) / (sample_rate * window_size)
    lpf1 = 10 * np.log10(pf1)
    w = np.arange(1, window_size / 2 + 1)
    lp = lpf1[1 : int(window_size / 2)]
    w1 = sample_rate * w / window_size

    return w1, lp


def psd(
    x: list or pd.Series,
    sample_rate: int,
    window_size: int = 4096,
    window: str = "blackmanharris",
    scaling: str = "spectrum",
    time_format="amplitude",
) -> tuple:  # pragma: no cover
    if window == "blackmanharris":
        window = signal.windows.blackmanharris(window_size)
    elif window == "hann":
        window = signal.windows.hann(window_size)
    elif window == "hamming":
        window = signal.windows.hamming(window_size)
    elif window == "bartlett":
        window = signal.windows.bartlett(window_size)
    elif window == "blackman":
        window = signal.windows.blackman(window_size)
    elif window == "boxcar":
        window = signal.windows.boxcar(window_size)

    freq, amp = signal.periodogram(x, fs=sample_rate, window=window, scaling=scaling)

    if time_format == "amplitude":
        amp = 10 * np.log10(amp)
    else:
        amp = amp

    return freq, amp


# %%
def peak_hold(data, window=8 * 1024, sample_rate=24000):  # pragma: no cover
    df = pd.DataFrame()
    samples = 0
    while samples < sample_rate * len(data):
        d = data[samples : samples + window]
        if len(d) < window:
            break

        freq, amp = signal.periodogram(
            d,
            fs=sample_rate,
            window=signal.windows.blackmanharris(window),
            scaling="spectrum",
        )

        if "frequency" not in df.columns:
            df["frequency"] = freq
        if "amplitude" not in df.columns:
            df["amplitude"] = amp
        else:
            df["amplitude"] = [
                amp[i] if amp[i] > df.amplitude[i] else df.amplitude[i]
                for i in range(len(amp))
            ]
        samples += window
    return df


def average_hold(data, window=1024, sample_rate=24000):  # pragma: no cover
    df = pd.DataFrame()
    samples = 0
    while samples < sample_rate * len(data):
        d = data[samples : samples + window]
        if len(d) < window:
            break

        freq, amp = signal.periodogram(
            d,
            fs=sample_rate,
            window=signal.windows.blackmanharris(window),
            scaling="spectrum",
        )

        if "frequency" not in df.columns:
            df["frequency"] = freq
        if "amplitude" not in df.columns:
            df["amplitude"] = amp
        else:
            df["amplitude"] += amp

        samples += window

    df["amplitude"] = df["amplitude"] / (samples)

    return df


def fade_in(data, sample_rate, fade_time=0.1, window="hann"):  # pragma: no cover
    fade_samples = int(sample_rate * fade_time)

    if window == "hann":
        fade = signal.windows.hann(fade_samples * 2)[:fade_samples]

    data[:fade_samples] = data[:fade_samples] * fade

    return data


def fade_out(data, sample_rate, fade_time=0.1, window="hann"):  # pragma: no cover
    fade_samples = int(sample_rate * fade_time)

    if window == "hann":
        fade = signal.windows.hann(fade_samples * 2)[fade_samples:]

    data[-fade_samples:] = data[-fade_samples:] * fade

    return data


def echo(data,  sample_rate, delay=0.1, decay=0.5):
    """
    Adds echo to audio data.

    Args:
        data (array | np.array | pd.Series): audio data to add echo to
        sample_rate (int | float): sample rate of audio data
        delay (int | float, optional): delay in the echo. Defaults to 0.1.
        decay (int | float, optional): decay of the echo. Defaults to 0.5.

    Returns:
        list|np.array|pd.Series: _description_
    """


    delay_samples = int(sample_rate * delay)
    # decay_samples = int(sample_rate * decay)
    echo = np.zeros(len(data) + delay_samples)
    echo[delay_samples:] = data * decay
    echo[: len(data)] += data

    return echo

def spectral_subtraction(signal, noise, scaling_factor=1.0):
    """
    Perform spectral subtraction with optional scaling of the noise spectrum.

    Parameters:
        signal (array-like): The input signal.
        noise (array-like): The noise signal.
        scaling_factor (float): A factor to scale the noise spectrum before subtraction.

    Returns:
        array-like: The cleaned signal after spectral subtraction.
    """
    signal_fft = np.fft.fft(signal)
    noise_fft = np.fft.fft(noise)
    
    # Scale the noise spectrum
    scaled_noise_fft = scaling_factor * np.abs(noise_fft)
    
    # Perform spectral subtraction
    clean_fft = np.maximum(np.abs(signal_fft) - scaled_noise_fft, 0)
    clean_signal = np.fft.ifft(clean_fft * np.exp(1j * np.angle(signal_fft)))
    
    return np.real(clean_signal)


def wavelet_denoise(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, mode='symmetric', level=level)
    threshold = np.sqrt(2 * np.log(len(signal))) * np.std(coeffs[-1])
    
    denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c for i, c in enumerate(coeffs)]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    
    return denoised_signal

def autocorrelation(signal):
    autocorr = np.correlate(signal, signal, mode='same')
    return autocorr[len(signal)-1:]