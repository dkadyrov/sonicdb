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
from mpl_toolkits.axes_grid1 import make_axes_locatable


np.seterr(divide="ignore")
valid_audio = ["wav", "flac", "mp3", "ogg", "aiff", "au"]


class Audio:  # pragma: no cover
    """
    Audio class for handling audio files
    """

    def __init__(self, filepath=None, audio=None, sample_rate=None, start=None):
        if filepath:
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
        Adds data from another audio file to this one

        Args:
            filepath (str): filepath to audio file to add
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

    def trim(
        self,
        start: datetime or str or float or int,
        end: datetime or str or float or int = None,
        length: float = None,
        time_format="datetime",
        restart=False,
    ):
        """
        Trims audio to specified start and end times or length

        Args:
            start (datetime or str): Start time of audio
            end (datetime or str, optional): End time of audio. Defaults to None.
            length (float, optional): Length of audio sample in seconds, milliseconds, or samples. Defaults to None.

        Returns:
            audio.Audio: Trimmed audio sample
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
        Resamples audio to sample rate

        Args:
            sample_rate (int): Sample rate to resample audio to
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

    def spectrogram(
        self,
        window="hann",
        window_size: int = 8192,
        nfft: int = 8192,
        noverlap: int = 4096,
        nperseg: int = 8192,
        time_format="datetime",
    ) -> tuple:
        """
        Generates spectrogram of audio

        Args:
            window_size (int, optional): Window size. Defaults to 8192.
            nfft (int, optional): Number for FFT. Defaults to 4096.
            noverlap (int, optional): Sample overlap. Defaults to 4096.
            nperseg (int, optional): Number of Samples. Defaults to 8192.

        Returns:
            tuple: time, frequency, Pxx
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

    def plot_spectrogram(
        self,
        window="hann",
        window_size: int = 8192,
        nfft: int = 8192,
        noverlap: int = 4096,
        nperseg: int = 8192,
        zmin: int = None,
        zmax: int = None,
        gain: int = 0,
        showscale: bool = False,
        cmap="jet",
        aspect="auto",
        time_format="datetime",
        fig=None,
        ax=None,
    ):
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

    def plot_melspectrogram(
        self,
        window="hann",
        nmels: int = 8192,
        window_size: int = 8192,
        nfft: int = 8192,
        noverlap: int = 4096,
        nperseg: int = 8192,
        zmin: int = None,
        zmax: int = None,
        gain: int = 0,
        showscale: bool = False,
        cmap="jet",
        aspect="auto",
        time_format="datetime",
        ax=None,
        fmin=0,
        fmax=None,
    ):
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
        Generates the power spectral density of the audio

        Args:
            window_size (int, optional): Sample window size. Defaults to 4096.

        Returns:
            tuple: frequency, power
        """
        frequency, power = psd(
            self.data.signal, self.sample_rate, window_size=window_size
        )

        return frequency, power

    def plot_psd(self, window_size: int = 4096) -> tuple:
        """
        Plots the power spectral density of the audio

        Args:
            window_size (int, optional): Sample window size. Defaults to 4096.

        Returns:
            tuple: figure, axis
        """
        frequency, power = self.psd(window_size=window_size)

        fig, ax = plt.subplots()
        ax.plot(frequency, power)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power [dB]")

        return fig, ax

    def lowpass_filter(self, cutoff, order=4, overwrite=False, type="sos"):
        """
        Lowpass filter using Butterworth filter

        Args:
            cutoff (int): Cutoff frequency
            order (int): Order of filter

        Returns:
            Audio: Filtered audio
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
        Highpass filter using Butterworth filter

        Args:
            cutoff (int): Cutoff frequency
            order (int): Order of filter

        Returns:
            Audio: Filtered audio
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
        Bandpass filter using Butterworth filter

        Args:
            lowcut (int): Low cutoff frequency
            highcut (int): High cutoff frequency
            order (int): Order of filter

        Returns:
            Audio: Filtered audio
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
        envelope = np.abs(signal.hilbert(self.data.signal))

        if overwrite is True:
            self.data.signal = envelope
            self.audio = envelope

        if overwrite is False:
            return envelope

    def write_audio(self, filepath: str) -> None:
        """
        Writes audiofile of data with set samplerate. Omit extension, will output only wav.

        Args:
            data (list or pd.Series): data to output
            filepath (str): filepath of output
            sample_rate (int): desired file sample rate
        """
        if ".wav" not in filepath:
            filepath = filepath + ".wav"

        sf.write(filepath, self.data.signal, self.sample_rate)

    def fade_in(self, fade_time=0.1, window="hann", overwrite=False):
        data = fade_in(self.data.signal.values, self.sample_rate, fade_time, window)

        if overwrite is True:
            self.data.signal = data
            self.audio = self.data.signal
        else:
            return data

    def fade_out(self, fade_time=0.1, window="hann", overwrite=False):
        data = fade_out(self.data.signal.values, self.sample_rate, fade_time, window)

        if overwrite is True:
            self.data.signal = data
            self.audio = self.data.signal
        else:
            return data


def combine_audio(list_of_files):  # pragma: no cover
    """
    Combines audio files into one audio file

    Args:
        list_of_files (list): List of audio files to combine

    Returns:
        Audio: Combined audio file
    """

    combined = None

    for f in list_of_files:
        if combined is None:
            combined = Audio(f)
        else:
            combined.data.np.append(Audio(f).data)

    return combined


def butter_lowpass(cutoff, fs, order, type="sos"):  # pragma: no cover
    nyq = 0.5 * fs
    cutoff = cutoff / nyq

    if type == "ab":
        b, a = signal.butter(order, cutoff, btype="lowpass", analog=False)
        return b, a
    elif type == "sos":
        sos = signal.butter(order, cutoff, btype="lowpass", analog=False, output="sos")
        return sos


def butter_lowpass_filter(data, cutoff, fs, order=5, type="sos"):  # pragma: no cover
    if type == "ab":
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
    elif type == "sos":
        sos = butter_lowpass(cutoff, fs, order=order, type="sos")
        y = signal.sosfiltfilt(sos, data)

    return y


def butter_highpass(cutoff, fs, order=5, type="sos"):  # pragma: no cover
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
    if type == "ab":
        b, a = butter_highpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
    elif type == "sos":
        sos = butter_highpass(cutoff, fs, order=order, type="sos")
        y = signal.sosfiltfilt(sos, data)

    return y


def butter_bandpass(lowcut, highcut, fs, order=5, type="sos"):  # pragma: no cover
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


def echo(data, sample_rate, delay=0.1, decay=0.5):  # pragma: no cover
    delay_samples = int(sample_rate * delay)
    decay_samples = int(sample_rate * decay)
    echo = np.zeros(len(data) + delay_samples)
    echo[delay_samples:] = data * decay
    echo[: len(data)] += data

    return echo
