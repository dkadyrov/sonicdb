# type: ignore
import os
import re
from datetime import datetime
from datetime import timedelta

import librosa
import pandas as pd
from dateutil.parser import parse


audiofiles = [
    "wav",
    "mp3",
    "aiff",
    "flac",
    "ogg",
    "wma",
    "m4a",
    "aac",
    "alac",
    "aif",
    "aifc",
    "aiffc",
    "au",
    "snd",
    "cdda",
    "raw",
    "mpc",
    "vqf",
    "tta",
    "wv",
    "ape",
    "ac3",
    "dts",
    "dtsma",
    "dtshr",
    "dtshd",
    "eac3",
    "thd",
    "thd+ac3",
    "thd+dts",
    "thd+dd",
    "thd+dd+ac3",
    "thd+dd+dts",
    "thd+dd+dtsma",
    "thd+dd+dtshr",
    "thd+dd+dtshd",
]


def lower_keys(tree: dict) -> dict[str, int]:  # pragma: no cover
    """Normalizes a dictionary to have lowercase and snakecase keys

    :param tree: tree
    :type tree: dict
    :return: normalized tree
    :rtype: dict
    """

    data = {}
    for k in tree.keys():
        if isinstance(tree[k], dict):
            data[k.lower().replace(" ", "_")] = lower_keys(tree[k])
        else:
            data[k.lower().replace(" ", "_")] = tree[k]
    return data


def read_datetime(string: str) -> datetime:  # pragma: no cover
    """
    Reads and converts datetime

    Args:
        string (String): datetime string

    Returns:
        datetime.Datetime: converted datetime
    """

    try:
        return datetime.strptime(string, "%Y_%m_%d")
    except Exception:
        try:
            return datetime.strptime(string, "%Y_%m_%d_%H_%M_%S.%f")
        except Exception:
            try:
                return datetime.strptime(string, "%Y_%m_%d_%H_%M_%S")
            except Exception:
                return parse(timestr=string, fuzzy=True)


def metadata(filepath: str, extended=False) -> dict:  # pragma: no cover
    """
    Generates metadata of file

    Args:
        filepath (str): filepath of file

    Returns:
        dict: metadata of file
    """

    metadata = {}
    metadata["filepath"] = filepath
    metadata["filename"] = os.path.basename(filepath)
    metadata["extension"] = os.path.splitext(filepath)[1].replace(".", "")
    metadata["directory"] = os.path.dirname(filepath)
    metadata["size"] = os.path.getsize(filepath)
    metadata["modified"] = datetime.fromtimestamp(os.path.getmtime(filepath))
    metadata["created"] = datetime.fromtimestamp(os.path.getctime(filepath))
    metadata["accessed"] = datetime.fromtimestamp(os.path.getatime(filepath))

    if extended:
        if metadata["extension"] in audiofiles:
            try:
                metadata["channel"] = int(
                    re.findall(r"\d+", metadata["filename"].split("_")[-2])[0]
                )
            except Exception:
                metadata["channel"] = None

            try:
                metadata["sample_rate"] = librosa.get_samplerate(metadata["filepath"])
            except Exception:
                metadata["sample_rate"] = None

            try:
                metadata["duration"] = librosa.get_duration(path=metadata["filepath"])
            except Exception:
                metadata["duration"] = None

            try:
                metadata["record_number"] = int(
                    metadata["filename"].split("_")[-1].split(".")[0]
                )
            except Exception:
                metadata["record_number"] = None

            try:
                metadata["start"] = read_datetime(metadata["filename"][:23])
            except Exception:
                metadata["start"] = None

            try:
                metadata["end"] = metadata["start"] + timedelta(
                    seconds=metadata["duration"]
                )
            except Exception:
                metadata["end"] = None

    return metadata


def metadatas(
    filepaths: list, extended=False, stevens=False
) -> pd.DataFrame:  # pragma: no cover
    """
    Generates metadata of files

    Args:
        filepaths (list): list of filepaths

    Returns:
        pd.DataFrame: metadata of files
    """

    metadatas = pd.DataFrame([metadata(filepath, extended) for filepath in filepaths])

    if len(metadatas) == 0:
        return metadatas

    if stevens:
        for row, group in metadatas.groupby(["start", "channel"]):
            if len(group) > 1:
                for j, row in group.iterrows():
                    if row.record_number > 1:
                        metadatas.at[row.name, "start"] = metadatas.iloc[
                            row.name - 1
                        ].end
                        metadatas.at[row.name, "end"] = metadatas.iloc[
                            row.name
                        ].start + timedelta(seconds=row.duration)

    return metadatas
