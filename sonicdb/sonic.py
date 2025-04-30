# type: ignore

import pathlib

import librosa
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy_utils import database_exists

from sonicdb import audio
from sonicdb.models import Base, Channel, Sensor, Event, Subject, File
from sonicdb import utilities

from datetime import datetime


class Database:  # pragma: no cover
    """SONIC Database class"""

    def __init__(self, db: str):
        # TODO Add support for other databases
        self.engine = create_engine(f"sqlite:///{db}?check_same_thread=False")
        if database_exists(self.engine.url):
            Base.metadata.bind = self.engine
        else:
            Base.metadata.create_all(self.engine)
        DBSession = sessionmaker(bind=self.engine, autoflush=False)
        DBSession = scoped_session(DBSession)
        self.session = DBSession()
        """
        Inherits the DBSession class from SQLAlchemy. `Available here <https://docs.sqlalchemy.org/en/14/orm/session.html>`_.
        """

        # TODO Set values through YAML, JSON, and/or XML input
        self.session.sample_duration = 60
        """int: duration of the sample in seconds"""

        self.session.sample_overlap = 0
        """int: overlap of the sample in seconds"""

        self.session.sample_rate = 25000
        """int: default sample rate """

        self.session.directory = pathlib.Path(db).parent

    def get_audio(
        self,
        start: datetime = None,
        end: datetime = None,
        event: Event = None,
        sensor: Sensor = None,
        channel: Channel = None,
        channel_number: int = None,
    ) -> audio.Audio:
        """
        Get audio data from the database.

        Args:
            start (datetime): Start time of the audio data.
            end (datetime): End time of the audio data.
            event (Event): Event object to get the audio data from.
            sensor (Sensor): Sensor object to get the audio data from.
            channel (Channel): Channel object to get the audio data from.
            channel_number (int): Channel number to get the audio data from.

        Returns:
            audio.Audio: Audio object containing the audio data.
        """

        init_start = start
        init_end = end

        if start:
            if not isinstance(start, utilities.datetime):
                start = utilities.read_datetime(start)

        if end:
            if not isinstance(start, utilities.datetime):
                end = utilities.read_datetime(end)

        if event:
            if not start:
                start = event.start
            if not end:
                end = event.end

        if channel:
            if isinstance(channel, Channel):
                channel = channel
                sensor = channel.sensor
                channel_number = channel.number

        elif sensor:
            if isinstance(sensor, Sensor):
                sensor = sensor
            else:
                sensor = self.get_sensor(sensor)

        files = (
            self.session.query(File)
            .filter(File.start <= end)
            .filter(File.end >= start)
            .filter(File.sensor == sensor)
            .filter(File.channel_number == channel_number)
            .all()
        )

        if len(files) == 0:
            return None

        sample_rate = files[0].sample_rate

        file_start = start

        length = abs((end - start).total_seconds() * sample_rate)

        data = []
        for file in files:
            filepath = pathlib.PurePath(self.session.directory, file.filepath)
            offset = (start - file.start).total_seconds()

            if offset < 0:
                data.extend([0] * int(-offset * sample_rate))
                offset = 0

            duration = (file.end - end).total_seconds()

            if duration < 0:
                if offset >= file.duration:
                    data.extend(librosa.load(filepath, sr=sample_rate)[0].tolist())
                else:
                    data.extend(
                        librosa.load(filepath, offset=offset, sr=sample_rate)[
                            0
                        ].tolist()
                    )
            else:
                data.extend(
                    librosa.load(
                        filepath,
                        offset=offset,
                        duration=file.duration - duration - offset,
                        sr=sample_rate,
                    )[0].tolist()
                )

            start = file.end

        if len(data) < length:
            data.extend([0.0] * int(length - len(data)))
        if len(data) > length:
            data = data[: int(length)]

        a = audio.Audio(
            audio=np.asarray(data), sample_rate=sample_rate, start=file_start
        )
        a = a.trim(init_start, init_end)

        return a

    def get_sensor(
        self, sensor: Sensor | int | dict[str, int] | str
    ) -> Sensor | None: 
        """
        Get a sensor from the database.

        Args:
            sensor (Sensor | int | dict[str, int] | str): Sensor object, sensor ID, or sensor name.
                If a dictionary is passed, it should contain the keys "name" and "subname". 
        
        Returns:
            Sensor | None: Sensor object if found, None otherwise.         
        """
        if isinstance(sensor, Sensor):
            return sensor
        elif isinstance(sensor, int):
            return self.session.query(Sensor).get(sensor)
        elif isinstance(sensor, dict):
            sensor = utilities.lower_keys(sensor)
            s = (
                self.session.query(Sensor)
                .filter(
                    Sensor.name == sensor["name"],
                    Sensor.subname == sensor["subname"],
                )
                .all()
            )
            if len(s) == 0:
                return None
            return s[0]
        elif isinstance(sensor, str):
            s = self.session.query(Sensor).filter(Sensor.name == sensor).all()
            if len(s) == 0:
                return None
            return s[0]

        return None

    def get_subject(
        self, subject: Subject | int | dict[str, int] | str
    ) -> Subject | None:
        """
        Get a subject from the database.

        Args:
            subject (Subject | int | dict[str, int] | str): Subject object, subject ID, or subject name.
                If a dictionary is passed, it should contain the key "name".

        Returns:
            Subject | None: Subject object if found, None otherwise.
        """
        if isinstance(subject, Subject):
            return subject
        elif isinstance(subject, int):
            return self.session.query(Subject).get(subject)
        elif isinstance(subject, dict):
            subject = utilities.lower_keys(subject)
            s = (
                self.session.query(Subject)
                .filter(Subject.name == subject["name"])
                .all()
            )
            if len(s) == 0:
                return None
            return s[0]
        elif isinstance(subject, str):
            s = self.session.query(Subject).filter(Subject.name == subject).all()
            if len(s) == 0:
                return None
            return s[0]

        return None

    def get_channel(
        self, channel: Channel | int | dict[str, int], sensor: Sensor | None = None
    ) -> Channel | None:
        """
        Get a channel from the database.

        Args:
            channel (Channel | int | dict[str, int]): Channel object, channel ID, or channel details as a dictionary.
                If a dictionary is passed, it should contain the key "number".
            sensor (Sensor | None): Sensor object or sensor ID associated with the channel.

        Returns:
            Channel | None: Channel object if found, None otherwise.
        """
        if isinstance(channel, Channel):
            return channel
        elif isinstance(channel, int):
            if sensor is None:
                return self.session.query(Channel).get(channel)
            else:
                channel = {"number": channel}

        channel = utilities.lower_keys(channel)
        if sensor:
            channel["sensor"] = sensor

        if not isinstance(channel["sensor"], Sensor):
            channel["sensor"] = self.get_sensor(channel["sensor"])

        c = (
            self.session.query(Channel)
            .filter(Channel.number == channel["number"])
            .filter(Channel.sensor == channel["sensor"])
            .all()
        )

        if len(c) == 0:
            return None

        return c[0]

    # TODO Add sample support

    # TODO Add resample support
