from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import relationship
from sqlalchemy.orm.session import Session

from .base import Base
import pathlib

from sonicdb import audio


class File(Base):  # type: ignore # pragma: no cover
    """
    Represents an audio file in the database.

    Attributes:
        id (int): Unique identifier for the file.
        filepath (str): Path to the file on the filesystem.
        filename (str): Name of the file.
        extension (str): File extension (e.g., '.wav').
        sample_rate (int): Sampling rate of the audio file in Hz.
        start (datetime): Start time of the audio file.
        end (datetime): End time of the audio file.
        duration (float): Duration of the audio file in seconds.
        channel_number (int): Channel number associated with the file.
        channel_id (int): Foreign key referencing the associated channel.
        channel (Channel): Relationship to the Channel object.
        sensor_id (int): Foreign key referencing the associated sensor.
        sensor (Sensor): Relationship to the Sensor object.
        samples (list): List of Sample objects generated from the file.
    """

    __tablename__ = "file"
    id = Column(Integer, primary_key=True)
    """int: Unique identifier for the file in the database."""

    filepath = Column(String)
    """str: Path to the file on the filesystem."""

    filename = Column(String)
    """str: Name of the file."""

    extension = Column(String)
    """str: File extension (e.g., '.wav')."""

    sample_rate = Column(Integer)
    """int: Sampling rate of the audio file in Hz."""

    start = Column(DateTime)
    """datetime: Start time of the audio file."""

    end = Column(DateTime)
    """datetime: End time of the audio file."""

    duration = Column(Float)
    """float: Duration of the audio file in seconds."""

    channel_number = Column(Integer)
    """int: Channel number associated with the file."""

    channel_id = Column(Integer, ForeignKey("channel.id"))
    """int: Foreign key referencing the associated channel."""

    channel = relationship(
        "sonicdb.database.channel.Channel",
        back_populates="files",
        enable_typechecks=False,
    )
    """Channel: Relationship to the Channel object."""

    sensor_id = Column(Integer, ForeignKey("sensor.id"))
    """int: Foreign key referencing the associated sensor."""

    sensor = relationship(
        "sonicdb.database.sensor.Sensor", back_populates="files", enable_typechecks=False
    )
    """Sensor: Relationship to the Sensor object."""

    samples = relationship(
        "sonicdb.database.sample.Sample", back_populates="file", enable_typechecks=False
    )
    """list: List of Sample objects generated from the file."""

    __mapper_args__ = {
        "polymorphic_identity": "file",
    }

    def __repr__(self) -> str:  # pragma: no cover
        """
        Returns a string representation of the File object.

        Returns:
            str: A string in the format 'File <id>: <filename>'.
        """
        return f"File {self.id}: {self.filename}"

    def get_audio(self) -> audio.Audio:  # type: ignore # pragma: no cover
        """
        Retrieves the audio data for the file.

        Returns:
            audio.Audio: An Audio object containing the audio data.
        """
        session = Session.object_session(self)
        directory = session.directory
        filepath = pathlib.PurePath(directory, self.filepath)

        return audio.Audio(filepath)
