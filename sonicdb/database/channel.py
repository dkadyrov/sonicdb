from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Float
from sqlalchemy.orm import relationship

from .base import Base
from datetime import datetime


class Channel(Base):  # type: ignore
    """
    Represents a channel of a sensor in the database.

    Attributes:
        id (int): Unique identifier for the channel.
        number (int): Channel number of the sensor.
        type_class (str): Type of sensor.
        gain (float): Gain of the sensor.
        sensor_id (int): Foreign key referencing the associated sensor.
        sensor (Sensor): Relationship to the Sensor object.
        files (list[File]): List of File objects associated with the channel.
        samples (list[Sample]): List of Sample objects associated with the channel.
        events (list[EventChannel]): List of EventChannel objects associated with the channel.
    """

    __tablename__ = "channel"
    id = Column(Integer, primary_key=True)
    """int: Unique identifier for the channel."""

    number = Column(Integer)
    """int: Channel number of the sensor."""

    type_class = Column(String)
    """str: Type or classification of the sensor."""

    gain = Column(Float)
    """float: Gain value of the sensor."""

    sensor_id = Column(Integer, ForeignKey("sensor.id"))
    """int: Foreign key linking to the sensor's ID."""

    sensor = relationship(
        "sonicdb.database.sensor.Sensor",
        back_populates="channels",
        enable_typechecks=False,
    )
    """Sensor: Associated sensor object."""

    files = relationship(
        "sonicdb.database.file.File", back_populates="channel", enable_typechecks=False
    )
    """list[File]: List of files associated with the channel."""

    samples = relationship(
        "sonicdb.database.sample.Sample",
        back_populates="channel",
        enable_typechecks=False,
    )
    """list[Sample]: List of samples associated with the channel."""

    events = relationship(
        "sonicdb.database.event.EventChannel",
        back_populates="channel",
        enable_typechecks=False,
    )
    """list[EventChannel]: List of events associated with the channel."""

    __mapper_args__ = {"polymorphic_identity": "channel"}

    def __repr__(self) -> str:  # pragma: no cover
        """Return a string representation of the channel.

        Returns:
            str: String representation of the channel.
        """
        return f"Channel: {self.number} of Sensor"

    def get_start(self: "Channel") -> datetime:  # pragma: no cover
        """Get the earliest datetime from the channel's files.

        Returns:
            datetime: The earliest recording datetime.
        """
        return min([f.start for f in self.files if isinstance(f.start, datetime)])

    def get_end(self: "Channel") -> datetime:  # pragma: no cover
        """Get the latest datetime from the channel's files.

        Returns:
            datetime: The latest recording datetime.
        """
        return max([f.end for f in self.files if isinstance(f.end, datetime)])

    def dict(self: "Channel") -> dict[str, int]:  # pragma: no cover
        """Convert the channel's attributes to a dictionary.

        Returns:
            dict[str, int]: Dictionary of channel attributes.
        """
        return {k.title(): v for k, v in self.__dict__.items()}
