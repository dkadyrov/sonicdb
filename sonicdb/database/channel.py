from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Float
from sqlalchemy.orm import relationship

from .base import Base
from datetime import datetime


class Channel(Base):  # type: ignore
    """Channel object"""

    __tablename__ = "channel"
    id = Column(Integer, primary_key=True)
    """int: Channel database ID"""

    number = Column(Integer)
    """int: Channel number of the sensor"""

    type_class = Column(String)
    """str: Type of sensor"""

    gain = Column(Float)
    """int: Gain of the sensor"""

    sensor_id = Column(Integer, ForeignKey("sensor.id"))
    """int: Sensor database ID"""

    sensor = relationship(
        "sonicdb.database.sensor.Sensor",
        back_populates="channels",
        enable_typechecks=False,
    )
    """Sensor: Sensor object"""

    files = relationship(
        "sonicdb.database.file.File", back_populates="channel", enable_typechecks=False
    )
    """list: List of channel's File objects"""

    samples = relationship(
        "sonicdb.database.sample.Sample",
        back_populates="channel",
        enable_typechecks=False,
    )
    """list: List of channel's Sample objects"""

    events = relationship(
        "sonicdb.database.event.EventChannel",
        back_populates="channel",
        enable_typechecks=False,
    )

    __mapper_args__ = {"polymorphic_identity": "channel"}

    def __repr__(self) -> str:  # pragma: no cover
        """Returns object representation"""

        return f"Channel: {self.number} of Sensor"

    def get_start(self: "Channel") -> datetime:  # pragma: no cover
        """Returns the earliest datetime of channel files

        :return: Earliest recording from channel
        :rtype: datetime
        """

        return min([f.start for f in self.files if isinstance(f.start, datetime)])

    def get_end(self: "Channel") -> datetime:  # pragma: no cover
        """Returns the last datetime of channel files

        :return: Final recording time from channel
        :rtype: datetime
        """

        return max([f.end for f in self.files if isinstance(f.end, datetime)])

    def dict(self: "Channel") -> dict[str, int]:  # pragma: no cover
        """Returns channel parameters and attributes as dictionary

        :return: channel parameters as dictionary
        :rtype: dict
        """

        return {k.title(): v for k, v in self.__dict__.items()}
