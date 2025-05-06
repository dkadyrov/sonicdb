from datetime import timedelta

from sqlalchemy import Column, DateTime, ForeignKey, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.orm.session import Session

from sonicdb import audio

from .base import Base


class Sample(Base):  # type: ignore
    """
    Represents a sample in the database.

    Attributes:
        id (int): Unique identifier for the sample.
        event_id (int): Foreign key linking to the associated event.
        event (Event): Relationship to the Event object.
        sensor_id (int): Foreign key linking to the associated sensor.
        sensor (Sensor): Relationship to the Sensor object.
        channel_id (int): Foreign key linking to the associated channel.
        channel (Channel): Relationship to the Channel object.
        subject_id (int): Foreign key linking to the associated subject.
        subject (Subject): Relationship to the Subject object.
        datetime (datetime): Datetime of the sample data.
        file_id (int): Foreign key linking to the associated file.
        file (File): Relationship to the File object where the sample audio comes from.
    """

    __tablename__ = "sample"
    id = Column(Integer, primary_key=True)
    """int: Unique identifier for the sample."""

    event_id = Column(Integer, ForeignKey("event.id"))
    """int: Foreign key linking to the event's ID."""

    event = relationship(
        "sonicdb.database.event.Event",
        back_populates="samples",
        enable_typechecks=False,
    )
    """Event: Associated event object."""

    sensor_id = Column(Integer, ForeignKey("sensor.id"))
    """int: Foreign key linking to the sensor's ID."""

    sensor = relationship(
        "sonicdb.database.sensor.Sensor",
        back_populates="samples",
        enable_typechecks=False,
    )
    """Sensor: Associated sensor object."""

    channel_id = Column(Integer, ForeignKey("channel.id"))
    """int: Foreign key linking to the channel's ID."""

    channel = relationship(
        "sonicdb.database.channel.Channel",
        back_populates="samples",
        enable_typechecks=False,
    )
    """Channel: Associated channel object."""

    subject_id = Column(Integer, ForeignKey("subject.id"))
    """int: Foreign key linking to the subject's ID."""

    subject = relationship(
        "sonicdb.database.subject.Subject",
        back_populates="samples",
        enable_typechecks=False,
    )
    """Subject: Associated subject object."""

    datetime = Column(DateTime)
    """datetime: Datetime of the sample data."""

    file_id = Column(Integer, ForeignKey("file.id"))
    """int: Foreign key linking to the file's ID."""

    file = relationship(
        "sonicdb.database.file.File", back_populates="samples", enable_typechecks=False
    )
    """File: Associated file object where the sample audio comes from."""

    __mapper_args__ = {
        "polymorphic_identity": "sample",
    }

    def get_audio(self) -> audio.Audio:
        """
        Retrieves the audio data associated with the sample.

        Returns:
            audio.Audio: The audio data of the sample.
        """

        session = Session.object_session(self)
        if session is None:
            raise ValueError("Session is not available for this sample.")

        a = session.get_audio(
            start=self.datetime,
            end=self.datetime + timedelta(seconds=session.sample_duration),
            sensor=self.sensor,
            channel=self.channel,
        )

        return a
