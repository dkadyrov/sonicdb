from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import relationship

from .base import Base


class Event(Base):  # type: ignore
    """
    Represents an event in the database.

    Attributes:
        id (int): Unique identifier for the event.
        name (str): Name of the event.
        start (datetime): Start time of the event.
        end (datetime): End time of the event.
        description (str): Description of the event.
        subject_id (int): Foreign key linking to the associated subject.
        subject (Subject): Relationship to the Subject object.
        samples (list[Sample]): List of Sample objects associated with the event.
        channels (list[EventChannel]): List of EventChannel objects associated with the event.
    """

    __tablename__ = "event"
    id = Column(Integer, primary_key=True)
    """int: Unique identifier for the event."""

    name = Column(String)
    """str: Name of the event."""

    start = Column(DateTime)
    """datetime: Start time of the event."""

    end = Column(DateTime)
    """datetime: End time of the event."""

    description = Column(String)
    """str: Description of the event."""

    subject_id = Column(Integer, ForeignKey("subject.id"))
    """int: Foreign key linking to the subject's ID."""

    subject = relationship(
        "sonicdb.database.subject.Subject",
        back_populates="events",
        enable_typechecks=False,
    )
    """Subject: Associated subject object."""

    samples = relationship(
        "sonicdb.database.sample.Sample", back_populates="event", enable_typechecks=False
    )
    """list[Sample]: List of samples associated with the event."""

    channels = relationship(
        "sonicdb.database.event.EventChannel",
        back_populates="event",
        enable_typechecks=False,
    )
    """list[EventChannel]: List of channels associated with the event."""

    __mapper_args__ = {
        "polymorphic_identity": "event",
    }

    def __repr__(self) -> str:  # pragma: no cover
        """Return a string representation of the event.

        Returns:
            str: String representation of the event.
        """
        return f"Event: {self.name}"


class EventChannel(Base):  # type: ignore
    """
    Represents the association between an event and a channel.

    Attributes:
        event_id (int): Foreign key linking to the event's ID.
        event (Event): Relationship to the Event object.
        channel_id (int): Foreign key linking to the channel's ID.
        channel (Channel): Relationship to the Channel object.
    """

    __tablename__ = "event_channel"

    event_id = Column(ForeignKey("event.id"), primary_key=True)  # type: ignore
    """int: Foreign key linking to the event's ID."""

    event = relationship(
        "sonicdb.database.event.Event", back_populates="channels", enable_typechecks=False
    )
    """Event: Associated event object."""

    channel_id = Column(ForeignKey("channel.id"), primary_key=True)  # type: ignore
    """int: Foreign key linking to the channel's ID."""

    channel = relationship(
        "sonicdb.database.channel.Channel",
        back_populates="events",
        enable_typechecks=False,
    )
    """Channel: Associated channel object."""

    __mapper_args__ = {
        "polymorphic_identity": "event_channel",
    }

    def __repr__(self) -> str:  # pragma: no cover
        """Return a string representation of the event-channel association.

        Returns:
            str: String representation of the event-channel association.
        """
        return f"{self.event} {self.channel}"
