from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import relationship

from .base import Base


class Subject(Base):  # type: ignore
    """
    Represents a subject in the database.

    Attributes:
        id (int): Unique identifier for the subject.
        name (str): Name of the subject.
        samples (list): List of Sample objects featuring the subject.
        events (list): List of Event objects featuring the subject.
    """

    __tablename__ = "subject"
    id = Column(Integer, primary_key=True)
    """int: Unique identifier for the subject."""

    name = Column(String)
    """str: Name of the subject."""

    samples = relationship(
        "sonicdb.database.sample.Sample",
        back_populates="subject",
        enable_typechecks=False,
    )
    """list: List of Sample objects featuring the subject."""

    events = relationship(
        "sonicdb.database.event.Event", back_populates="subject", enable_typechecks=False
    )
    """list: List of Event objects featuring the subject."""

    __mapper_args__ = {"polymorphic_identity": "subject"}

    def __repr__(self) -> str:  # pragma: no cover
        return f"Subject: {self.name}"
