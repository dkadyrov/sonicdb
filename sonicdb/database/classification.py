from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Float
from sqlalchemy.orm import relationship

from .base import Base


class Classification(Base):  # type: ignore
    """Classification object"""

    __tablename__ = "classification"
    id = Column(Integer, primary_key=True)
    """int: Classification database ID"""

    datetime = Column(DateTime)
    """datetime: Datetime of classification"""

    classifier = Column(String)
    """int: classifier identification. Can reference id in an external lookup table """
    # TODO maybe have a table for different classifiers

    classification = Column(Float)
    """int: classification encoding. Accessed using :func:`~starpy.sad.database.Classification.class_encoding`"""
    # TODO maybe have a table for different classification encodings

    sensor_id = Column(Integer, ForeignKey("sensor.id"))
    """int: Sensor database ID"""

    sensor = relationship(
        "sonicdb.database.sensor.Sensor",
        back_populates="classifications",
        enable_typechecks=False,
    )
    """Sensor: Sensor object"""

    sample_id = Column(Integer, ForeignKey("sample.id"))
    """int: Sample database ID"""

    sample = relationship(
        "sonicdb.database.sample.Sample",
        back_populates="classifications",
        enable_typechecks=False,
    )
    """Sample: Sample object"""

    __mapper_args__ = {
        "polymorphic_identity": "classification",
    }

    def __repr__(self) -> str:  # pragma: no cover
        """Returns object representation"""

        return f"Classifier: {self.classifier} from Sensor {self.sensor}"
