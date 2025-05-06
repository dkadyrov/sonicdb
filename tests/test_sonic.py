import os
from datetime import timedelta

from sonicdb import audio, models, sonic


def test_database():
    if os.path.exists("examples/example.db"):
        os.remove("examples/example.db")

    db = sonic.Database("examples/example.db")
    sonic.database_exists(db.engine.url)

    sensor = models.Sensor(name="test_sensor")
    db.session.add(sensor)

    channel = models.Channel(sensor=sensor, number=0)
    db.session.add(channel)

    f = audio.Audio("examples/sonic.wav")

    file = models.File(
        filepath=f.filename,
        filename=f.filename,
        extension=f.extension,
        sample_rate=f.sample_rate,
        start=f.start,
        end=f.end,
        duration=f.duration,
        channel=channel,
        sensor=sensor,
        channel_number=0,
    )
    db.session.add(file)

    subject = models.Subject(name="SONIC audio")
    db.session.add(subject)
    db.session.commit()

    event = models.Event(
        name="SONIC waveform",
        description="A waveform that displays SONIC when viewed as a spectrogram",
        start=file.start,
        end=file.end,
    )
    db.session.add(event)
    db.session.commit()

    for i in range(5):
        sample = models.Sample(
            datetime=file.start + timedelta(seconds=float(i * file.duration / 5)),
            sensor=sensor,
            channel=channel,
            subject=subject,
            file=file,
        )
        db.session.add(sample)
        db.session.commit()

    db.session.commit()
    db.session.close()

    file = db.session.query(models.File).first()
    channel = db.session.query(models.Channel).first()

    start = channel.get_start()
    end = channel.get_end()

    c = db.get_channel(channel)

    assert start == file.start
    assert end == file.end
    assert c == channel

    sensor = db.session.query(models.Sensor).first()
    s = db.get_sensor(sensor)

    assert s == sensor

    subject = db.session.query(models.Subject).first()
    s = db.get_subject(subject)

    assert s == subject
    db.session.close()

    assert db.session.query(models.Sample).count() == 5

    file = db.session.query(models.File).first()
    file_audio = file.get_audio()

    a2 = db.get_audio(start=file.start, end=file.end, channel=file.channel)

    assert (f.data.signal == file_audio.data.signal).all()
    assert (a2.data.signal == file_audio.data.signal).all()

    db.session.close()
    db.engine.dispose()

    os.remove("examples/example.db")


test_database()
