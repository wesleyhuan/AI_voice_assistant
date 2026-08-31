import pytest
from audio import record_audio, record_chunk, MAX_DURATION, MAX_SAMPLE_RATE

def test_record_audio_limits():
    # Lower bound tests
    with pytest.raises(ValueError):
        record_audio(duration=0, sample_rate=16000)
    with pytest.raises(ValueError):
        record_audio(duration=-1, sample_rate=16000)
    with pytest.raises(ValueError):
        record_audio(duration=5, sample_rate=0)
    with pytest.raises(ValueError):
        record_audio(duration=5, sample_rate=-16000)

    # Upper bound tests
    with pytest.raises(ValueError):
        record_audio(duration=MAX_DURATION + 1, sample_rate=16000)
    with pytest.raises(ValueError):
        record_audio(duration=5, sample_rate=MAX_SAMPLE_RATE + 1)

def test_record_chunk_limits():
    # Lower bound tests
    with pytest.raises(ValueError):
        record_chunk(duration=0, sample_rate=16000)
    with pytest.raises(ValueError):
        record_chunk(duration=-1, sample_rate=16000)
    with pytest.raises(ValueError):
        record_chunk(duration=5, sample_rate=0)
    with pytest.raises(ValueError):
        record_chunk(duration=5, sample_rate=-16000)

    # Upper bound tests
    with pytest.raises(ValueError):
        record_chunk(duration=MAX_DURATION + 1, sample_rate=16000)
    with pytest.raises(ValueError):
        record_chunk(duration=5, sample_rate=MAX_SAMPLE_RATE + 1)
