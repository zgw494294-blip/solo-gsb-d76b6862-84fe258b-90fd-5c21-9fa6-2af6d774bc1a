import base64
import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

RATE = 48000
N = 2 * RATE  # 2 s of audio
HOP = 256
MIN_GAP_SAMPLES = 3840  # 80 ms


def wav_b64(pcm: np.ndarray, channels: int = 1, rate: int = RATE) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.astype("<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def make_clicks(positions, amps, n=N):
    """Single-sample impulses (broadband transients) at the given indices."""
    sig = np.zeros(n)
    for pos, amp in zip(positions, amps):
        sig[pos] = amp
    return np.round(sig * 32768).astype("<i2")


def make_sine_bursts(starts, n=N, amp=0.5, freq=440.0, burst_len=2400):
    """50 ms sine bursts starting at the given indices, silence elsewhere."""
    sig = np.zeros(n)
    t = np.arange(burst_len) / RATE
    for start in starts:
        sig[start : start + burst_len] = amp * np.sin(2 * np.pi * freq * t)
    return np.round(sig * 32768).astype("<i2")


def post_onsets(pcm, channels=1, rate=RATE, **kwargs):
    payload = {"wav_base64": wav_b64(pcm, channels, rate)}
    payload.update(kwargs)
    return client.post("/v1/onsets", json=payload)


def post_onsets_raw(pcm, **kwargs):
    # Raw body so non-finite floats (NaN/Infinity, allowed by Python's json
    # encoder and decoder) actually reach the server.
    payload = {"wav_base64": wav_b64(pcm)}
    payload.update(kwargs)
    return client.post(
        "/v1/onsets",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )


def test_detects_clicks_at_known_positions():
    positions = [24000, 48000, 72000]  # 0.5 s, 1.0 s, 1.5 s
    resp = post_onsets(make_clicks(positions, [0.8] * 3))
    assert resp.status_code == 200
    onsets = resp.json()["onsets"]
    assert len(onsets) == 3
    for onset, pos in zip(onsets, positions):
        assert abs(onset["frame_index"] - pos) <= 512
        assert 0.0 < onset["strength"] <= 1.0
    # The strongest onset is normalized to exactly 1.
    assert max(o["strength"] for o in onsets) == 1.0


def test_detects_sine_bursts():
    starts = [24000, 48000, 72000]
    resp = post_onsets(make_sine_bursts(starts))
    assert resp.status_code == 200
    onsets = resp.json()["onsets"]
    assert len(onsets) == 3
    for onset, start in zip(onsets, starts):
        assert abs(onset["frame_index"] - start) <= 1024


def test_min_relative_flux_defaults_to_01():
    # A click at 1/4 of the peak amplitude passes the default 0.1 gate.
    pcm = make_clicks([48000, 90000], [0.8, 0.2])
    resp = post_onsets(pcm)
    assert resp.status_code == 200
    assert len(resp.json()["onsets"]) == 2


def test_min_relative_flux_filters_weaker_onsets():
    pcm = make_clicks([48000, 90000], [0.8, 0.2])  # weak click ~0.25x flux
    resp = post_onsets(pcm, min_relative_flux=0.5)
    assert resp.status_code == 200
    onsets = resp.json()["onsets"]
    assert len(onsets) == 1
    assert abs(onsets[0]["frame_index"] - 48000) <= 512
    assert onsets[0]["strength"] == 1.0


def test_min_relative_flux_one_keeps_only_strongest():
    pcm = make_clicks([48000, 90000], [0.8, 0.2])
    resp = post_onsets(pcm, min_relative_flux=1.0)
    assert resp.status_code == 200
    onsets = resp.json()["onsets"]
    assert len(onsets) == 1
    assert abs(onsets[0]["frame_index"] - 48000) <= 512
    assert onsets[0]["strength"] == 1.0


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_min_relative_flux_bounds_accepted(value):
    resp = post_onsets(make_clicks([48000], [0.8]), min_relative_flux=value)
    assert resp.status_code == 200


def test_onsets_closer_than_80ms_collapse_to_stronger():
    # Two clicks 40 ms apart: only the stronger (first) one is reported.
    resp = post_onsets(make_clicks([48000, 49920], [0.8, 0.5]))
    assert resp.status_code == 200
    onsets = resp.json()["onsets"]
    assert len(onsets) == 1
    assert abs(onsets[0]["frame_index"] - 48000) <= 512


def test_stronger_click_wins_when_too_close():
    # Same 40 ms spacing, but the second click is the stronger one.
    resp = post_onsets(make_clicks([48000, 49920], [0.5, 0.8]))
    assert resp.status_code == 200
    onsets = resp.json()["onsets"]
    assert len(onsets) == 1
    assert abs(onsets[0]["frame_index"] - 49920) <= 512


def test_onsets_exactly_80ms_apart_are_both_kept():
    resp = post_onsets(make_clicks([48000, 51840], [0.8, 0.5]))
    assert resp.status_code == 200
    onsets = resp.json()["onsets"]
    assert len(onsets) == 2
    gap = onsets[1]["frame_index"] - onsets[0]["frame_index"]
    assert gap >= MIN_GAP_SAMPLES


def test_selected_onsets_respect_minimum_spacing():
    # A click every 30 ms: detected onsets must still be >= 80 ms apart.
    positions = list(range(24000, 72000, 1440))
    resp = post_onsets(make_clicks(positions, [0.8] * len(positions)))
    assert resp.status_code == 200
    indices = [o["frame_index"] for o in resp.json()["onsets"]]
    assert len(indices) >= 2
    assert indices == sorted(indices)
    for earlier, later in zip(indices, indices[1:]):
        assert later - earlier >= MIN_GAP_SAMPLES


def test_stereo_uses_louder_channel():
    # Click on the right channel only; the left channel is silent.
    pcm = np.zeros((N, 2))
    pcm[48000, 1] = 0.8
    pcm = np.round(pcm * 32768).astype("<i2")
    resp = post_onsets(pcm, channels=2)
    assert resp.status_code == 200
    onsets = resp.json()["onsets"]
    assert len(onsets) == 1
    assert abs(onsets[0]["frame_index"] - 48000) <= 512


def test_stereo_detects_onsets_from_either_channel():
    pcm = np.zeros((N, 2))
    pcm[48000, 0] = 0.8  # left-only click at 1.0 s
    pcm[90000, 1] = 0.8  # right-only click at 1.875 s
    pcm = np.round(pcm * 32768).astype("<i2")
    resp = post_onsets(pcm, channels=2)
    assert resp.status_code == 200
    onsets = resp.json()["onsets"]
    assert len(onsets) == 2
    assert abs(onsets[0]["frame_index"] - 48000) <= 512
    assert abs(onsets[1]["frame_index"] - 90000) <= 512


def test_response_sorted_and_frame_indices_in_bounds():
    positions = [72000, 24000, 90000, 48000]  # unordered transients
    resp = post_onsets(make_clicks(positions, [0.8] * 4))
    assert resp.status_code == 200
    onsets = resp.json()["onsets"]
    indices = [o["frame_index"] for o in onsets]
    assert indices == sorted(indices)
    for index in indices:
        assert 0 <= index <= N - 1
    for onset in onsets:
        assert 0.0 < onset["strength"] <= 1.0


def test_silence_returns_empty_list():
    resp = post_onsets(np.zeros(N, dtype="<i2"))
    assert resp.status_code == 200
    assert resp.json()["onsets"] == []


def test_invalid_base64_returns_422():
    resp = client.post(
        "/v1/onsets",
        json={"wav_base64": "!!!not-base64!!!"},
    )
    assert resp.status_code == 422


def test_garbage_wav_returns_422():
    resp = client.post(
        "/v1/onsets",
        json={"wav_base64": base64.b64encode(b"not a wav").decode()},
    )
    assert resp.status_code == 422


def test_non_compliant_wav_returns_422():
    pcm = make_clicks([48000], [0.8])
    assert post_onsets(pcm, rate=44100).status_code == 422


def test_too_short_wav_returns_422():
    pcm = make_clicks([12000], [0.8], n=RATE // 2)  # 0.5 s
    assert post_onsets(pcm).status_code == 422


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_min_relative_flux_out_of_range_returns_422(value):
    pcm = make_clicks([48000], [0.8])
    assert post_onsets(pcm, min_relative_flux=value).status_code == 422


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_min_relative_flux_non_finite_returns_422(bad):
    pcm = make_clicks([48000], [0.8])
    assert post_onsets_raw(pcm, min_relative_flux=bad).status_code == 422


def test_min_relative_flux_wrong_type_returns_422():
    pcm = make_clicks([48000], [0.8])
    assert post_onsets(pcm, min_relative_flux="high").status_code == 422


def test_missing_wav_base64_returns_422():
    assert client.post("/v1/onsets", json={}).status_code == 422
