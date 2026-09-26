"""Spectral-flux onset detection for edit-point localization.

Each channel is analysed with a centred 1024-point Hann window at a
256-sample hop (zero-padded on both ends, real FFT): frame k is centred at
sample k*256 and covers [k*256 - 512, k*256 + 512). The per-frame spectral
flux is the sum over bins of the positive magnitude difference between the
current and the previous frame; for stereo the larger of the two channel
fluxes is used. A frame becomes an onset candidate when its flux

* is strictly greater than the median of the 20 previous and 20 following
  frames' fluxes plus 3 times the MAD of that neighbourhood,
* reaches ``min_relative_flux`` times the clip-wide maximum flux, and
* is a local peak (on a plateau of equal fluxes the earliest frame wins).

Candidates are then greedily selected in descending flux order (ties by
ascending time), keeping only frames at least 80 ms (15 hops of 256
samples) away from every already-selected one. The result is reported in
ascending time order; each onset carries the window-centre sample index
clamped to the audio bounds and the flux as a fraction of the clip
maximum. The audio itself is never modified; an all-silent clip yields an
empty list.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .loudness import _read_wav

N_FFT = 1024
HOP = 256
# Centred window: frame k covers samples [k*HOP - N_FFT//2, k*HOP + N_FFT//2).
LEAD_PAD = N_FFT // 2
NEIGHBOURHOOD_FRAMES = 20  # adaptive threshold: +-20 frames around the peak
MAD_SCALE = 3.0
# 80 ms at 48 kHz = 3840 samples = 15 hops of 256 samples.
MIN_SEPARATION_FRAMES = 15

# Periodic Hann, same shape as the denoise STFT window.
WINDOW = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)


@dataclass
class Onset:
    frame_index: int
    strength: float


@dataclass
class OnsetsResult:
    onsets: list[Onset]


def _stft_magnitudes(channel: np.ndarray) -> np.ndarray:
    """RFFT magnitudes (n_frames, N_FFT // 2 + 1) of the centred frames."""
    n = channel.shape[0]
    # Frame k starts at k*HOP - LEAD_PAD in input coordinates; the last
    # frame starts at floor(n / HOP) * HOP so every sample is covered.
    n_frames = n // HOP + 1
    padded = np.zeros((n_frames - 1) * HOP + N_FFT, dtype=np.float64)
    padded[LEAD_PAD : LEAD_PAD + n] = channel
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP]
    return np.abs(np.fft.rfft(frames * WINDOW, axis=1))


def _spectral_flux(magnitudes: np.ndarray) -> np.ndarray:
    """Sum of positive bin-wise magnitude differences vs. the previous frame."""
    flux = np.zeros(magnitudes.shape[0], dtype=np.float64)
    flux[1:] = np.maximum(magnitudes[1:] - magnitudes[:-1], 0.0).sum(axis=1)
    return flux


def _is_candidate(flux: np.ndarray, k: int, threshold: float) -> bool:
    """Adaptive-threshold and local-peak tests for frame k."""
    value = flux[k]
    if value < threshold:
        return False
    # Local peak; on a plateau of equal fluxes the earliest frame wins.
    if k > 0 and not value > flux[k - 1]:
        return False
    if k + 1 < flux.shape[0] and not value >= flux[k + 1]:
        return False
    lo = max(0, k - NEIGHBOURHOOD_FRAMES)
    hi = min(flux.shape[0], k + NEIGHBOURHOOD_FRAMES + 1)
    neighbourhood = np.concatenate((flux[lo:k], flux[k + 1 : hi]))
    median = float(np.median(neighbourhood))
    mad = float(np.median(np.abs(neighbourhood - median)))
    return bool(value > median + MAD_SCALE * mad)


def detect_onsets(wav_bytes: bytes, min_relative_flux: float) -> OnsetsResult:
    samples = _read_wav(wav_bytes)
    n_samples, channels = samples.shape

    flux = np.zeros(n_samples // HOP + 1, dtype=np.float64)
    for ch in range(channels):
        channel_flux = _spectral_flux(_stft_magnitudes(samples[:, ch]))
        flux = np.maximum(flux, channel_flux)

    max_flux = float(flux.max())
    if max_flux <= 0.0:
        # All-silent (or constant) audio has no transients.
        return OnsetsResult(onsets=[])

    threshold = min_relative_flux * max_flux
    candidates = [
        k for k in range(flux.shape[0]) if _is_candidate(flux, k, threshold)
    ]
    # Strongest first, earliest on ties; keep onsets >= 80 ms apart.
    candidates.sort(key=lambda k: (-flux[k], k))
    selected: list[int] = []
    for k in candidates:
        if all(abs(k - s) >= MIN_SEPARATION_FRAMES for s in selected):
            selected.append(k)
    selected.sort()

    return OnsetsResult(
        onsets=[
            Onset(
                frame_index=min(k * HOP, n_samples - 1),
                strength=flux[k] / max_flux,
            )
            for k in selected
        ]
    )
