"""
Audio Preprocessing Module for MedVision-AI.

Provides audio preprocessing for medical audio data including heart sounds,
lung auscultation, voice recordings, and other clinical audio signals.
Supports resampling, volume normalization, noise removal, feature extraction,
and spectrogram generation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Type alias
AudioType = Union[np.ndarray, Tuple[np.ndarray, int]]  # (samples,) or (samples, sample_rate)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AudioPreprocessConfig:
    """Configuration for the AudioPreprocessor.

    Attributes:
        target_sample_rate: Default resampling target in Hz.
        normalize_method: Volume normalization method (``"peak"`` or ``"rms"``).
        noise_reduction_method: Method for noise removal (``"spectral_gate"`` or ``"wiener"``).
        noise_reduction_strength: Strength factor for noise reduction [0, 1].
        spectrogram_n_fft: FFT window size for spectrogram.
        spectrogram_hop_length: Hop length for spectrogram.
        spectrogram_n_mels: Number of mel bands (for mel spectrogram).
        spectrogram_fmin: Minimum frequency for mel spectrogram.
        spectrogram_fmax: Maximum frequency for mel spectrogram (``None`` = sample_rate / 2).
        mfcc_n_mfcc: Number of MFCC coefficients to extract.
        pre_emphasis: Pre-emphasis filter coefficient.
        frame_length: Frame length in samples for framing operations.
        frame_step: Frame step in samples for framing operations.
        window_function: Window function name (``"hann"``, ``"hamming"``, ``"blackman"``).
    """

    target_sample_rate: int = 16000
    normalize_method: str = "peak"
    noise_reduction_method: str = "spectral_gate"
    noise_reduction_strength: float = 0.5
    spectrogram_n_fft: int = 512
    spectrogram_hop_length: int = 160
    spectrogram_n_mels: int = 64
    spectrogram_fmin: float = 0.0
    spectrogram_fmax: Optional[float] = None
    mfcc_n_mfcc: int = 13
    pre_emphasis: float = 0.97
    frame_length: int = 400
    frame_step: int = 160
    window_function: str = "hann"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_window(name: str, length: int) -> np.ndarray:
    """Return a window function of the given length.

    Args:
        name: Window function name.
        length: Number of samples.

    Returns:
        1-D numpy array.
    """
    if name == "hann":
        return np.hanning(length)
    elif name == "hamming":
        return np.hamming(length)
    elif name == "blackman":
        return np.blackman(length)
    else:
        logger.warning("Unknown window '%s', falling back to Hann", name)
        return np.hanning(length)


def _frame_signal(
    signal: np.ndarray,
    frame_length: int,
    frame_step: int,
    window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Split a 1-D signal into overlapping frames.

    Args:
        signal: 1-D numpy array.
        frame_length: Samples per frame.
        frame_step: Step between frame starts.
        window: Optional window to apply to each frame.

    Returns:
        2-D array of shape ``(n_frames, frame_length)``.
    """
    sig_len = len(signal)
    if sig_len < frame_length:
        # Pad to at least one frame
        padded = np.zeros(frame_length, dtype=signal.dtype)
        padded[:sig_len] = signal
        frames = padded[np.newaxis, :]
    else:
        n_frames = 1 + (sig_len - frame_length) // frame_step
        indices = (
            np.arange(frame_length)[np.newaxis, :]
            + np.arange(n_frames)[:, np.newaxis] * frame_step
        )
        frames = signal[indices]

    if window is not None:
        frames = frames * window[np.newaxis, :]

    return frames


def _mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Compute a mel-scale filter bank matrix.

    Args:
        n_mels: Number of mel bands.
        n_fft: FFT size.
        sample_rate: Audio sample rate.
        fmin: Minimum frequency.
        fmax: Maximum frequency (defaults to sample_rate / 2).

    Returns:
        Filter bank matrix of shape ``(n_mels, n_fft // 2 + 1)``.
    """
    fmax = fmax or sample_rate / 2.0

    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])

    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    n_bins = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_bins))

    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # Rising slope
        for j in range(left, center):
            if center > left and j < n_bins:
                filterbank[i, j] = (j - left) / (center - left)

        # Falling slope
        for j in range(center, right):
            if right > center and j < n_bins:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


def _dct_ii(x: np.ndarray, n_out: Optional[int] = None) -> np.ndarray:
    """Compute Type-II Discrete Cosine Transform.

    Args:
        x: Input array (1-D or 2-D).  If 2-D, DCT is applied along axis 1.
        n_out: Number of coefficients to keep.

    Returns:
        DCT coefficients.
    """
    n = x.shape[-1]
    n_out = n_out or n
    k = np.arange(n_out)[:, np.newaxis] if x.ndim == 1 else np.arange(n_out)[np.newaxis, :, np.newaxis]
    n_idx = np.arange(n)[np.newaxis, :] if x.ndim == 1 else np.arange(n)[np.newaxis, np.newaxis, :]

    if x.ndim == 1:
        cos_table = np.cos(np.pi * k * (2 * n_idx + 1) / (2 * n))
        return (x[np.newaxis, :] * cos_table).sum(axis=-1)
    else:
        cos_table = np.cos(np.pi * k * (2 * n_idx + 1) / (2 * n))
        return (x[:, np.newaxis, :] * cos_table).sum(axis=-1)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AudioPreprocessor:
    """Audio preprocessor for medical audio signals.

    Supports resampling, volume normalisation, noise removal, feature
    extraction (MFCCs, spectral features), and spectrogram generation.
    Designed for clinical audio such as heart sounds, lung auscultation,
    and patient voice recordings.

    Args:
        config: An :class:`AudioPreprocessConfig` instance.

    Example::

        preprocessor = AudioPreprocessor(AudioPreprocessConfig(target_sample_rate=16000))
        processed = preprocessor.preprocess(audio_array, sample_rate=44100)
    """

    def __init__(self, config: Optional[AudioPreprocessConfig] = None) -> None:
        self._config = config or AudioPreprocessConfig()
        logger.info(
            "AudioPreprocessor initialised (target_sr=%d, normalize=%s)",
            self._config.target_sample_rate,
            self._config.normalize_method,
        )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def preprocess(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Execute the full preprocessing pipeline on *audio*.

        Pipeline order:
        1. Resample to target sample rate
        2. Pre-emphasis filter
        3. Volume normalisation
        4. Noise reduction

        Args:
            audio: 1-D numpy array of audio samples.
            sample_rate: Original sample rate.

        Returns:
            Preprocessed audio array.
        """
        result = audio.astype(np.float64)

        # 1. Resample
        if sample_rate != self._config.target_sample_rate:
            result = self.resample(result, self._config.target_sample_rate, orig_sr=sample_rate)

        # 2. Pre-emphasis
        result = self._apply_preemphasis(result)

        # 3. Volume normalisation
        result = self.normalize_volume(result)

        # 4. Noise reduction
        result = self.remove_noise(result)

        logger.debug("Preprocessed audio: %d samples", len(result))
        return result.astype(np.float32)

    # ------------------------------------------------------------------
    # Individual operations
    # ------------------------------------------------------------------

    def resample(
        self,
        audio: np.ndarray,
        target_sr: int,
        orig_sr: Optional[int] = None,
    ) -> np.ndarray:
        """Resample *audio* to *target_sr* using linear interpolation.

        Args:
            audio: 1-D audio array.
            target_sr: Target sample rate in Hz.
            orig_sr: Original sample rate.  Defaults to config target.

        Returns:
            Resampled audio array.
        """
        orig = orig_sr or self._config.target_sample_rate
        if orig == target_sr:
            return audio.copy()

        if target_sr <= 0 or orig <= 0:
            raise ValueError(f"Invalid sample rates: orig={orig}, target={target_sr}")

        duration = len(audio) / orig
        n_target = int(duration * target_sr)

        # Linear interpolation resampling
        orig_indices = np.arange(len(audio), dtype=np.float64)
        target_indices = np.linspace(0, len(audio) - 1, n_target)

        resampled = np.interp(target_indices, orig_indices, audio.astype(np.float64))
        logger.debug("Resampled %d -> %d samples (%d Hz -> %d Hz)", len(audio), n_target, orig, target_sr)
        return resampled.astype(audio.dtype)

    def normalize_volume(
        self,
        audio: np.ndarray,
        method: Optional[str] = None,
        target_db: float = -3.0,
    ) -> np.ndarray:
        """Normalize the volume of *audio*.

        Args:
            audio: 1-D audio array.
            method: ``"peak"`` for peak normalisation or ``"rms"`` for
                RMS-based normalisation.  Defaults to config value.
            target_db: Target level in dB (used by ``"peak"`` method).

        Returns:
            Volume-normalised audio.
        """
        method = method or self._config.normalize_method
        audio_float = audio.astype(np.float64)

        if method == "peak":
            peak = np.abs(audio_float).max()
            if peak < 1e-10:
                logger.warning("Audio is silent; skipping peak normalization.")
                return audio.astype(np.float32)
            target_amplitude = 10.0 ** (target_db / 20.0)
            gain = target_amplitude / peak
            result = audio_float * gain

        elif method == "rms":
            rms = np.sqrt(np.mean(audio_float ** 2))
            if rms < 1e-10:
                logger.warning("Audio RMS is near-zero; skipping RMS normalization.")
                return audio.astype(np.float32)
            target_rms = 10.0 ** (target_db / 20.0)
            gain = target_rms / rms
            # Limit gain to avoid extreme amplification of quiet signals
            gain = min(gain, 100.0)
            result = audio_float * gain

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return result.astype(np.float32)

    def remove_noise(
        self,
        audio: np.ndarray,
        method: Optional[str] = None,
        strength: Optional[float] = None,
    ) -> np.ndarray:
        """Remove noise from *audio*.

        Args:
            audio: 1-D audio array.
            method: ``"spectral_gate"`` or ``"wiener"``. Defaults to config.
            strength: Noise reduction strength [0, 1]. Defaults to config.

        Returns:
            Denoised audio array.
        """
        method = method or self._config.noise_reduction_method
        s = strength if strength is not None else self._config.noise_reduction_strength

        if method == "spectral_gate":
            return self._spectral_gate(audio, s)
        elif method == "wiener":
            return self._wiener_filter(audio, s)
        else:
            raise ValueError(f"Unknown noise reduction method: {method}")

    def extract_features(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        features: Optional[Sequence[str]] = None,
    ) -> dict[str, np.ndarray]:
        """Extract audio features from *audio*.

        Supported features:
        - ``"mfcc"``: Mel-frequency cepstral coefficients.
        - ``"mel_spectrogram"``: Mel-scale spectrogram.
        - ``"spectral_centroid"``: Spectral centroid over time.
        - ``"zero_crossing_rate"``: Zero-crossing rate over time.
        - ``"rms_energy"``: Root-mean-square energy over time.

        Args:
            audio: 1-D audio array.
            sample_rate: Sample rate. Defaults to config target.
            features: List of feature names to extract. ``None`` extracts all.

        Returns:
            Dictionary mapping feature names to numpy arrays.
        """
        sr = sample_rate or self._config.target_sample_rate
        all_features = ["mfcc", "mel_spectrogram", "spectral_centroid", "zero_crossing_rate", "rms_energy"]
        target_features = features or all_features

        result: dict[str, np.ndarray] = {}

        for feat in target_features:
            if feat == "mfcc":
                result["mfcc"] = self._compute_mfcc(audio, sr)
            elif feat == "mel_spectrogram":
                result["mel_spectrogram"] = self.spectrogram(audio, sr, mel=True)
            elif feat == "spectral_centroid":
                result["spectral_centroid"] = self._compute_spectral_centroid(audio, sr)
            elif feat == "zero_crossing_rate":
                result["zero_crossing_rate"] = self._compute_zero_crossing_rate(audio)
            elif feat == "rms_energy":
                result["rms_energy"] = self._compute_rms_energy(audio)
            else:
                logger.warning("Unknown feature: %s", feat)

        return result

    def spectrogram(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        mel: bool = False,
    ) -> np.ndarray:
        """Compute the spectrogram of *audio*.

        Args:
            audio: 1-D audio array.
            sample_rate: Sample rate. Defaults to config target.
            mel: If ``True``, return a mel-scale spectrogram.

        Returns:
            2-D numpy array of shape ``(n_freq_bins, n_time_frames)``.
        """
        sr = sample_rate or self._config.target_sample_rate
        n_fft = self._config.spectrogram_n_fft
        hop = self._config.spectrogram_hop_length

        # Frame the signal
        window = _get_window(self._config.window_function, n_fft)
        frames = _frame_signal(audio, n_fft, hop, window=window)

        # FFT
        fft_result = np.fft.rfft(frames, axis=1)
        power_spectrum = np.abs(fft_result) ** 2 / n_fft

        if mel:
            n_mels = self._config.spectrogram_n_mels
            fmin = self._config.spectrogram_fmin
            fmax = self._config.spectrogram_fmax or sr / 2.0
            fb = _mel_filterbank(n_mels, n_fft, sr, fmin, fmax)
            mel_spec = np.dot(power_spectrum, fb.T)
            # Convert to dB
            mel_spec_db = 10.0 * np.log10(mel_spec + 1e-10)
            return mel_spec_db.T  # (n_mels, n_frames)

        # Convert to dB
        spec_db = 10.0 * np.log10(power_spectrum + 1e-10)
        return spec_db.T  # (n_freq_bins, n_frames)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _apply_preemphasis(self, audio: np.ndarray) -> np.ndarray:
        """Apply a pre-emphasis filter: y[n] = x[n] - α * x[n-1]."""
        alpha = self._config.pre_emphasis
        if alpha == 0:
            return audio
        result = np.zeros_like(audio)
        result[0] = audio[0]
        result[1:] = audio[1:] - alpha * audio[:-1]
        return result

    def _spectral_gate(self, audio: np.ndarray, strength: float) -> np.ndarray:
        """Apply spectral gating noise reduction.

        Estimates noise from the quietest portion of the signal and
        subtracts it from the spectrum.

        Args:
            audio: Input audio.
            strength: Threshold multiplier [0, 1].

        Returns:
            Denoised audio.
        """
        n_fft = self._config.spectrogram_n_fft
        hop = self._config.spectrogram_hop_length
        window = _get_window(self._config.window_function, n_fft)

        # STFT
        frames = _frame_signal(audio, n_fft, hop, window=window)
        stft = np.fft.rfft(frames, axis=1)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Estimate noise from first ~0.5 seconds (assumed to be noise-only)
        noise_frames = min(frames.shape[0], max(1, int(0.5 * self._config.target_sample_rate / hop)))
        noise_spectrum = magnitude[:noise_frames].mean(axis=0)

        # Gate: suppress bins below threshold
        threshold = noise_spectrum * (1.0 + strength * 5.0)
        mask = magnitude > threshold[np.newaxis, :]
        cleaned_magnitude = magnitude * mask

        # Reconstruct
        cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
        cleaned_frames = np.fft.irfft(cleaned_stft, n=n_fft, axis=1)

        # Overlap-add synthesis
        output_len = (cleaned_frames.shape[0] - 1) * hop + n_fft
        output = np.zeros(output_len, dtype=np.float64)
        window_sum = np.zeros(output_len, dtype=np.float64)

        for i in range(cleaned_frames.shape[0]):
            start = i * hop
            output[start : start + n_fft] += cleaned_frames[i] * window
            window_sum[start : start + n_fft] += window ** 2

        # Normalise by window sum
        nonzero = window_sum > 1e-8
        output[nonzero] /= window_sum[nonzero]

        # Trim to original length
        output = output[: len(audio)]
        return output.astype(np.float32)

    def _wiener_filter(self, audio: np.ndarray, strength: float) -> np.ndarray:
        """Apply a simple Wiener-style filter for noise reduction.

        Args:
            audio: Input audio.
            strength: Smoothing factor [0, 1].

        Returns:
            Denoised audio.
        """
        n_fft = self._config.spectrogram_n_fft
        hop = self._config.spectrogram_hop_length
        window = _get_window(self._config.window_function, n_fft)

        frames = _frame_signal(audio, n_fft, hop, window=window)
        stft = np.fft.rfft(frames, axis=1)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Estimate noise power from quietest frames
        frame_energies = (magnitude ** 2).sum(axis=1)
        n_noise = max(1, int(len(frame_energies) * 0.1))
        quiet_indices = np.argsort(frame_energies)[:n_noise]
        noise_power = (magnitude[quiet_indices] ** 2).mean(axis=0)

        # Wiener gain
        signal_power = magnitude ** 2
        gain = np.maximum(signal_power - noise_power[np.newaxis, :] * (1 + strength), 0) / (
            signal_power + 1e-10
        )
        gain = np.sqrt(gain)

        # Apply gain
        cleaned_magnitude = magnitude * gain
        cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
        cleaned_frames = np.fft.irfft(cleaned_stft, n=n_fft, axis=1)

        # Overlap-add
        output_len = (cleaned_frames.shape[0] - 1) * hop + n_fft
        output = np.zeros(output_len, dtype=np.float64)
        window_sum = np.zeros(output_len, dtype=np.float64)

        for i in range(cleaned_frames.shape[0]):
            start = i * hop
            output[start : start + n_fft] += cleaned_frames[i] * window
            window_sum[start : start + n_fft] += window ** 2

        nonzero = window_sum > 1e-8
        output[nonzero] /= window_sum[nonzero]
        output = output[: len(audio)]
        return output.astype(np.float32)

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_mfcc(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Compute Mel-Frequency Cepstral Coefficients (MFCCs).

        Args:
            audio: 1-D audio array.
            sample_rate: Audio sample rate.

        Returns:
            MFCC array of shape ``(n_mfcc, n_frames)``.
        """
        n_fft = self._config.spectrogram_n_fft
        hop = self._config.spectrogram_hop_length
        n_mels = self._config.spectrogram_n_mels
        n_mfcc = self._config.mfcc_n_mfcc
        fmin = self._config.spectrogram_fmin
        fmax = self._config.spectrogram_fmax or sample_rate / 2.0

        window = _get_window(self._config.window_function, n_fft)
        frames = _frame_signal(audio, n_fft, hop, window=window)

        # Power spectrum
        fft_result = np.fft.rfft(frames, axis=1)
        power_spectrum = np.abs(fft_result) ** 2 / n_fft

        # Mel filterbank
        fb = _mel_filterbank(n_mels, n_fft, sample_rate, fmin, fmax)
        mel_spec = np.dot(power_spectrum, fb.T)

        # Log mel spectrum
        log_mel = np.log(mel_spec + 1e-10)

        # DCT
        mfcc = _dct_ii(log_mel, n_out=n_mfcc)

        return mfcc.T  # (n_mfcc, n_frames)

    def _compute_spectral_centroid(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Compute spectral centroid over time.

        Args:
            audio: 1-D audio array.
            sample_rate: Audio sample rate.

        Returns:
            1-D array of spectral centroid values per frame.
        """
        n_fft = self._config.spectrogram_n_fft
        hop = self._config.spectrogram_hop_length
        window = _get_window(self._config.window_function, n_fft)
        frames = _frame_signal(audio, n_fft, hop, window=window)

        fft_result = np.fft.rfft(frames, axis=1)
        magnitude = np.abs(fft_result)

        freqs = np.linspace(0, sample_rate / 2, magnitude.shape[1])
        centroid = (magnitude * freqs[np.newaxis, :]).sum(axis=1) / (magnitude.sum(axis=1) + 1e-10)

        return centroid

    def _compute_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """Compute zero-crossing rate over time.

        Args:
            audio: 1-D audio array.

        Returns:
            1-D array of ZCR values per frame.
        """
        frame_len = self._config.frame_length
        frame_step = self._config.frame_step

        signs = np.sign(audio)
        signs[signs == 0] = 1  # treat zero as positive
        crossings = np.abs(np.diff(signs)) / 2.0

        n_frames = 1 + (len(crossings) - frame_len) // frame_step
        zcr = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * frame_step
            zcr[i] = crossings[start : start + frame_len].mean()

        return zcr

    def _compute_rms_energy(self, audio: np.ndarray) -> np.ndarray:
        """Compute root-mean-square energy over time.

        Args:
            audio: 1-D audio array.

        Returns:
            1-D array of RMS values per frame.
        """
        frame_len = self._config.frame_length
        frame_step = self._config.frame_step
        frames = _frame_signal(audio, frame_len, frame_step)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        return rms
