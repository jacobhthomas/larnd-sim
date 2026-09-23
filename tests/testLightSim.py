#!/usr/bin/env python
"""REQUIRES A CUDA CAPABLE GPU

Tests for the FFT-based convolutions in the light simulation, comparing the GPU FFT result against a float64 CPU convolution.

Must reproduce:

    y[i] = sum_{j=max(i-K+1, 0)}^{i} h[i-j] * x[j],  K = min(len(h), conv_ticks+1, N)
    
"""

import numpy as np
import pytest

from larndsim import consts

consts.load_properties("larndsim/detector_properties/2x2.yaml",
                       "larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml",
                       "larndsim/bin/response_44_v2a_full.npz",
                       "larndsim/simulation_properties/2x2_NuMI_sim.yaml")

from larndsim.consts import light  # noqa: E402
from larndsim import light_sim  # noqa: E402

cp = pytest.importorskip("cupy")

#: Tolerance as a fraction of the largest reference value. Single-precision
#: FFTs of this length accumulate roughly 1e-6 relative error.
RTOL = 1e-4

NDET = 4
NTICKS = 512


@pytest.fixture(autouse=True)
def short_window(monkeypatch):
    """Use a short convolution window so the tests run quickly."""
    monkeypatch.setattr(light, "LIGHT_WINDOW", (0.0, 0.2))
    monkeypatch.setattr(light, "LIGHT_TICK_SIZE", 0.001)


def reference(signal, kernel, scale=None):
    """Direct causal convolution in float64, matching the original kernels."""
    signal = np.asarray(signal, dtype='f8')
    kernel = np.asarray(kernel, dtype='f8')
    conv_ticks = int(np.ceil((light.LIGHT_WINDOW[1] - light.LIGHT_WINDOW[0])
                             / light.LIGHT_TICK_SIZE))
    nticks = signal.shape[-1]
    klen = min(kernel.shape[0], conv_ticks + 1, nticks)
    out = np.empty_like(signal)
    for idet in range(signal.shape[0]):
        out[idet] = np.convolve(signal[idet], kernel[:klen])[:nticks]
    if scale is not None:
        out *= np.asarray(scale, dtype='f8').reshape(-1, 1)
    return out


def run_fft(signal, kernel, scale=None):
    """Run the FFT helper on the GPU and return the result as a numpy array."""
    sig = cp.asarray(signal, dtype='f4')
    out = cp.zeros_like(sig)
    light_sim._fft_convolve_time_axis(
        sig, np.asarray(kernel, dtype='f4'), out,
        None if scale is None else cp.asarray(scale, dtype='f4').reshape(-1, 1))
    return cp.asnumpy(out)


def assert_close(got, want, label=""):
    scale = max(np.abs(want).max(), 1e-12)
    err = np.abs(got - want).max() / scale
    assert err < RTOL, f"{label}: max relative error {err:.3e} exceeds {RTOL:.0e}"


def decaying_kernel(n=256, tau=40.0):
    return np.exp(-np.arange(n) / tau).astype('f4')


@pytest.mark.parametrize("position", [0, 1, NTICKS // 2, NTICKS - 1])
def test_impulse(position):
    """A single photon at one tick must reproduce a shifted copy of the kernel."""
    signal = np.zeros((NDET, NTICKS), dtype='f4')
    signal[:, position] = 1.0
    kernel = decaying_kernel()
    assert_close(run_fft(signal, kernel), reference(signal, kernel),
                 f"impulse at {position}")


def test_sparse_random():
    """Realistic case: photons arriving on a few ticks per channel."""
    rng = np.random.default_rng(0)
    signal = np.zeros((NDET, NTICKS), dtype='f4')
    for idet in range(NDET):
        ticks = rng.choice(NTICKS, size=20, replace=False)
        signal[idet, ticks] = rng.uniform(1, 100, size=ticks.size)
    kernel = decaying_kernel()
    assert_close(run_fft(signal, kernel), reference(signal, kernel), "sparse")


def test_dense_random():
    rng = np.random.default_rng(1)
    signal = rng.uniform(0, 10, size=(NDET, NTICKS)).astype('f4')
    kernel = decaying_kernel()
    assert_close(run_fft(signal, kernel), reference(signal, kernel), "dense")


def test_zero_signal_stays_zero():
    signal = np.zeros((NDET, NTICKS), dtype='f4')
    assert np.count_nonzero(run_fft(signal, decaying_kernel())) == 0


def test_signal_shorter_than_kernel():
    """Waveform shorter than the response kernel must still be truncated right."""
    signal = np.zeros((NDET, 64), dtype='f4')
    signal[:, 0] = 1.0
    kernel = decaying_kernel(n=256)
    assert_close(run_fft(signal, kernel), reference(signal, kernel), "short signal")


@pytest.mark.parametrize("nticks", [127, 128, 129])
def test_odd_and_even_lengths(nticks):
    rng = np.random.default_rng(2)
    signal = rng.uniform(0, 10, size=(NDET, nticks)).astype('f4')
    kernel = decaying_kernel(n=64)
    assert_close(run_fft(signal, kernel), reference(signal, kernel), f"n={nticks}")


def test_kernel_truncated_at_window():
    """Kernel samples beyond the convolution window must be ignored.

    The FFT leaves rounding residue (~1e-7) where the direct sum gave exact
    zeros, so compare against a threshold rather than testing for zero.
    """
    signal = np.zeros((NDET, NTICKS), dtype='f4')
    signal[:, 0] = 1.0
    kernel = np.ones(NTICKS, dtype='f4')          # longer than the window
    got = run_fft(signal, kernel)
    conv_ticks = int(np.ceil((light.LIGHT_WINDOW[1] - light.LIGHT_WINDOW[0])
                             / light.LIGHT_TICK_SIZE))
    significant = np.abs(got[0]) > 1e-3
    assert np.count_nonzero(significant) == conv_ticks + 1
    assert np.abs(got[0][conv_ticks + 1:]).max() < 1e-3


def test_output_accumulates():
    """The helper adds into its output array rather than overwriting it."""
    signal = np.zeros((NDET, NTICKS), dtype='f4')
    signal[:, 0] = 1.0
    kernel = decaying_kernel()
    sig = cp.asarray(signal)
    out = cp.full_like(sig, 5.0)
    light_sim._fft_convolve_time_axis(sig, kernel, out)
    assert_close(cp.asnumpy(out) - 5.0, reference(signal, kernel), "accumulate")


def test_per_channel_gain():
    """The SiPM step scales each channel by its own gain."""
    rng = np.random.default_rng(3)
    signal = rng.uniform(0, 10, size=(NDET, NTICKS)).astype('f4')
    kernel = decaying_kernel()
    gain = np.array([1.0, 2.5, 0.5, 10.0], dtype='f4')
    assert_close(run_fft(signal, kernel, gain),
                 reference(signal, kernel, gain), "gain")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))