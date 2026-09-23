#!/usr/bin/env python
"""REQUIRES A CUDA CAPABLE GPU

Tests for the float64 scintillation FFT (LARNDSIM_LIGHT_CONV=fft64).

The candidate computes the scintillation convolution with a float64 FFT, casts
it back to float32, and sets every tick outside the physical support to exactly
zero. The statistical-fluctuation sampler is unchanged. These tests check:

* support construction: empty channels, delayed first light, gaps longer than
  the convolution window, truncation at the final tick;
* accuracy against a float64 direct-convolution reference, with the tolerance
  defined in `assert_matches_reference`;
* RNG regression: with identical incoming RNG states, sampled PE and outgoing
  states match the original float32 direct sum on fixtures taken from real
  simulation batches (float32 FFT residue before first light, and faint
  supported ticks that the float32 FFT pushed to <= 0);
* one constructed mean-30 sampling-boundary case;
* supported nonpositive outputs are reported, not corrected.

The fixtures come from simulation batches where the float32 FFT broke RNG
alignment. How the float32 FFT behaves on them is recorded as diagnostic
evidence in the fixture file (`float32_fft_observed_*`) but is not asserted:
float32 rounding can differ between GPUs, cuFFT versions and execution
configurations.
"""

import os
import warnings
from math import ceil

import numpy as np
import pytest

from larndsim import consts

consts.load_properties("larndsim/detector_properties/2x2.yaml",
                       "larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml",
                       "larndsim/bin/response_44_v2a_full.npz",
                       "larndsim/simulation_properties/2x2_NuMI_sim.yaml")

from larndsim.consts import light, sim  # noqa: E402
from larndsim import light_sim  # noqa: E402

cp = pytest.importorskip("cupy")
from numba import cuda  # noqa: E402
import numba.cuda.random  # noqa: E402,F401  (registers cuda.random for calc_stat_fluctuations)
from numba.cuda.random import create_xoroshiro128p_states  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "data", "light_fft64_fixtures.npz")
TPB = (1, 64)
SEED = 321


# ---------------------------------------------------------------- helpers ---

def scint_kernel(nticks):
    kernel = np.zeros(nticks, dtype=np.float32)
    light_sim.scintillation_array(kernel)
    return kernel


def kernel_length(kernel, nticks):
    return min(light_sim._convolution_kernel_length(kernel), nticks)


def reference64(signal, kernel):
    """Float64 direct causal convolution (itself approximate, but ~1e-16 relative)."""
    signal = np.asarray(signal, dtype='f8')
    nticks = signal.shape[-1]
    k = np.asarray(kernel[:kernel_length(kernel, nticks)], dtype='f8')
    return np.stack([np.convolve(row, k)[:nticks] for row in signal])


def run_fft64(signal, kernel):
    sig = cp.asarray(signal, dtype='f4')
    out = cp.zeros_like(sig)
    light_sim._fft64_scintillation(sig, kernel, out)
    return out


def run_fft32(signal, kernel):
    sig = cp.asarray(signal, dtype='f4')
    out = cp.zeros_like(sig)
    light_sim._fft_convolve_time_axis(sig, kernel, out)
    return out


@cuda.jit
def _direct32_kernel(light_sample_inc, light_sample_inc_scint, scint_model, conv_ticks):
    """Signal part of the original float32 direct-sum scintillation kernel (pre-a178cf4)."""
    idet, itick = cuda.grid(2)
    if idet < light_sample_inc.shape[0]:
        if itick < light_sample_inc.shape[1]:
            for jtick in range(max(itick - conv_ticks, 0), itick + 1):
                if light_sample_inc[idet, jtick] == 0:
                    continue
                tick_weight = scint_model[itick - jtick]
                light_sample_inc_scint[idet, itick] += tick_weight * light_sample_inc[idet, jtick]


def run_direct32(signal, kernel):
    sig = cp.asarray(signal, dtype='f4')
    out = cp.zeros_like(sig)
    conv_ticks = ceil((light.LIGHT_WINDOW[1] - light.LIGHT_WINDOW[0]) / light.LIGHT_TICK_SIZE)
    _direct32_kernel[grid(sig), TPB](sig, out, kernel, conv_ticks)
    cuda.synchronize()
    return out


def grid(arr):
    return (arr.shape[0], ceil(arr.shape[1] / TPB[1]))


def fresh_states(arr, seed=SEED):
    bpg = grid(arr)
    return create_xoroshiro128p_states(bpg[0] * bpg[1] * TPB[0] * TPB[1], seed=seed).copy_to_host()


def sample(scint, states_in):
    """Unmodified sampler on a copy of `states_in`; returns (PE counts, outgoing states)."""
    states = cuda.to_device(states_in.copy())
    disc = cp.zeros_like(scint)
    light_sim.calc_stat_fluctuations[grid(scint), TPB](scint, disc, states)
    cuda.synchronize()
    pe = np.rint(cp.asnumpy(disc).astype('f8') * light.LIGHT_TICK_SIZE).astype(np.int64)
    return pe, states.copy_to_host()


def brute_force_support(signal, klen):
    nonzero = np.asarray(signal) != 0
    out = np.zeros_like(nonzero)
    for i in range(nonzero.shape[-1]):
        out[..., i] = nonzero[..., max(i - klen + 1, 0):i + 1].any(axis=-1)
    return out


def assert_matches_reference(got, ref, label):
    """Elementwise tolerance for float64 FFT -> float32 cast.

    |got - ref| <= spacing_f32(|ref|) + 1e-12 * max|ref| (per channel), where
    spacing_f32 is one float32 ulp at the reference value (the cast contributes
    at most half an ulp) and the second term allows for float64 FFT round-off
    (~1e4 x float64 epsilon relative to the channel maximum). Supported ticks
    with a positive reference must also stay positive.
    """
    got = np.asarray(got, dtype='f8')
    ref = np.asarray(ref, dtype='f8')
    scale = np.abs(ref).max(axis=-1, keepdims=True)
    tol = np.spacing(np.abs(ref).astype(np.float32)).astype('f8') + 1e-12 * scale
    bad = np.abs(got - ref) > tol
    assert not bad.any(), (f"{label}: {bad.sum()} ticks exceed tolerance; worst "
                           f"|err|/tol = {(np.abs(got - ref) / tol).max():.3g}")
    assert not ((ref > 0) & (got <= 0)).any(), f"{label}: positive reference ticks became <= 0"


def load_fixture(name):
    d = np.load(FIXTURES)
    rows = np.zeros(tuple(d[f'{name}_shape']), dtype='f4')
    rows[d[f'{name}_row'], d[f'{name}_tick']] = d[f'{name}_value']
    return rows


@pytest.fixture
def short_window(monkeypatch):
    """0.2 us window at 1 ns ticks: retained kernel length 201 ticks."""
    monkeypatch.setattr(light, "LIGHT_WINDOW", (0.0, 0.2))
    monkeypatch.setattr(light, "LIGHT_TICK_SIZE", 0.001)


@pytest.fixture
def no_warnings():
    """Fail if the fft64 nonpositive-support diagnostic fires."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="fft64 scintillation")
        yield


# ----------------------------------------------------- support correctness ---

NT = 512


def test_support_matches_definition(short_window):
    rng = np.random.default_rng(0)
    signal = np.zeros((6, NT), dtype='f4')
    for row, ticks in enumerate([[], [0], [NT - 1], [30, 300], [5, 6, 7, 400], rng.choice(NT, 15, replace=False)]):
        signal[row, ticks] = rng.uniform(0.5, 50, size=len(ticks))
    kernel = scint_kernel(NT)
    got = cp.asnumpy(light_sim.scintillation_support(cp.asarray(signal), kernel))
    assert np.array_equal(got, brute_force_support(signal, kernel_length(kernel, NT)))


def test_empty_channel_is_exactly_zero(short_window, no_warnings):
    signal = np.zeros((3, NT), dtype='f4')
    signal[1, 100] = 1000.
    out = cp.asnumpy(run_fft64(signal, scint_kernel(NT)))
    assert np.count_nonzero(out[0]) == 0 and np.count_nonzero(out[2]) == 0
    assert np.count_nonzero(out[1]) > 0


def test_delayed_first_light(short_window, no_warnings):
    signal = np.zeros((2, NT), dtype='f4')
    signal[0, 137] = 250.
    signal[1, 301] = 3.
    signal[1, 305] = 7.
    kernel = scint_kernel(NT)
    out = cp.asnumpy(run_fft64(signal, kernel))
    assert np.count_nonzero(out[0, :137]) == 0
    assert np.count_nonzero(out[1, :301]) == 0
    assert_matches_reference(out, reference64(signal, kernel), "delayed first light")


def test_gap_longer_than_window(short_window, no_warnings):
    """Ticks more than one window after the last light are exactly zero; the faint
    end of the window before them stays supported and positive."""
    signal = np.zeros((1, NT), dtype='f4')
    signal[0, 10] = 5000.
    signal[0, 400] = 1.
    kernel = scint_kernel(NT)
    klen = kernel_length(kernel, NT)
    out = cp.asnumpy(run_fft64(signal, kernel))[0]
    last_supported = 10 + klen - 1
    assert np.count_nonzero(out[last_supported + 1:400]) == 0
    assert (out[10:last_supported + 1] > 0).all()
    assert (out[400:] > 0).all()
    assert_matches_reference(out[None], reference64(signal, kernel), "gap")


def test_truncation_at_final_tick(short_window, no_warnings):
    """Light at the end of the waveform is truncated, not wrapped to the start."""
    signal = np.zeros((2, NT), dtype='f4')
    signal[0, NT - 1] = 40.
    signal[1, NT - 50] = 900.
    kernel = scint_kernel(NT)
    out = cp.asnumpy(run_fft64(signal, kernel))
    assert np.count_nonzero(out[0, :NT - 1]) == 0
    assert out[0, NT - 1] == np.float32(np.float64(kernel[0]) * 40.)
    assert np.count_nonzero(out[1, :NT - 50]) == 0
    assert (out[1, NT - 50:] > 0).all()
    assert_matches_reference(out, reference64(signal, kernel), "truncation")


def test_support_rejects_nonpositive_kernel(short_window):
    kernel = scint_kernel(NT)
    kernel[3] = 0.
    with pytest.raises(ValueError):
        light_sim.scintillation_support(cp.zeros((1, NT), dtype='f4'), kernel)


# ------------------------------------------------------ numerical accuracy ---

NT_REAL = 16000


def test_accuracy_sparse(no_warnings):
    rng = np.random.default_rng(1)
    signal = np.zeros((4, NT_REAL), dtype='f4')
    for row in range(4):
        ticks = rng.choice(NT_REAL, size=20, replace=False)
        signal[row, ticks] = np.exp(rng.uniform(0, np.log(1e4), size=20))
    kernel = scint_kernel(NT_REAL)
    assert_matches_reference(cp.asnumpy(run_fft64(signal, kernel)), reference64(signal, kernel), "sparse")


def test_accuracy_bright_pulse_then_faint_light(no_warnings):
    """A 2e4 pulse followed by sub-photon light: the late tail (~1e-4) must stay
    positive and accurate."""
    signal = np.zeros((2, NT_REAL), dtype='f4')
    signal[0, 1000] = 2e4
    signal[1, 1000] = 2e4
    signal[1, [9000, 12000, 15990]] = [0.5, 0.05, 0.01]
    kernel = scint_kernel(NT_REAL)
    ref = reference64(signal, kernel)
    assert ref[0, -1] < 1e-3 * ref[0].max()
    assert_matches_reference(cp.asnumpy(run_fft64(signal, kernel)), ref, "bright then faint")


@pytest.mark.parametrize("name", ["removed_calls", "extra_calls"])
def test_accuracy_simulation_fixtures(name, no_warnings):
    signal = load_fixture(name)
    kernel = scint_kernel(signal.shape[1])
    assert_matches_reference(cp.asnumpy(run_fft64(signal, kernel)), reference64(signal, kernel), name)


# ----------------------------------------------------------- RNG regression ---

@pytest.mark.parametrize("name", ["removed_calls", "extra_calls"])
def test_rng_matches_direct(name, no_warnings):
    """Same incoming states -> same PE and bitwise-equal outgoing states as the
    original float32 direct sum."""
    signal = load_fixture(name)
    kernel = scint_kernel(signal.shape[1])
    direct = run_direct32(signal, kernel)
    candidate = run_fft64(signal, kernel)
    states_in = fresh_states(direct)
    pe_d, st_d = sample(direct, states_in)
    pe_c, st_c = sample(candidate, states_in)
    assert np.array_equal(cp.asnumpy(candidate) > 0, cp.asnumpy(direct) > 0)
    assert np.array_equal(pe_c, pe_d)
    assert np.array_equal(st_c, st_d)


# ------------------------------------------------------- mean-30 boundary ---

def boundary_inputs():
    """Impulse and two recorded pulse shapes scaled so the direct-sum peak mean
    is 30 * (1 + eps) for eps in +-[1e-8, 1e-3] and 0."""
    shapes = [np.zeros(NT_REAL, dtype='f4'), *load_fixture("boundary_shapes")]
    shapes[0][3000] = 1.
    kernel = scint_kernel(NT_REAL)
    eps = np.concatenate([-np.logspace(-3, -8, 21), [0.], np.logspace(-8, -3, 21)])
    rows = []
    for s in shapes:
        peak = float(run_direct32(s[None], kernel).max()) * light.LIGHT_TICK_SIZE
        rows += [(s * np.float32(30 * (1 + e) / peak)).astype('f4') for e in eps]
    return np.stack(rows), kernel


def test_mean30_boundary_fixture(no_warnings):
    """On THIS constructed fixture the candidate lands on the same side of mean 30
    as the float32 direct sum everywhere, so PE and RNG states match.

    This is agreement on a fixture, not a guarantee: both results are rounded to
    float32, and for other inputs a value within an ulp of 30 can still fall on
    different sides and select a different sampling branch."""
    signal, kernel = boundary_inputs()
    direct = run_direct32(signal, kernel)
    candidate = run_fft64(signal, kernel)
    md = cp.asnumpy(direct).astype('f8') * light.LIGHT_TICK_SIZE
    mc = cp.asnumpy(candidate).astype('f8') * light.LIGHT_TICK_SIZE
    assert (np.abs(md - 30) < 1e-5).sum() > 20      # the fixture really sits on the boundary
    assert not (((md < 30) != (mc < 30)) & (md > 0)).any()
    states_in = fresh_states(direct)
    pe_d, st_d = sample(direct, states_in)
    pe_c, st_c = sample(candidate, states_in)
    assert np.array_equal(pe_c, pe_d)
    assert np.array_equal(st_c, st_d)


# --------------------------------------------------- exceptional behaviour ---

def test_supported_nonpositive_values_warn_and_are_not_corrected(short_window):
    """Unphysical negative input gives supported ticks <= 0: warn, leave them as computed."""
    signal = np.zeros((1, NT), dtype='f4')
    signal[0, 50] = -3.
    kernel = scint_kernel(NT)
    with pytest.warns(UserWarning, match=r"fft64 scintillation: \d+ supported ticks are nonpositive"):
        out = cp.asnumpy(run_fft64(signal, kernel))
    ref = reference64(signal, kernel)
    assert (out[0, 50:50 + kernel_length(kernel, NT)] < 0).all()
    assert np.count_nonzero(out[0, :50]) == 0
    assert np.abs(out - ref).max() <= np.spacing(np.float32(np.abs(ref).max())) + 1e-12 * np.abs(ref).max()


# ------------------------------------------------- mode switch and truth ---

def truth_arrays(signal):
    ids = np.full(signal.shape + (sim.MAX_MC_TRUTH_IDS,), -1, dtype='i8')
    photons = np.zeros(signal.shape + (sim.MAX_MC_TRUTH_IDS,), dtype='f8')
    rows, ticks = np.nonzero(signal)
    ids[rows, ticks, 0] = 7
    photons[rows, ticks, 0] = signal[rows, ticks]
    return cp.asarray(ids), cp.asarray(photons)


def run_calc_scintillation_effect(signal, kernel):
    sig = cp.asarray(signal)
    ids, photons = truth_arrays(signal)
    out = cp.zeros_like(sig)
    out_ids = cp.full_like(ids, -1)
    out_photons = cp.zeros_like(photons)
    light_sim.calc_scintillation_effect(grid(sig), TPB, sig, ids, photons, out, out_ids, out_photons, kernel)
    cuda.synchronize()
    return cp.asnumpy(out), cp.asnumpy(out_ids), cp.asnumpy(out_photons)


def test_mode_switch_and_unchanged_truth(monkeypatch, no_warnings):
    signal = load_fixture("extra_calls")
    kernel = scint_kernel(signal.shape[1])
    monkeypatch.delenv("LARNDSIM_LIGHT_CONV", raising=False)
    default = run_calc_scintillation_effect(signal, kernel)
    monkeypatch.setenv("LARNDSIM_LIGHT_CONV", "fft")
    explicit_fft = run_calc_scintillation_effect(signal, kernel)
    monkeypatch.setenv("LARNDSIM_LIGHT_CONV", "fft64")
    fft64 = run_calc_scintillation_effect(signal, kernel)

    assert np.array_equal(default[0], cp.asnumpy(run_fft32(signal, kernel)))   # default unchanged
    assert np.array_equal(explicit_fft[0], default[0])
    assert np.array_equal(fft64[0], cp.asnumpy(run_fft64(signal, kernel)))
    assert np.array_equal(fft64[1], default[1]) and np.array_equal(fft64[2], default[2])  # truth unchanged

    monkeypatch.setenv("LARNDSIM_LIGHT_CONV", "fft_support")
    with pytest.raises(ValueError):
        run_calc_scintillation_effect(signal, kernel)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
