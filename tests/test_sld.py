"""Tests for the SLD (Symmetric Logarithmic Derivative) module."""

import pytest
import numpy as np
import mpmath as mp

from lmg_qfi import (
    compute_sld_matrix,
    sld_expectation,
    sld_squared_expectation,
    compute_sld_info,
    quantum_fisher_information_mp,
    SLDInformation,
    create_spin_xyz_operators,
)

np.set_printoptions(precision=4, suppress=True)


def convert_mpmatrix_to_numpy(mp_math, precision=15):
    """Convert mpmath matrix to numpy array."""
    with mp.workdps(precision):
        np_mat = np.array(
            [[complex(x.real, x.imag) for x in row] for row in mp_math.tolist()],
            dtype=np.complex128
        )
    return np_mat


class TestSLDMatrix:
    """Tests for compute_sld_matrix."""

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_sld_matrix_hermitian(self, n):
        """L = 2(|∂ψ⟩⟨ψ| + |ψ⟩⟨∂ψ|) must be Hermitian."""
        ket = mp.zeros(n + 1, 1)
        ket[0] = mp.mpf('1.0')
        dket = mp.zeros(n + 1, 1)
        dket[1] = mp.mpf('0.1')

        L = compute_sld_matrix(dket, ket)
        L_np = convert_mpmatrix_to_numpy(L)

        assert np.allclose(L_np, L_np.T.conj(), atol=1e-12), "SLD matrix is not Hermitian"

    @pytest.mark.parametrize("n", [2, 5])
    def test_sld_equation(self, n):
        """(ρL + Lρ)/2 must equal ∂ρ for normalized states."""
        ket = mp.zeros(n + 1, 1)
        ket[0] = mp.mpf('1.0')
        dket = mp.zeros(n + 1, 1)
        dket[1] = mp.mpf('0.3')

        L = compute_sld_matrix(dket, ket)
        rho = ket * ket.transpose_conj()
        d_rho = dket * ket.transpose_conj() + ket * dket.transpose_conj()

        lhs = (rho * L + L * rho) * mp.mpf('0.5')
        lhs_np = convert_mpmatrix_to_numpy(lhs)
        d_rho_np = convert_mpmatrix_to_numpy(d_rho)

        assert np.allclose(lhs_np, d_rho_np, atol=1e-12), "SLD equation not satisfied"

    def test_sld_matrix_shape(self):
        """SLD matrix has shape (n+1, n+1)."""
        n = 4
        ket = mp.zeros(n + 1, 1)
        ket[0] = mp.mpf('1.0')
        dket = mp.zeros(n + 1, 1)
        dket[1] = mp.mpf('0.1')

        L = compute_sld_matrix(dket, ket)

        assert L.rows == n + 1
        assert L.cols == n + 1

    def test_sld_matrix_zero_derivative(self):
        """SLD is zero when derivative ket is zero."""
        n = 3
        ket = mp.zeros(n + 1, 1)
        ket[0] = mp.mpf('1.0')
        dket = mp.zeros(n + 1, 1)

        L = compute_sld_matrix(dket, ket)
        L_np = convert_mpmatrix_to_numpy(L)

        assert np.allclose(L_np, np.zeros((n + 1, n + 1)), atol=1e-12)


class TestSLDExpectation:
    """Tests for sld_expectation (⟨L⟩)."""

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_sld_expectation_zero_orthogonal(self, n):
        """⟨L⟩ = 0 when derivative is orthogonal to state."""
        ket = mp.zeros(n + 1, 1)
        ket[0] = mp.mpf('1.0')
        dket = mp.zeros(n + 1, 1)
        dket[1] = mp.mpf('0.5')  # orthogonal to ket[0]

        result = sld_expectation(dket, ket)

        assert mp.fabs(result) < 1e-12

    def test_sld_expectation_purely_imaginary_overlap(self):
        """⟨L⟩ = 0 when ⟨ψ|∂ψ⟩ is purely imaginary (normalized state)."""
        ket = mp.matrix([[mp.mpf('1.0')], [mp.mpf('0.0')]])
        # Purely imaginary overlap: dket = i * c * ket
        dket = mp.matrix([[mp.mpc(0, '0.3')], [mp.mpf('0.0')]])

        result = sld_expectation(dket, ket)

        # Re(⟨ψ|∂ψ⟩) = Re(0.3i) = 0, so ⟨L⟩ = 0
        assert mp.fabs(result) < 1e-12

    def test_sld_expectation_formula(self):
        """⟨L⟩ = 4 Re(⟨ψ|∂ψ⟩) matches direct computation."""
        ket = mp.matrix([[mp.mpf('0.6')], [mp.mpf('0.8')]])
        dket = mp.matrix([[mp.mpf('0.1')], [mp.mpf('0.2')]])

        result = sld_expectation(dket, ket)
        expected = 4 * mp.re((ket.transpose_conj() * dket)[0, 0])

        assert mp.fabs(result - expected) < 1e-12


class TestSLDSquaredExpectation:
    """Tests for sld_squared_expectation (⟨L²⟩)."""

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_sld_squared_matches_qfi(self, n):
        """⟨L²⟩ must equal the finite-difference QFI for a normalized state."""
        ket = mp.zeros(n + 1, 1)
        ket[0] = mp.mpf('1.0')
        dket = mp.zeros(n + 1, 1)
        dket[1] = mp.mpf('0.1')

        L2 = sld_squared_expectation(dket, ket)
        qfi = quantum_fisher_information_mp(dket, ket)

        assert mp.fabs(L2 - qfi) < 1e-12, f"⟨L²⟩={L2} != QFI={qfi}"

    def test_sld_squared_non_negative(self):
        """⟨L²⟩ must be non-negative."""
        ket = mp.matrix([[mp.mpf('0.6')], [mp.mpf('0.8')]])
        dket = mp.matrix([[mp.mpf('0.1')], [mp.mpf('-0.2')]])

        result = sld_squared_expectation(dket, ket)

        assert result >= 0

    def test_sld_squared_zero_for_zero_derivative(self):
        """⟨L²⟩ = 0 when derivative ket is zero."""
        ket = mp.matrix([[mp.mpf('1.0')], [mp.mpf('0.0')]])
        dket = mp.matrix([[mp.mpf('0.0')], [mp.mpf('0.0')]])

        result = sld_squared_expectation(dket, ket)

        assert mp.fabs(result) < 1e-12

    def test_sld_squared_via_full_matrix(self):
        """⟨L²⟩ matches Tr(ρ L²) computed from the full SLD matrix."""
        n = 3
        ket = mp.zeros(n + 1, 1)
        ket[0] = mp.mpf('1.0')
        dket = mp.zeros(n + 1, 1)
        dket[1] = mp.mpf('0.2')
        dket[2] = mp.mpf('0.1')

        L = compute_sld_matrix(dket, ket)
        rho = ket * ket.transpose_conj()
        # Tr(ρ L²) = ⟨ψ|L²|ψ⟩ for pure state
        tr_rho_L2 = mp.re((ket.transpose_conj() * L * L * ket)[0, 0])
        L2_fast = sld_squared_expectation(dket, ket)

        assert mp.fabs(tr_rho_L2 - L2_fast) < 1e-12

    @pytest.mark.parametrize("n", [2, 5])
    def test_sld_squared_global_phase_invariance(self, n):
        """⟨L²⟩ is invariant under global phase on ket."""
        ket = mp.zeros(n + 1, 1)
        ket[0] = mp.mpf('1.0') / mp.sqrt(2)
        ket[1] = mp.mpf('1.0') / mp.sqrt(2)
        dket = mp.zeros(n + 1, 1)
        dket[1] = mp.mpf('0.1')

        L2_original = sld_squared_expectation(dket, ket)

        phase = mp.exp(mp.j * mp.pi / 3)
        L2_phased = sld_squared_expectation(phase * dket, phase * ket)

        assert mp.fabs(L2_original - L2_phased) < 1e-12


class TestComputeSLDInfo:
    """Tests for compute_sld_info."""

    def test_returns_sld_information(self):
        """compute_sld_info returns an SLDInformation instance."""
        ket = mp.matrix([[mp.mpf('1.0')], [mp.mpf('0.0')]])
        dket = mp.matrix([[mp.mpf('0.0')], [mp.mpf('0.1')]])

        result = compute_sld_info(dket, ket, N=2, time=5)

        assert isinstance(result, SLDInformation)

    def test_time_field(self):
        """time field matches the input."""
        ket = mp.matrix([[mp.mpf('1.0')], [mp.mpf('0.0')]])
        dket = mp.matrix([[mp.mpf('0.0')], [mp.mpf('0.1')]])

        result = compute_sld_info(dket, ket, N=2, time=7)

        assert result.time == 7

    def test_qfi_from_sld_normalization(self):
        """qfi_from_sld = L_squared_expectation / (N² t²)."""
        ket = mp.matrix([[mp.mpf('1.0')], [mp.mpf('0.0')]])
        dket = mp.matrix([[mp.mpf('0.0')], [mp.mpf('0.1')]])
        N, t = 2, 5

        result = compute_sld_info(dket, ket, N=N, time=t)

        expected = result.L_squared_expectation / (N ** 2 * t ** 2)
        assert abs(result.qfi_from_sld - expected) < 1e-12

    def test_l_expectation_near_zero(self):
        """L_expectation is ~0 for a normalized state with orthogonal derivative."""
        ket = mp.matrix([[mp.mpf('1.0')], [mp.mpf('0.0')]])
        dket = mp.matrix([[mp.mpf('0.0')], [mp.mpf('0.3')]])

        result = compute_sld_info(dket, ket, N=2, time=3)

        assert abs(result.L_expectation) < 1e-12

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_qfi_from_sld_matches_finite_difference(self, n):
        """Normalized QFI from SLD matches finite-difference QFI / (N² t²)."""
        N = n
        t = 4
        ket = mp.zeros(n + 1, 1)
        ket[0] = mp.mpf('1.0')
        dket = mp.zeros(n + 1, 1)
        dket[1] = mp.mpf('0.15')

        result = compute_sld_info(dket, ket, N=N, time=t)
        qfi_fd = float(quantum_fisher_information_mp(dket, ket))
        qfi_normalized = qfi_fd / (N ** 2 * t ** 2)

        assert abs(result.qfi_from_sld - qfi_normalized) < 1e-12


class TestSLDExpectationExtended:
    """Additional tests for sld_expectation edge cases."""

    def test_sld_expectation_nonzero_for_real_overlap(self):
        """⟨L⟩ = 4*Re(⟨ψ|∂ψ⟩) != 0 when derivative has real component along ket."""
        ket = mp.matrix([[mp.mpf("1.0")], [mp.mpf("0.0")]])
        dket = mp.matrix([[mp.mpf("0.1")], [mp.mpf("0.0")]])

        result = sld_expectation(dket, ket)

        # ⟨ψ|∂ψ⟩ = 0.1 (real) → ⟨L⟩ = 4 * 0.1 = 0.4
        assert mp.fabs(result - mp.mpf("0.4")) < 1e-12

    @pytest.mark.parametrize("n", [2, 5])
    def test_sld_expectation_zero_for_sy_generator(self, n):
        """⟨L⟩ = 0 for Sy-generated derivative (purely imaginary overlap)."""
        Sz, Sx, Sy = create_spin_xyz_operators(n)
        ket = mp.zeros(n + 1, 1)
        ket[0] = mp.mpf("1.0")
        dket = mp.j * Sy * ket

        result = sld_expectation(dket, ket)

        assert float(mp.fabs(result)) < 1e-10

    def test_sld_expectation_superposition(self):
        """⟨L⟩ = 4*Re(⟨ψ|∂ψ⟩) for equal superposition state."""
        ket = mp.matrix(
            [[mp.mpf("1.0") / mp.sqrt(2)], [mp.mpf("1.0") / mp.sqrt(2)]]
        )
        dket = mp.matrix([[mp.mpc("0.1", "0.05")], [mp.mpc("-0.1", "0.05")]])

        result = sld_expectation(dket, ket)
        expected = 4 * mp.re((ket.transpose_conj() * dket)[0, 0])

        assert mp.fabs(result - expected) < 1e-12


class TestSLDSquaredExtended:
    """Additional tests for sld_squared_expectation."""

    @pytest.mark.parametrize("n", [2, 5])
    def test_sld_squared_matches_qfi_for_sx_generator(self, n):
        """⟨L²⟩ = QFI for Sx-generated derivative (imaginary overlap)."""
        Sz, Sx, Sy = create_spin_xyz_operators(n)
        ket = mp.zeros(n + 1, 1)
        ket[0] = mp.mpf("1.0")
        dket = mp.j * Sx * ket

        alpha = (ket.transpose_conj() * dket)[0, 0]
        assert mp.fabs(mp.re(alpha)) < 1e-10  # verify premise

        sld_sq = sld_squared_expectation(dket, ket)
        qfi = quantum_fisher_information_mp(dket, ket)

        assert mp.fabs(sld_sq - qfi) < 1e-10

    def test_sld_squared_known_value_orthogonal(self):
        """⟨L²⟩ = 4 for |ψ⟩=|0⟩ and |∂ψ⟩=|1⟩."""
        ket = mp.matrix([[mp.mpf("1")], [mp.mpf("0")]])
        dket = mp.matrix([[mp.mpf("0")], [mp.mpf("1")]])

        result = sld_squared_expectation(dket, ket)

        assert mp.fabs(result - 4) < 1e-12


class TestSLDInformation:
    """Tests for SLDInformation dataclass."""

    def test_sld_information_creation(self):
        """Test creating SLDInformation with all fields."""
        info = SLDInformation(
            time=10,
            L_expectation=0.001,
            L_squared_expectation=1.5,
            qfi_from_sld=0.015,
        )

        assert info.time == 10
        assert info.L_expectation == 0.001
        assert info.L_squared_expectation == 1.5
        assert info.qfi_from_sld == 0.015

    def test_sld_information_all_fields_accessible(self):
        """All expected fields are present on SLDInformation."""
        info = SLDInformation(
            time=1,
            L_expectation=0.0,
            L_squared_expectation=0.0,
            qfi_from_sld=0.0,
        )

        assert hasattr(info, "time")
        assert hasattr(info, "L_expectation")
        assert hasattr(info, "L_squared_expectation")
        assert hasattr(info, "qfi_from_sld")

    def test_sld_information_float_compatible(self):
        """Numeric fields are float-compatible."""
        info = SLDInformation(
            time=50,
            L_expectation=1e-5,
            L_squared_expectation=3.14,
            qfi_from_sld=0.0628,
        )

        assert isinstance(float(info.L_expectation), float)
        assert isinstance(float(info.L_squared_expectation), float)
        assert isinstance(float(info.qfi_from_sld), float)

    def test_sld_information_zero_values(self):
        """SLDInformation with zero values (stationary state)."""
        info = SLDInformation(
            time=100,
            L_expectation=0.0,
            L_squared_expectation=0.0,
            qfi_from_sld=0.0,
        )

        assert info.L_expectation == 0.0
        assert info.L_squared_expectation == 0.0
        assert info.qfi_from_sld == 0.0

    def test_sld_information_roundtrip_with_compute_sld_info(self):
        """SLDInformation produced by compute_sld_info has correct field types."""
        ket = mp.matrix([[mp.mpf("1.0")], [mp.mpf("0.0")]])
        dket = mp.matrix([[mp.mpf("0.0")], [mp.mpf("0.2")]])

        info = compute_sld_info(dket, ket, N=3, time=4)

        assert isinstance(info, SLDInformation)
        assert isinstance(info.time, (int, float))
        assert isinstance(info.L_expectation, float)
        assert isinstance(info.L_squared_expectation, float)
        assert isinstance(info.qfi_from_sld, float)
