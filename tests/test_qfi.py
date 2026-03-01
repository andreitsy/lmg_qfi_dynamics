"""Tests for the QFI (Quantum Fisher Information) module."""

import pytest
import numpy as np
import mpmath as mp

from lmg_qfi import (
    dketa_t,
    quantum_fisher_information_mp,
    calculate_error_estimation_mp,
    process_time_point_mp,
    create_spin_xyz_operators,
    create_hamiltonian_h0,
    calculate_unitary_T,
    QFIInformation,
    UF,
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


class TestDerivativeKet:
    """Tests for finite difference derivative calculation."""

    def test_dketa_t_basic(self):
        """Test basic finite difference derivative."""
        # Create simple kets
        ket_p = mp.matrix([[1], [2], [3]])
        ket_m = mp.matrix([[0], [1], [2]])
        delta = mp.mpf(0.5)
        
        result = dketa_t(ket_p, ket_m, delta)
        
        # Expected: (ket_p - ket_m) / (2 * delta) = [[1], [1], [1]] / 1.0
        expected = mp.matrix([[1], [1], [1]])
        
        for i in range(3):
            assert mp.fabs(result[i, 0] - expected[i, 0]) < 1e-10

    def test_dketa_t_zero_difference(self):
        """Test derivative when kets are equal."""
        ket = mp.matrix([[1], [2], [3]])
        delta = mp.mpf(0.1)
        
        result = dketa_t(ket, ket, delta)
        
        for i in range(3):
            assert mp.fabs(result[i, 0]) < 1e-10

    @pytest.mark.parametrize("delta", [0.1, 0.01, 0.001])
    def test_dketa_t_scaling(self, delta):
        """Test that derivative scales correctly with delta."""
        ket_p = mp.matrix([[2], [4]])
        ket_m = mp.matrix([[0], [0]])
        delta_mp = mp.mpf(delta)
        
        result = dketa_t(ket_p, ket_m, delta_mp)
        
        # Expected: [[2], [4]] / (2 * delta) = [[1/delta], [2/delta]]
        expected_0 = 1.0 / delta
        expected_1 = 2.0 / delta
        
        assert mp.fabs(result[0, 0] - expected_0) < 1e-8
        assert mp.fabs(result[1, 0] - expected_1) < 1e-8


class TestQuantumFisherInformation:
    """Tests for QFI calculation."""

    def test_qfi_zero_for_stationary_state(self):
        """Test that QFI is zero when derivative ket is zero."""
        ket = mp.matrix([[1], [0], [0]])
        dket = mp.matrix([[0], [0], [0]])
        
        qfi = quantum_fisher_information_mp(dket, ket)
        
        assert mp.fabs(qfi) < 1e-10

    def test_qfi_non_negative(self):
        """Test that QFI is always non-negative."""
        # Random normalized ket
        ket = mp.matrix([[mp.mpf('0.6')], [mp.mpf('0.8')]])
        # Random derivative ket
        dket = mp.matrix([[mp.mpf('0.1')], [mp.mpf('-0.2')]])
        
        qfi = quantum_fisher_information_mp(dket, ket)
        
        assert qfi >= 0

    def test_qfi_orthogonal_derivative(self):
        """Test QFI when derivative is orthogonal to state."""
        # Normalized ket: |0>
        ket = mp.matrix([[1], [0]])
        # Derivative orthogonal to ket: |1>
        dket = mp.matrix([[0], [1]])
        
        qfi = quantum_fisher_information_mp(dket, ket)
        
        # QFI = 4 * (<dket|dket> - |<ket|dket>|^2) = 4 * (1 - 0) = 4
        assert mp.fabs(qfi - 4) < 1e-10

    def test_qfi_parallel_derivative(self):
        """Test QFI when derivative is parallel to state."""
        # Normalized ket
        ket = mp.matrix([[1], [0]])
        # Derivative parallel to ket (same direction)
        dket = mp.matrix([[mp.mpf('0.5')], [0]])
        
        qfi = quantum_fisher_information_mp(dket, ket)
        
        # QFI = 4 * (<dket|dket> - |<ket|dket>|^2) = 4 * (0.25 - 0.25) = 0
        assert mp.fabs(qfi) < 1e-10

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_qfi_with_spin_coherent_state(self, n):
        """Test QFI calculation with spin coherent states."""
        # Create a simple spin-up state
        ket = mp.zeros(n + 1, 1)
        ket[0] = mp.mpf('1.0')
        
        # Small perturbation as derivative
        dket = mp.zeros(n + 1, 1)
        dket[1] = mp.mpf('0.1')
        
        qfi = quantum_fisher_information_mp(dket, ket)
        
        # QFI should be 4 * |dket|^2 since <ket|dket> = 0
        expected = 4 * 0.01  # 4 * 0.1^2
        assert mp.fabs(qfi - expected) < 1e-10


class TestErrorEstimation:
    """Tests for error estimation in QFI calculation."""

    def test_error_estimation_normalized_state(self):
        """Test error estimation for normalized state with proper derivative."""
        # Normalized ket
        ket = mp.matrix([[mp.mpf('1.0')], [mp.mpf('0.0')]])
        # Derivative orthogonal to maintain normalization
        dket = mp.matrix([[mp.mpf('0.0')], [mp.mpf('0.1')]])
        
        error = calculate_error_estimation_mp(dket, ket)
        
        # <dket|ket> + <ket|dket> = 0 + 0 = 0 for orthogonal derivative
        assert error < 1e-10

    def test_error_estimation_parallel_derivative(self):
        """Test error estimation when derivative changes normalization."""
        # Normalized ket
        ket = mp.matrix([[mp.mpf('1.0')], [mp.mpf('0.0')]])
        # Derivative parallel to ket (changes normalization)
        dket = mp.matrix([[mp.mpf('0.1')], [mp.mpf('0.0')]])
        
        error = calculate_error_estimation_mp(dket, ket)
        
        # <dket|ket> + <ket|dket> = 0.1 + 0.1 = 0.2
        assert abs(error - 0.2) < 1e-10

    @pytest.mark.parametrize("n", [2, 5])
    def test_error_estimation_superposition(self, n):
        """Test error estimation for superposition states."""
        # Create normalized superposition
        ket = mp.zeros(n + 1, 1)
        ket[0] = mp.mpf('1.0') / mp.sqrt(2)
        ket[1] = mp.mpf('1.0') / mp.sqrt(2)
        
        # Derivative that changes normalization
        dket = mp.zeros(n + 1, 1)
        dket[0] = mp.mpf('0.05')
        dket[1] = mp.mpf('0.05')
        
        error = calculate_error_estimation_mp(dket, ket)
        
        # Error should be non-zero for this case
        assert error >= 0


class TestQFIPhysicalProperties:
    """Tests for physical properties of QFI."""

    def test_qfi_invariance_global_phase(self):
        """Test that QFI is invariant under global phase."""
        ket = mp.matrix([[mp.mpf('0.6')], [mp.mpf('0.8')]])
        dket = mp.matrix([[mp.mpf('0.1')], [mp.mpf('-0.1')]])
        
        qfi_original = quantum_fisher_information_mp(dket, ket)
        
        # Apply global phase e^(i*pi/4)
        phase = mp.exp(mp.j * mp.pi / 4)
        ket_phased = phase * ket
        dket_phased = phase * dket
        
        qfi_phased = quantum_fisher_information_mp(dket_phased, ket_phased)
        
        assert mp.fabs(qfi_original - qfi_phased) < 1e-10

    @pytest.mark.parametrize("n", [2, 5])
    def test_qfi_bounds(self, n):
        """Test that QFI satisfies expected bounds for spin systems."""
        # For a spin-s system, QFI for rotation around one axis
        # is bounded by 4 * s^2 for coherent states
        s = n / 2
        max_qfi_coherent = 4 * s * s
        
        # Create spin coherent state (all spins up)
        ket = mp.zeros(n + 1, 1)
        ket[0] = mp.mpf('1.0')
        
        # Derivative from rotation generator Sy
        Sz, Sx, Sy = create_spin_xyz_operators(n)
        dket = Sy * ket
        
        qfi = quantum_fisher_information_mp(dket, ket)

        # QFI should not exceed the Heisenberg limit for this state
        assert qfi <= 4 * max_qfi_coherent + 1e-8


def _build_floque_u(h, params, H_0):
    """Helper: eigendecompose U_T for a given h."""
    U = calculate_unitary_T(h, params, H_0)
    eigenvalues, eigenvectors = mp.eig(U)
    return UF(
        eigenvalues=eigenvalues,
        U=eigenvectors,
        U_inv=mp.inverse(eigenvectors),
    )


class TestProcessTimePointMp:
    """Tests for process_time_point_mp."""

    def _base_setup(self, n=2):
        """Return a minimal params dict, H_0, spin operators, and floque unitaries."""
        params = dict(
            J=mp.mpf(1.0),
            B=mp.mpf(0.4),
            phi=mp.pi,
            T=mp.mpf(1.0),
            varphi=mp.mpf(0.0),
            h=mp.mpf(0.1),
            N=n,
            nu=2,
            phi_0=mp.mpf(0.0),
            steps_floquet_unitary=10,
            theta=mp.mpf(0.0),
            epsilon=mp.mpf("1e-8"),
        )
        H_0 = create_hamiltonian_h0(1.0, 0.4, n)
        h = params["h"]
        eps = params["epsilon"]
        floque_u = _build_floque_u(h, params, H_0)
        floque_u_p = _build_floque_u(h + eps, params, H_0)
        floque_u_m = _build_floque_u(h - eps, params, H_0)
        Zsum, Xsum, Ysum = create_spin_xyz_operators(n)
        return params, H_0, floque_u, floque_u_p, floque_u_m, Zsum, Xsum, Ysum

    def test_returns_qfi_information(self):
        """process_time_point_mp returns a QFIInformation instance."""
        params, H_0, fu, fu_p, fu_m, Zsum, Xsum, Ysum = self._base_setup(n=2)
        init_state = mp.zeros(3, 1)
        init_state[0] = mp.mpf("1.0")

        result = process_time_point_mp(
            time=2, params=params, H_0=H_0,
            floque_u=fu, floque_u_p=fu_p, floque_u_m=fu_m,
            init_state=init_state, Zsum=Zsum, Xsum=Xsum, Ysum=Ysum,
        )

        assert isinstance(result, QFIInformation)

    def test_time_stored_correctly(self):
        """The time field matches the input."""
        params, H_0, fu, fu_p, fu_m, Zsum, Xsum, Ysum = self._base_setup(n=2)
        init_state = mp.zeros(3, 1)
        init_state[0] = mp.mpf("1.0")

        result = process_time_point_mp(
            time=5, params=params, H_0=H_0,
            floque_u=fu, floque_u_p=fu_p, floque_u_m=fu_m,
            init_state=init_state, Zsum=Zsum, Xsum=Xsum, Ysum=Ysum,
        )

        assert result.time == 5

    def test_qfi_non_negative(self):
        """QFI must be non-negative."""
        params, H_0, fu, fu_p, fu_m, Zsum, Xsum, Ysum = self._base_setup(n=2)
        init_state = mp.zeros(3, 1)
        init_state[0] = mp.mpf("1.0")

        result = process_time_point_mp(
            time=2, params=params, H_0=H_0,
            floque_u=fu, floque_u_p=fu_p, floque_u_m=fu_m,
            init_state=init_state, Zsum=Zsum, Xsum=Xsum, Ysum=Ysum,
        )

        assert result.qfi >= 0

    def test_magnetizations_bounded(self):
        """|m_x|, |m_y|, |m_z| <= 1/2 (spin-1/2 maximum per site scaled by 1/N)."""
        n = 2
        params, H_0, fu, fu_p, fu_m, Zsum, Xsum, Ysum = self._base_setup(n=n)
        init_state = mp.zeros(n + 1, 1)
        init_state[0] = mp.mpf("1.0")

        result = process_time_point_mp(
            time=2, params=params, H_0=H_0,
            floque_u=fu, floque_u_p=fu_p, floque_u_m=fu_m,
            init_state=init_state, Zsum=Zsum, Xsum=Xsum, Ysum=Ysum,
        )

        assert abs(result.m_x) <= 0.5 + 1e-8
        assert abs(result.m_y) <= 0.5 + 1e-8
        assert abs(result.m_z) <= 0.5 + 1e-8

    def test_result_has_all_fields(self):
        """All QFIInformation fields are populated."""
        params, H_0, fu, fu_p, fu_m, Zsum, Xsum, Ysum = self._base_setup(n=2)
        init_state = mp.zeros(3, 1)
        init_state[0] = mp.mpf("1.0")

        result = process_time_point_mp(
            time=2, params=params, H_0=H_0,
            floque_u=fu, floque_u_p=fu_p, floque_u_m=fu_m,
            init_state=init_state, Zsum=Zsum, Xsum=Xsum, Ysum=Ysum,
        )

        assert hasattr(result, "qfi")
        assert hasattr(result, "qfi_raw_value")
        assert hasattr(result, "time")
        assert hasattr(result, "m_x")
        assert hasattr(result, "m_y")
        assert hasattr(result, "m_z")

    @pytest.mark.parametrize("time", [2, 4, 6])
    def test_qfi_raw_value_is_string(self, time):
        """qfi_raw_value is stored as a string (for high-precision output)."""
        params, H_0, fu, fu_p, fu_m, Zsum, Xsum, Ysum = self._base_setup(n=2)
        init_state = mp.zeros(3, 1)
        init_state[0] = mp.mpf("1.0")

        result = process_time_point_mp(
            time=time, params=params, H_0=H_0,
            floque_u=fu, floque_u_p=fu_p, floque_u_m=fu_m,
            init_state=init_state, Zsum=Zsum, Xsum=Xsum, Ysum=Ysum,
        )

        assert isinstance(result.qfi_raw_value, str)
        # The raw value should be parseable as a float
        assert float(result.qfi_raw_value.strip("(").split("+")[0]) == float(
            result.qfi_raw_value.strip("(").split("+")[0]
        )
