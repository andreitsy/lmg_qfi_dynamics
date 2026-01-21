"""Tests for the operators module."""

import pytest
import numpy as np
import mpmath as mp

from lmg_qfi import (
    create_z_operator,
    create_spin_minus_operators,
    create_spin_plus_operators,
    create_spin_xyz_operators,
    create_hamiltonian_h0,
    create_kick_operator,
    ac_time,
    create_v_operator,
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


class TestZOperator:
    """Tests for z-operator creation."""

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_z_operator_dimensions(self, n):
        """Test that z-operator has correct dimensions."""
        Z = create_z_operator(n)
        assert Z.rows == n + 1
        assert Z.cols == n + 1

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_z_operator_is_diagonal(self, n):
        """Test that z-operator is diagonal."""
        Z = create_z_operator(n)
        Z_np = convert_mpmatrix_to_numpy(Z)
        # Check off-diagonal elements are zero
        for i in range(n + 1):
            for j in range(n + 1):
                if i != j:
                    assert np.isclose(Z_np[i, j], 0)

    @pytest.mark.parametrize("n", [2, 4, 6, 10])
    def test_z_operator_trace_zero(self, n):
        """Test that z-operator has zero trace for even n."""
        Z = create_z_operator(n)
        trace = sum(Z[i, i] for i in range(n + 1))
        assert mp.fabs(trace) < 1e-10

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_z_operator_eigenvalues(self, n):
        """Test that z-operator diagonal values range from n/2 to -n/2."""
        Z = create_z_operator(n)
        half_n = n / 2
        for i in range(n + 1):
            expected = half_n - i
            assert mp.fabs(Z[i, i] - expected) < 1e-10


class TestSpinLadderOperators:
    """Tests for spin raising and lowering operators."""

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_spin_minus_dimensions(self, n):
        """Test S- operator dimensions."""
        Sminus = create_spin_minus_operators(n)
        assert Sminus.rows == n + 1
        assert Sminus.cols == n + 1

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_spin_plus_dimensions(self, n):
        """Test S+ operator dimensions."""
        Splus = create_spin_plus_operators(n)
        assert Splus.rows == n + 1
        assert Splus.cols == n + 1

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_spin_plus_is_hermitian_conjugate_of_minus(self, n):
        """Test that S+ = (S-)^dagger."""
        Splus = create_spin_plus_operators(n)
        Sminus = create_spin_minus_operators(n)
        
        Splus_np = convert_mpmatrix_to_numpy(Splus)
        Sminus_np = convert_mpmatrix_to_numpy(Sminus)
        
        assert np.allclose(Splus_np, Sminus_np.T.conj())

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_commutation_relation_sz_splus(self, n):
        """Test [Sz, S+] = S+."""
        Sz, Sx, Sy = create_spin_xyz_operators(n)
        Splus = create_spin_plus_operators(n)
        
        commutator = Sz * Splus - Splus * Sz
        
        commutator_np = convert_mpmatrix_to_numpy(commutator)
        Splus_np = convert_mpmatrix_to_numpy(Splus)
        
        assert np.allclose(commutator_np, Splus_np)

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_commutation_relation_sz_sminus(self, n):
        """Test [Sz, S-] = -S-."""
        Sz, Sx, Sy = create_spin_xyz_operators(n)
        Sminus = create_spin_minus_operators(n)
        
        commutator = Sz * Sminus - Sminus * Sz
        
        commutator_np = convert_mpmatrix_to_numpy(commutator)
        Sminus_np = convert_mpmatrix_to_numpy(Sminus)
        
        assert np.allclose(commutator_np, -Sminus_np)


class TestSpinXYZOperators:
    """Tests for Sx, Sy, Sz operators."""

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_spin_operators_hermitian(self, n):
        """Test that all spin operators are Hermitian."""
        Sz, Sx, Sy = create_spin_xyz_operators(n)
        
        for S, name in [(Sz, "Sz"), (Sx, "Sx"), (Sy, "Sy")]:
            S_np = convert_mpmatrix_to_numpy(S)
            assert np.allclose(S_np, S_np.T.conj()), f"{name} is not Hermitian"

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_commutation_relation_sx_sy(self, n):
        """Test [Sx, Sy] = i*Sz."""
        Sz, Sx, Sy = create_spin_xyz_operators(n)
        
        commutator = Sx * Sy - Sy * Sx
        expected = mp.j * Sz
        
        commutator_np = convert_mpmatrix_to_numpy(commutator)
        expected_np = convert_mpmatrix_to_numpy(expected)
        
        assert np.allclose(commutator_np, expected_np)

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_casimir_operator(self, n):
        """Test S^2 = Sx^2 + Sy^2 + Sz^2 = s(s+1)*I where s = n/2."""
        Sz, Sx, Sy = create_spin_xyz_operators(n)
        
        S_squared = Sx * Sx + Sy * Sy + Sz * Sz
        s = n / 2.0
        expected_eigenvalue = s * (s + 1)
        
        S_squared_np = convert_mpmatrix_to_numpy(S_squared)
        expected_np = expected_eigenvalue * np.eye(n + 1, dtype=np.complex128)
        
        assert np.allclose(S_squared_np, expected_np)


class TestHamiltonian:
    """Tests for Hamiltonian creation."""

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_hamiltonian_hermitian(self, n):
        """Test that Hamiltonian is Hermitian."""
        J, B = 1.0, 0.4
        H = create_hamiltonian_h0(J, B, n)
        H_np = convert_mpmatrix_to_numpy(H)
        assert np.allclose(H_np, H_np.T.conj())

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_hamiltonian_real_eigenvalues(self, n):
        """Test that Hamiltonian has real eigenvalues."""
        J, B = 1.0, 0.4
        H = create_hamiltonian_h0(J, B, n)
        eigenvalues, _ = mp.eigh(H)
        
        for ev in eigenvalues:
            assert mp.fabs(mp.im(ev)) < 1e-10

    def test_hamiltonian_zero_field(self):
        """Test Hamiltonian with B=0 is purely ZZ interaction."""
        n = 5
        J, B = 1.0, 0.0
        H = create_hamiltonian_h0(J, B, n)
        Sz, Sx, Sy = create_spin_xyz_operators(n)
        
        expected = -mp.mpf(J) * (2 / mp.mpf(n)) * (Sz * Sz)
        
        H_np = convert_mpmatrix_to_numpy(H)
        expected_np = convert_mpmatrix_to_numpy(expected)
        
        assert np.allclose(H_np, expected_np)


class TestKickOperator:
    """Tests for kick operator."""

    @pytest.mark.parametrize("n", [2, 5, 10])
    @pytest.mark.parametrize("phi", [0.0, np.pi/4, np.pi/2, np.pi])
    def test_kick_operator_unitary(self, n, phi):
        """Test that kick operator is unitary."""
        Sz, Sx, Sy = create_spin_xyz_operators(n)
        K = create_kick_operator(phi, Sx)
        K_np = convert_mpmatrix_to_numpy(K)
        
        # U * U^dagger = I
        product = K_np @ K_np.T.conj()
        assert np.allclose(product, np.eye(n + 1))

    @pytest.mark.parametrize("n", [2, 5])
    def test_kick_operator_zero_phi(self, n):
        """Test that kick operator with phi=0 is identity."""
        Sz, Sx, Sy = create_spin_xyz_operators(n)
        K = create_kick_operator(0.0, Sx)
        K_np = convert_mpmatrix_to_numpy(K)
        
        assert np.allclose(K_np, np.eye(n + 1))


class TestACField:
    """Tests for AC field operators."""

    @pytest.mark.parametrize("n", [2, 5])
    def test_ac_time_hermitian(self, n):
        """Test that AC field operator is Hermitian."""
        Sz, Sx, Sy = create_spin_xyz_operators(n)
        omega = mp.mpf(2.0)
        phi_0 = mp.mpf(0.0)
        t_k = mp.mpf(0.5)
        theta = mp.mpf(0.1)
        varphi = mp.mpf(0.2)
        
        H_ac = ac_time(Sx, Sy, Sz, omega, phi_0, t_k, theta, varphi)
        H_ac_np = convert_mpmatrix_to_numpy(H_ac)
        
        assert np.allclose(H_ac_np, H_ac_np.T.conj())

    @pytest.mark.parametrize("n", [2, 5])
    def test_ac_time_zero_at_specific_times(self, n):
        """Test that AC field is zero when sin(omega*t + phi_0) = 0."""
        Sz, Sx, Sy = create_spin_xyz_operators(n)
        omega = mp.mpf(1.0)
        phi_0 = mp.mpf(0.0)
        # t_k such that omega * t_k = pi (sin = 0)
        t_k = mp.pi
        theta = mp.mpf(0.5)
        varphi = mp.mpf(0.3)
        
        H_ac = ac_time(Sx, Sy, Sz, omega, phi_0, t_k, theta, varphi)
        H_ac_np = convert_mpmatrix_to_numpy(H_ac)
        
        assert np.allclose(H_ac_np, np.zeros((n + 1, n + 1)), atol=1e-10)

    @pytest.mark.parametrize("n", [2, 5])
    def test_v_operator_equals_h0_when_h_zero(self, n):
        """Test that V = H0 when h = 0."""
        Sz, Sx, Sy = create_spin_xyz_operators(n)
        J, B = 1.0, 0.4
        H_0 = create_hamiltonian_h0(J, B, n)
        
        omega = mp.mpf(2.0)
        phi_0 = mp.mpf(0.0)
        h = mp.mpf(0.0)
        t_k = mp.mpf(0.5)
        theta = mp.mpf(0.1)
        varphi = mp.mpf(0.2)
        
        V = create_v_operator(H_0, Sx, Sy, Sz, omega, phi_0, h, t_k, theta, varphi)
        
        V_np = convert_mpmatrix_to_numpy(V)
        H0_np = convert_mpmatrix_to_numpy(H_0)
        
        assert np.allclose(V_np, H0_np)
