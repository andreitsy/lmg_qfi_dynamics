"""Tests for the evolution module."""

import pytest
import numpy as np
import mpmath as mp

from lmg_qfi import (
    find_power_r_mpmath,
    evalution_T_step,
    calculate_unitary_T,
    create_spin_xyz_operators,
    create_hamiltonian_h0,
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


class TestFindPowerR:
    """Tests for Floquet unitary power calculation."""

    @pytest.mark.parametrize("n", [2, 5])
    def test_power_zero_returns_identity(self, n):
        """Test that U^0 = I."""
        # Create a simple unitary
        H = create_hamiltonian_h0(1.0, 0.4, n)
        U = mp.expm(-mp.j * H)
        
        eigenvalues, eigenvectors = mp.eig(U)
        uf = UF(
            eigenvalues=eigenvalues,
            U=eigenvectors,
            U_inv=mp.inverse(eigenvectors)
        )
        
        result = find_power_r_mpmath(uf, 0)
        result_np = convert_mpmatrix_to_numpy(result)
        
        assert np.allclose(result_np, np.eye(n + 1))

    @pytest.mark.parametrize("n", [2, 5])
    def test_power_one_returns_original(self, n):
        """Test that U^1 = U."""
        H = create_hamiltonian_h0(1.0, 0.4, n)
        U = mp.expm(-mp.j * H)
        
        eigenvalues, eigenvectors = mp.eig(U)
        uf = UF(
            eigenvalues=eigenvalues,
            U=eigenvectors,
            U_inv=mp.inverse(eigenvectors)
        )
        
        result = find_power_r_mpmath(uf, 1)
        
        result_np = convert_mpmatrix_to_numpy(result)
        U_np = convert_mpmatrix_to_numpy(U)
        
        assert np.allclose(result_np, U_np)

    @pytest.mark.parametrize("n", [2, 5])
    @pytest.mark.parametrize("r", [2, 3, 5])
    def test_power_r_equals_repeated_multiplication(self, n, r):
        """Test that U^r computed via eigendecomposition equals U*U*...*U."""
        H = create_hamiltonian_h0(1.0, 0.4, n)
        U = mp.expm(-mp.j * H)
        
        eigenvalues, eigenvectors = mp.eig(U)
        uf = UF(
            eigenvalues=eigenvalues,
            U=eigenvectors,
            U_inv=mp.inverse(eigenvectors)
        )
        
        # Compute via eigendecomposition
        result_eigen = find_power_r_mpmath(uf, r)
        
        # Compute via repeated multiplication
        result_mult = mp.eye(n + 1)
        for _ in range(r):
            result_mult = U * result_mult
        
        result_eigen_np = convert_mpmatrix_to_numpy(result_eigen)
        result_mult_np = convert_mpmatrix_to_numpy(result_mult)
        
        assert np.allclose(result_eigen_np, result_mult_np)

    @pytest.mark.parametrize("n", [2, 5])
    def test_power_preserves_unitarity(self, n):
        """Test that U^r is still unitary."""
        H = create_hamiltonian_h0(1.0, 0.4, n)
        U = mp.expm(-mp.j * H)
        
        eigenvalues, eigenvectors = mp.eig(U)
        uf = UF(
            eigenvalues=eigenvalues,
            U=eigenvectors,
            U_inv=mp.inverse(eigenvectors)
        )
        
        result = find_power_r_mpmath(uf, 5)
        result_np = convert_mpmatrix_to_numpy(result)
        
        # U * U^dagger = I
        product = result_np @ result_np.T.conj()
        assert np.allclose(product, np.eye(n + 1))


class TestEvolutionTStep:
    """Tests for single period evolution step."""

    @pytest.mark.parametrize("n", [2, 5])
    def test_evolution_preserves_unitarity(self, n):
        """Test that evolution step produces unitary operator."""
        H_0 = create_hamiltonian_h0(1.0, 0.4, n)
        Zsum, Xsum, Ysum = create_spin_xyz_operators(n)
        
        T = mp.mpf(1.0)
        steps = 10
        t_delta = T / steps
        omega = mp.mpf(2.0) * mp.pi / T
        
        floquet_unitary = mp.eye(n + 1)
        floquet_unitary = evalution_T_step(
            floquet_unitary,
            h=mp.mpf(0.0),
            T=T,
            varphi=mp.mpf(0.0),
            theta=mp.mpf(0.0),
            phi_0=mp.mpf(0.0),
            H_0=H_0,
            Xsum=Xsum,
            Ysum=Ysum,
            Zsum=Zsum,
            omega=omega,
            p=1,
            t_delta=t_delta,
            steps_floquet_unitary=steps,
        )
        
        U_np = convert_mpmatrix_to_numpy(floquet_unitary)
        product = U_np @ U_np.T.conj()
        
        assert np.allclose(product, np.eye(n + 1))

    @pytest.mark.parametrize("n", [2, 5])
    def test_evolution_with_zero_h_is_pure_h0(self, n):
        """Test that evolution with h=0 is exp(-i*H0*T)."""
        H_0 = create_hamiltonian_h0(1.0, 0.4, n)
        Zsum, Xsum, Ysum = create_spin_xyz_operators(n)
        
        T = mp.mpf(1.0)
        steps = 50  # High steps for accuracy
        t_delta = T / steps
        omega = mp.mpf(2.0) * mp.pi / T
        
        floquet_unitary = mp.eye(n + 1)
        floquet_unitary = evalution_T_step(
            floquet_unitary,
            h=mp.mpf(0.0),
            T=T,
            varphi=mp.mpf(0.0),
            theta=mp.mpf(0.0),
            phi_0=mp.mpf(0.0),
            H_0=H_0,
            Xsum=Xsum,
            Ysum=Ysum,
            Zsum=Zsum,
            omega=omega,
            p=1,
            t_delta=t_delta,
            steps_floquet_unitary=steps,
        )
        
        # Expected: exp(-i * H_0 * T)
        expected = mp.expm(-mp.j * H_0 * T)
        
        U_np = convert_mpmatrix_to_numpy(floquet_unitary)
        expected_np = convert_mpmatrix_to_numpy(expected)
        
        assert np.allclose(U_np, expected_np, atol=1e-3)


class TestCalculateUnitaryT:
    """Tests for full period unitary calculation."""

    @pytest.mark.parametrize("n", [2, 5])
    def test_unitary_T_is_unitary(self, n):
        """Test that calculated U(T) is unitary."""
        H_0 = create_hamiltonian_h0(1.0, 0.4, n)
        params = dict(
            J=mp.mpf(1.0),
            B=mp.mpf(0.4),
            phi=mp.pi,
            T=mp.mpf(1.0),
            varphi=mp.mpf(0.0),
            h=mp.mpf(0.0),
            N=n,
            nu=2,
            phi_0=mp.mpf(0.0),
            steps_floquet_unitary=20,
            theta=mp.mpf(0.0),
        )
        
        U_T = calculate_unitary_T(mp.mpf(0.0), params, H_0)
        U_T_np = convert_mpmatrix_to_numpy(U_T)
        
        product = U_T_np @ U_T_np.T.conj()
        assert np.allclose(product, np.eye(n + 1))

    @pytest.mark.parametrize("n", [2, 5])
    @pytest.mark.parametrize("nu", [1, 2, 3])
    def test_unitary_T_different_frequencies(self, n, nu):
        """Test U(T) calculation with different driving frequencies."""
        H_0 = create_hamiltonian_h0(1.0, 0.4, n)
        params = dict(
            J=mp.mpf(1.0),
            B=mp.mpf(0.4),
            phi=mp.pi,
            T=mp.mpf(1.0),
            varphi=mp.mpf(0.0),
            h=mp.mpf(0.0),
            N=n,
            nu=nu,
            phi_0=mp.mpf(0.0),
            steps_floquet_unitary=20,
            theta=mp.mpf(0.0),
        )
        
        U_T = calculate_unitary_T(mp.mpf(0.0), params, H_0)
        U_T_np = convert_mpmatrix_to_numpy(U_T)
        
        # Check unitarity
        product = U_T_np @ U_T_np.T.conj()
        assert np.allclose(product, np.eye(n + 1))

    @pytest.mark.parametrize("n", [2, 5])
    def test_determinant_magnitude_one(self, n):
        """Test that det(U) has magnitude 1."""
        H_0 = create_hamiltonian_h0(1.0, 0.4, n)
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
            steps_floquet_unitary=20,
            theta=mp.mpf(0.1),
        )
        
        U_T = calculate_unitary_T(mp.mpf(0.1), params, H_0)
        det = mp.det(U_T)
        
        assert mp.fabs(mp.fabs(det) - 1) < 1e-8
