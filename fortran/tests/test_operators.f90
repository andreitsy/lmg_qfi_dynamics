program test_operators
  use lmg_constants, only: dp, CI, CZERO, CONE, PI
  use lmg_operators
  use lmg_linalg, only: eye_complex
  use test_utils
  implicit none

  call test_suite_start('lmg_operators')

  call test_z_operator_dimensions()
  call test_z_operator_diagonal()
  call test_z_operator_trace_zero()
  call test_spin_plus_is_adjoint_of_minus()
  call test_spin_operators_hermitian()
  call test_commutation_sx_sy()
  call test_commutation_sy_sz()
  call test_commutation_sz_sx()
  call test_casimir_operator()
  call test_hamiltonian_hermitian()
  call test_hamiltonian_zero_field()
  call test_hamiltonian_zero_zz()
  call test_kick_unitary()
  call test_kick_zero_phi()
  call test_ac_time_hermitian()
  call test_ac_time_zero_at_sin_zero()
  call test_ac_theta_zero_gives_sz()
  call test_ac_theta_pi2_varphi_0_gives_sx()
  call test_ac_theta_pi2_varphi_pi2_gives_sy()
  call test_v_equals_h0_when_h_zero()
  call test_v_hermitian()

  call test_suite_end()
  if (tests_failed > 0) stop 1

contains

  subroutine test_z_operator_dimensions()
    integer, parameter :: n = 5
    complex(dp) :: Sz(n+1, n+1)
    call test_start('z_operator_dimensions')
    call create_z_operator(n, Sz)
    ! Just check it doesn't crash; dimension is compile-time
    call test_pass()
  end subroutine

  subroutine test_z_operator_diagonal()
    integer, parameter :: n = 4
    complex(dp) :: Sz(n+1, n+1)
    real(dp) :: half_n
    integer :: i
    call test_start('z_operator_diagonal_values')
    call create_z_operator(n, Sz)
    half_n = real(n, dp) / 2.0_dp
    do i = 1, n+1
      call assert_close(real(Sz(i,i), dp), half_n - real(i-1, dp), 1e-12_dp, 'Sz diag')
    end do
    call test_pass()
  end subroutine

  subroutine test_z_operator_trace_zero()
    integer, parameter :: n = 6
    complex(dp) :: Sz(n+1, n+1)
    complex(dp) :: trace
    integer :: i
    call test_start('z_operator_trace_zero')
    call create_z_operator(n, Sz)
    trace = CZERO
    do i = 1, n+1
      trace = trace + Sz(i,i)
    end do
    call assert_close(abs(trace), 0.0_dp, 1e-12_dp, 'Sz trace')
    call test_pass()
  end subroutine

  subroutine test_spin_plus_is_adjoint_of_minus()
    integer, parameter :: n = 5
    complex(dp) :: Splus(n+1,n+1), Sminus(n+1,n+1), Sminus_dag(n+1,n+1)
    call test_start('S+ = (S-)^dag')
    call create_spin_plus_operator(n, Splus)
    call create_spin_minus_operator(n, Sminus)
    Sminus_dag = conjg(transpose(Sminus))
    call assert_matrix_close(Splus, Sminus_dag, n+1, 1e-12_dp, 'S+ = S-^dag')
    call test_pass()
  end subroutine

  subroutine test_spin_operators_hermitian()
    integer, parameter :: n = 5
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1)
    call test_start('spin_operators_hermitian')
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    call assert_matrix_hermitian(Sz, n+1, 1e-12_dp, 'Sz Hermitian')
    call assert_matrix_hermitian(Sx, n+1, 1e-12_dp, 'Sx Hermitian')
    call assert_matrix_hermitian(Sy, n+1, 1e-12_dp, 'Sy Hermitian')
    call test_pass()
  end subroutine

  subroutine test_commutation_sx_sy()
    integer, parameter :: n = 5
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1)
    complex(dp) :: comm(n+1,n+1), expected(n+1,n+1)
    call test_start('[Sx,Sy] = i*Sz')
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    comm = matmul(Sx, Sy) - matmul(Sy, Sx)
    expected = CI * Sz
    call assert_matrix_close(comm, expected, n+1, 1e-10_dp, '[Sx,Sy]=iSz')
    call test_pass()
  end subroutine

  subroutine test_commutation_sy_sz()
    integer, parameter :: n = 5
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1)
    complex(dp) :: comm(n+1,n+1), expected(n+1,n+1)
    call test_start('[Sy,Sz] = i*Sx')
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    comm = matmul(Sy, Sz) - matmul(Sz, Sy)
    expected = CI * Sx
    call assert_matrix_close(comm, expected, n+1, 1e-10_dp, '[Sy,Sz]=iSx')
    call test_pass()
  end subroutine

  subroutine test_commutation_sz_sx()
    integer, parameter :: n = 5
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1)
    complex(dp) :: comm(n+1,n+1), expected(n+1,n+1)
    call test_start('[Sz,Sx] = i*Sy')
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    comm = matmul(Sz, Sx) - matmul(Sx, Sz)
    expected = CI * Sy
    call assert_matrix_close(comm, expected, n+1, 1e-10_dp, '[Sz,Sx]=iSy')
    call test_pass()
  end subroutine

  subroutine test_casimir_operator()
    integer, parameter :: n = 5
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1)
    complex(dp) :: S2(n+1,n+1), I_mat(n+1,n+1), expected(n+1,n+1)
    real(dp) :: s
    call test_start('S^2 = s(s+1)*I')
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    S2 = matmul(Sx,Sx) + matmul(Sy,Sy) + matmul(Sz,Sz)
    s = real(n, dp) / 2.0_dp
    call eye_complex(n+1, I_mat)
    expected = s * (s + 1.0_dp) * I_mat
    call assert_matrix_close(S2, expected, n+1, 1e-10_dp, 'Casimir')
    call test_pass()
  end subroutine

  subroutine test_hamiltonian_hermitian()
    integer, parameter :: n = 5
    complex(dp) :: H0(n+1, n+1)
    call test_start('H0_hermitian')
    call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
    call assert_matrix_hermitian(H0, n+1, 1e-12_dp, 'H0 Hermitian')
    call test_pass()
  end subroutine

  subroutine test_hamiltonian_zero_field()
    integer, parameter :: n = 5
    complex(dp) :: H0(n+1,n+1), Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1)
    complex(dp) :: expected(n+1,n+1)
    call test_start('H0_zero_field')
    call create_hamiltonian_h0(1.0_dp, 0.0_dp, n, H0)
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    expected = -1.0_dp * (2.0_dp / real(n, dp)) * matmul(Sz, Sz)
    call assert_matrix_close(H0, expected, n+1, 1e-12_dp, 'H0 with B=0')
    call test_pass()
  end subroutine

  subroutine test_hamiltonian_zero_zz()
    integer, parameter :: n = 5
    complex(dp) :: H0(n+1,n+1), Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1)
    complex(dp) :: expected(n+1,n+1)
    call test_start('H0_zero_zz')
    call create_hamiltonian_h0(0.0_dp, 1.0_dp, n, H0)
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    expected = -2.0_dp * Sx
    call assert_matrix_close(H0, expected, n+1, 1e-12_dp, 'H0 with J=0')
    call test_pass()
  end subroutine

  subroutine test_kick_unitary()
    integer, parameter :: n = 5
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1), kick(n+1,n+1)
    call test_start('kick_unitary')
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    call create_kick_operator(PI, Sx, n, kick)
    call assert_matrix_unitary(kick, n+1, 1e-10_dp, 'kick unitary')
    call test_pass()
  end subroutine

  subroutine test_kick_zero_phi()
    integer, parameter :: n = 5
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1), kick(n+1,n+1), I_mat(n+1,n+1)
    call test_start('kick_zero_phi')
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    call create_kick_operator(0.0_dp, Sx, n, kick)
    call eye_complex(n+1, I_mat)
    call assert_matrix_close(kick, I_mat, n+1, 1e-12_dp, 'kick(phi=0)=I')
    call test_pass()
  end subroutine

  subroutine test_ac_time_hermitian()
    integer, parameter :: n = 5
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1), H_ac(n+1,n+1)
    call test_start('ac_time_hermitian')
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    call ac_time(Sx, Sy, Sz, n, 2.0_dp, 0.0_dp, 0.5_dp, 0.1_dp, 0.2_dp, H_ac)
    call assert_matrix_hermitian(H_ac, n+1, 1e-12_dp, 'ac_time Hermitian')
    call test_pass()
  end subroutine

  subroutine test_ac_time_zero_at_sin_zero()
    integer, parameter :: n = 5
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1), H_ac(n+1,n+1)
    complex(dp) :: zero_mat(n+1,n+1)
    call test_start('ac_time_zero_at_sin_zero')
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    ! omega=1, phi_0=0, t_k=pi => sin(pi) = 0
    call ac_time(Sx, Sy, Sz, n, 1.0_dp, 0.0_dp, PI, 0.5_dp, 0.3_dp, H_ac)
    zero_mat = CZERO
    call assert_matrix_close(H_ac, zero_mat, n+1, 1e-10_dp, 'ac=0 at sin=0')
    call test_pass()
  end subroutine

  subroutine test_ac_theta_zero_gives_sz()
    integer, parameter :: n = 5
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1), H_ac(n+1,n+1)
    call test_start('ac_theta0_gives_Sz')
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    ! omega=1, t_k=pi/2 => sin(pi/2)=1, theta=0 => cos(0)*Sz = Sz
    call ac_time(Sx, Sy, Sz, n, 1.0_dp, 0.0_dp, PI/2.0_dp, 0.0_dp, 0.3_dp, H_ac)
    call assert_matrix_close(H_ac, Sz, n+1, 1e-10_dp, 'theta=0 => Sz')
    call test_pass()
  end subroutine

  subroutine test_ac_theta_pi2_varphi_0_gives_sx()
    integer, parameter :: n = 5
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1), H_ac(n+1,n+1)
    call test_start('ac_theta_pi2_varphi0_gives_Sx')
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    call ac_time(Sx, Sy, Sz, n, 1.0_dp, 0.0_dp, PI/2.0_dp, PI/2.0_dp, 0.0_dp, H_ac)
    call assert_matrix_close(H_ac, Sx, n+1, 1e-10_dp, 'theta=pi/2,varphi=0 => Sx')
    call test_pass()
  end subroutine

  subroutine test_ac_theta_pi2_varphi_pi2_gives_sy()
    integer, parameter :: n = 5
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1), H_ac(n+1,n+1)
    call test_start('ac_theta_pi2_varphi_pi2_gives_Sy')
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    call ac_time(Sx, Sy, Sz, n, 1.0_dp, 0.0_dp, PI/2.0_dp, PI/2.0_dp, PI/2.0_dp, H_ac)
    call assert_matrix_close(H_ac, Sy, n+1, 1e-10_dp, 'theta=pi/2,varphi=pi/2 => Sy')
    call test_pass()
  end subroutine

  subroutine test_v_equals_h0_when_h_zero()
    integer, parameter :: n = 5
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1)
    complex(dp) :: H0(n+1,n+1), V(n+1,n+1)
    call test_start('V=H0_when_h=0')
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
    call create_v_operator(H0, Sx, Sy, Sz, n, 2.0_dp, 0.0_dp, 0.0_dp, 0.5_dp, 0.1_dp, 0.2_dp, V)
    call assert_matrix_close(V, H0, n+1, 1e-12_dp, 'V=H0 when h=0')
    call test_pass()
  end subroutine

  subroutine test_v_hermitian()
    integer, parameter :: n = 5
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1)
    complex(dp) :: H0(n+1,n+1), V(n+1,n+1)
    call test_start('V_hermitian')
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
    call create_v_operator(H0, Sx, Sy, Sz, n, 2.0_dp, 0.1_dp, 0.5_dp, 0.3_dp, 0.4_dp, 0.7_dp, V)
    call assert_matrix_hermitian(V, n+1, 1e-12_dp, 'V Hermitian')
    call test_pass()
  end subroutine

end program test_operators
