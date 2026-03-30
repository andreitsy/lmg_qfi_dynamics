program test_evolution
  use lmg_constants, only: dp, CI, CZERO, CONE, PI
  use lmg_types, only: UF_type, SimulationParams_type
  use lmg_linalg, only: eig, matrix_inverse, eye_complex, expm_hermitian
  use lmg_operators, only: create_spin_xyz_operators, create_hamiltonian_h0, create_kick_operator
  use lmg_evolution, only: evolution_T_step, find_power_r, build_uf, &
                            calculate_unitary_T, calculate_unitary_at_time
  use test_utils
  implicit none

  call test_suite_start('lmg_evolution')

  call test_power_zero_identity()
  call test_power_one_original()
  call test_power_r_repeated_mult()
  call test_power_preserves_unitarity()
  call test_evolution_step_unitary()
  call test_evolution_h0_trotter()
  call test_unitary_T_is_unitary()
  call test_unitary_T_det_magnitude()
  call test_at_time_zero_identity()
  call test_at_time_nu_equals_T()
  call test_at_time_result_unitary()

  call test_suite_end()
  if (tests_failed > 0) stop 1

contains

  function make_params(n, nu) result(params)
    integer, intent(in) :: n, nu
    type(SimulationParams_type) :: params

    params%N = n
    params%J = 1.0_dp
    params%B = 0.4_dp
    params%T = 1.0_dp
    params%phi_kick_phase = PI
    params%h = 0.0_dp
    params%varphi = 0.0_dp
    params%theta = 0.0_dp
    params%phi_0 = 0.0_dp
    params%freq = nu
    params%steps_floquet_unitary = 20
    params%num_points = 200
    params%dps = 15
    params%output_dir = 'results'
  end function make_params

  subroutine make_uf_from_h(n, uf)
    integer, intent(in) :: n
    type(UF_type), intent(out) :: uf
    complex(dp) :: H0(n+1,n+1), U(n+1,n+1)
    integer :: dim

    dim = n + 1
    call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
    call expm_hermitian(H0, dim, 1.0_dp, U)  ! exp(-i*H0)
    call build_uf(U, dim, uf)
  end subroutine make_uf_from_h

  subroutine test_power_zero_identity()
    integer, parameter :: n = 5
    type(UF_type) :: uf
    complex(dp) :: result(n+1,n+1), I_mat(n+1,n+1)

    call test_start('U^0 = I')
    call make_uf_from_h(n, uf)
    call find_power_r(uf, 0, n+1, result)
    call eye_complex(n+1, I_mat)
    call assert_matrix_close(result, I_mat, n+1, 1e-12_dp, 'U^0=I')
    call test_pass()
  end subroutine

  subroutine test_power_one_original()
    integer, parameter :: n = 5
    type(UF_type) :: uf
    complex(dp) :: H0(n+1,n+1), U(n+1,n+1), result(n+1,n+1)

    call test_start('U^1 = U')
    call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
    call expm_hermitian(H0, n+1, 1.0_dp, U)
    call build_uf(U, n+1, uf)
    call find_power_r(uf, 1, n+1, result)
    call assert_matrix_close(result, U, n+1, 1e-10_dp, 'U^1=U')
    call test_pass()
  end subroutine

  subroutine test_power_r_repeated_mult()
    integer, parameter :: n = 2, r = 3
    type(UF_type) :: uf
    complex(dp) :: H0(n+1,n+1), U(n+1,n+1), result_eigen(n+1,n+1), result_mult(n+1,n+1)
    integer :: i

    call test_start('U^r == U*U*...*U')
    call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
    call expm_hermitian(H0, n+1, 1.0_dp, U)
    call build_uf(U, n+1, uf)
    call find_power_r(uf, r, n+1, result_eigen)

    call eye_complex(n+1, result_mult)
    do i = 1, r
      result_mult = matmul(U, result_mult)
    end do

    call assert_matrix_close(result_eigen, result_mult, n+1, 1e-10_dp, 'U^r repeated')
    call test_pass()
  end subroutine

  subroutine test_power_preserves_unitarity()
    integer, parameter :: n = 5
    type(UF_type) :: uf
    complex(dp) :: result(n+1,n+1)

    call test_start('U^r unitary')
    call make_uf_from_h(n, uf)
    call find_power_r(uf, 5, n+1, result)
    call assert_matrix_unitary(result, n+1, 1e-10_dp, 'U^5 unitary')
    call test_pass()
  end subroutine

  subroutine test_evolution_step_unitary()
    integer, parameter :: n = 5
    complex(dp) :: H0(n+1,n+1), Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1)
    complex(dp) :: I_mat(n+1,n+1), result(n+1,n+1)
    real(dp) :: T, t_delta, omega
    integer :: steps

    call test_start('evolution_step_unitary')

    call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
    call create_spin_xyz_operators(n, Sz, Sx, Sy)

    T = 1.0_dp
    steps = 10
    t_delta = T / real(steps, dp)
    omega = 2.0_dp * PI / T

    call eye_complex(n+1, I_mat)
    call evolution_T_step(I_mat, 0.0_dp, T, 0.0_dp, 0.0_dp, 0.0_dp, &
                          H0, Sx, Sy, Sz, n, omega, 1, t_delta, steps, result)
    call assert_matrix_unitary(result, n+1, 1e-10_dp, 'evolution step unitary')
    call test_pass()
  end subroutine

  subroutine test_evolution_h0_trotter()
    integer, parameter :: n = 5
    complex(dp) :: H0(n+1,n+1), Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1)
    complex(dp) :: I_mat(n+1,n+1), result(n+1,n+1), expected(n+1,n+1)
    real(dp) :: T, t_delta, omega
    integer :: steps

    call test_start('h=0 Trotter ~ exp(-i*H0*T)')

    call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
    call create_spin_xyz_operators(n, Sz, Sx, Sy)

    T = 1.0_dp
    steps = 50  ! high accuracy
    t_delta = T / real(steps, dp)
    omega = 2.0_dp * PI / T

    call eye_complex(n+1, I_mat)
    call evolution_T_step(I_mat, 0.0_dp, T, 0.0_dp, 0.0_dp, 0.0_dp, &
                          H0, Sx, Sy, Sz, n, omega, 1, t_delta, steps, result)
    call expm_hermitian(H0, n+1, T, expected)
    call assert_matrix_close(result, expected, n+1, 1e-3_dp, 'Trotter ~ exact')
    call test_pass()
  end subroutine

  subroutine test_unitary_T_is_unitary()
    integer, parameter :: n = 5
    type(SimulationParams_type) :: params
    complex(dp) :: H0(n+1,n+1), U_T(n+1,n+1)
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1), kick(n+1,n+1)

    call test_start('U_T unitary')
    params = make_params(n, 2)
    call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    call create_kick_operator(params%phi_kick_phase, Sx, n, kick)
    call calculate_unitary_T(0.0_dp, params, H0, Sz, Sx, Sy, kick, n, U_T)
    call assert_matrix_unitary(U_T, n+1, 1e-10_dp, 'U_T unitary')
    call test_pass()
  end subroutine

  subroutine test_unitary_T_det_magnitude()
    integer, parameter :: n = 2
    type(SimulationParams_type) :: params
    complex(dp) :: H0(n+1,n+1), U_T(n+1,n+1)
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1), kick(n+1,n+1)
    complex(dp) :: det_val
    ! For a small matrix, compute det via cofactor expansion
    ! 3x3: det = a(ei-fh) - b(di-fg) + c(dh-eg)

    call test_start('|det(U_T)| = 1')
    params = make_params(n, 2)
    params%h = 0.1_dp
    params%theta = 0.1_dp
    call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    call create_kick_operator(params%phi_kick_phase, Sx, n, kick)
    call calculate_unitary_T(0.1_dp, params, H0, Sz, Sx, Sy, kick, n, U_T)

    det_val = U_T(1,1) * (U_T(2,2)*U_T(3,3) - U_T(2,3)*U_T(3,2)) &
            - U_T(1,2) * (U_T(2,1)*U_T(3,3) - U_T(2,3)*U_T(3,1)) &
            + U_T(1,3) * (U_T(2,1)*U_T(3,2) - U_T(2,2)*U_T(3,1))

    call assert_close(abs(det_val), 1.0_dp, 1e-8_dp, '|det|=1')
    call test_pass()
  end subroutine

  subroutine test_at_time_zero_identity()
    integer, parameter :: n = 5
    type(SimulationParams_type) :: params
    type(UF_type) :: uf
    complex(dp) :: H0(n+1,n+1), U_T(n+1,n+1), result(n+1,n+1), I_mat(n+1,n+1)
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1), kick(n+1,n+1)

    call test_start('U(t=0) = I')
    params = make_params(n, 2)
    call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    call create_kick_operator(params%phi_kick_phase, Sx, n, kick)
    call calculate_unitary_T(0.0_dp, params, H0, Sz, Sx, Sy, kick, n, U_T)
    call build_uf(U_T, n+1, uf)
    call calculate_unitary_at_time(0.0_dp, 0, params, H0, uf, Sz, Sx, Sy, kick, n, result)
    call eye_complex(n+1, I_mat)
    call assert_matrix_close(result, I_mat, n+1, 1e-10_dp, 'U(0)=I')
    call test_pass()
  end subroutine

  subroutine test_at_time_nu_equals_T()
    integer, parameter :: n = 5, nu = 2
    type(SimulationParams_type) :: params
    type(UF_type) :: uf
    complex(dp) :: H0(n+1,n+1), U_T(n+1,n+1), result(n+1,n+1)
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1), kick(n+1,n+1)

    call test_start('U(nu) = U_T')
    params = make_params(n, nu)
    call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    call create_kick_operator(params%phi_kick_phase, Sx, n, kick)
    call calculate_unitary_T(0.0_dp, params, H0, Sz, Sx, Sy, kick, n, U_T)
    call build_uf(U_T, n+1, uf)
    call calculate_unitary_at_time(0.0_dp, nu, params, H0, uf, Sz, Sx, Sy, kick, n, result)
    call assert_matrix_close(result, U_T, n+1, 1e-8_dp, 'U(nu)=U_T')
    call test_pass()
  end subroutine

  subroutine test_at_time_result_unitary()
    integer, parameter :: n = 5
    type(SimulationParams_type) :: params
    type(UF_type) :: uf
    complex(dp) :: H0(n+1,n+1), U_T(n+1,n+1), result(n+1,n+1)
    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1), kick(n+1,n+1)

    call test_start('U(t) unitary')
    params = make_params(n, 2)
    params%h = 0.1_dp
    call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    call create_kick_operator(params%phi_kick_phase, Sx, n, kick)
    call calculate_unitary_T(0.1_dp, params, H0, Sz, Sx, Sy, kick, n, U_T)
    call build_uf(U_T, n+1, uf)
    call calculate_unitary_at_time(0.1_dp, 4, params, H0, uf, Sz, Sx, Sy, kick, n, result)
    call assert_matrix_unitary(result, n+1, 1e-6_dp, 'U(t=4) unitary')
    call test_pass()
  end subroutine

end program test_evolution
