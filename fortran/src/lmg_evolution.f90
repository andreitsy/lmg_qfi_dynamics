module lmg_evolution
  use lmg_constants, only: dp, CI, CZERO, CONE, PI
  use lmg_types, only: UF_type, SimulationParams_type
  use lmg_linalg, only: eig, matrix_inverse, eye_complex, expm_hermitian
  use lmg_operators, only: create_spin_xyz_operators, create_kick_operator, create_v_operator
  implicit none
  private
  public :: evolution_T_step, find_power_r, build_uf, calculate_unitary_T, calculate_unitary_at_time

contains

  !> One Trotter step over a single period [T*(p-1), T*p].
  !> Accumulates U_step * floquet_unitary for each time slice.
  subroutine evolution_T_step(floquet_unitary, h, T, varphi, theta, phi_0, &
                               H0, Sx, Sy, Sz, n, omega, p, t_delta, steps, result)
    complex(dp), intent(in) :: floquet_unitary(n+1,n+1)
    real(dp), intent(in) :: h, T, varphi, theta, phi_0, omega, t_delta
    complex(dp), intent(in) :: H0(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1), Sz(n+1,n+1)
    integer, intent(in) :: n, p, steps
    complex(dp), intent(out) :: result(n+1,n+1)

    complex(dp) :: V_mat(n+1,n+1), U_step(n+1,n+1)
    real(dp) :: t_start, t_end, t_k
    integer :: i, dim

    dim = n + 1
    t_start = T * real(p - 1, dp)
    t_end = T * real(p, dp)

    result = floquet_unitary

    do i = 0, steps - 1
      t_k = t_start + real(i, dp) * (t_end - t_start) / real(steps - 1, dp)
      call create_v_operator(H0, Sx, Sy, Sz, n, omega, phi_0, h, t_k, theta, varphi, V_mat)
      ! U_step = exp(-i * t_delta * V)
      ! V is Hermitian, so we can use expm for Hermitian
      call expm_hermitian(V_mat, dim, t_delta, U_step)
      result = matmul(U_step, result)
    end do
  end subroutine evolution_T_step

  !> Compute U^r via eigendecomposition: U^r = P * diag(lambda^r) * P^{-1}
  subroutine find_power_r(uf, r, dim, result)
    type(UF_type), intent(in) :: uf
    integer, intent(in) :: r, dim
    complex(dp), intent(out) :: result(dim, dim)

    complex(dp) :: diag_r(dim, dim)
    integer :: i

    if (r <= 0) then
      call eye_complex(dim, result)
      return
    end if

    diag_r = CZERO
    do i = 1, dim
      diag_r(i,i) = uf%eigenvalues(i) ** r
    end do

    result = matmul(uf%U, matmul(diag_r, uf%U_inv))
  end subroutine find_power_r

  !> Build UF_type from a unitary matrix: eigendecompose and store inverse.
  subroutine build_uf(U_T, dim, uf)
    complex(dp), intent(in) :: U_T(dim, dim)
    integer, intent(in) :: dim
    type(UF_type), intent(out) :: uf

    uf%dim = dim
    if (allocated(uf%eigenvalues)) deallocate(uf%eigenvalues)
    if (allocated(uf%U)) deallocate(uf%U)
    if (allocated(uf%U_inv)) deallocate(uf%U_inv)

    allocate(uf%eigenvalues(dim))
    allocate(uf%U(dim, dim))
    allocate(uf%U_inv(dim, dim))

    call eig(U_T, dim, uf%eigenvalues, uf%U)
    call matrix_inverse(uf%U, dim, uf%U_inv)
  end subroutine build_uf

  !> Calculate Floquet unitary for one complete period (nu kicks).
  subroutine calculate_unitary_T(h, params, H0, n, U_T)
    real(dp), intent(in) :: h
    type(SimulationParams_type), intent(in) :: params
    complex(dp), intent(in) :: H0(n+1, n+1)
    integer, intent(in) :: n
    complex(dp), intent(out) :: U_T(n+1, n+1)

    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1)
    complex(dp) :: kick(n+1,n+1), temp(n+1,n+1)
    real(dp) :: t_delta, omega
    integer :: p, dim

    dim = n + 1
    call create_spin_xyz_operators(n, Sz, Sx, Sy)

    t_delta = params%T / real(params%steps_floquet_unitary, dp)
    omega = 2.0_dp * PI / (real(params%freq, dp) * params%T)

    call eye_complex(dim, U_T)
    call create_kick_operator(params%phi_kick_phase, Sx, n, kick)

    do p = 1, params%freq
      call evolution_T_step(U_T, h, params%T, params%varphi, params%theta, &
                            params%phi_0, H0, Sx, Sy, Sz, n, omega, p, t_delta, &
                            params%steps_floquet_unitary, temp)
      U_T = matmul(kick, temp)
    end do
  end subroutine calculate_unitary_T

  !> Calculate unitary at arbitrary discrete time.
  !> time is measured in kick periods: U(time) = U_F^r * extra kicks
  subroutine calculate_unitary_at_time(h, time, params, H0, uf, n, result)
    real(dp), intent(in) :: h
    integer, intent(in) :: time
    type(SimulationParams_type), intent(in) :: params
    complex(dp), intent(in) :: H0(n+1, n+1)
    type(UF_type), intent(in) :: uf
    integer, intent(in) :: n
    complex(dp), intent(out) :: result(n+1, n+1)

    complex(dp) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1)
    complex(dp) :: kick(n+1,n+1), temp(n+1,n+1)
    real(dp) :: t_delta, omega
    integer :: r, p, dim, nu

    dim = n + 1
    nu = params%freq

    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    t_delta = params%T / real(params%steps_floquet_unitary, dp)
    omega = 2.0_dp * PI / (real(nu, dp) * params%T)

    r = time / nu  ! integer division
    call find_power_r(uf, r, dim, result)
    call create_kick_operator(params%phi_kick_phase, Sx, n, kick)

    ! Extra intervals beyond r*nu
    do p = r * nu + 1, time
      call evolution_T_step(result, h, params%T, params%varphi, params%theta, &
                            params%phi_0, H0, Sx, Sy, Sz, n, omega, p, t_delta, &
                            params%steps_floquet_unitary, temp)
      result = matmul(kick, temp)
    end do
  end subroutine calculate_unitary_at_time

end module lmg_evolution
