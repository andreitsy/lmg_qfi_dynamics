module lmg_evolution
  use lmg_constants, only: dp, CI, CZERO, CONE, PI
  use lmg_types, only: UF_type, SimulationParams_type
  use lmg_linalg, only: eig, matrix_inverse, eye_complex, expm_hermitian
  use lmg_operators, only: create_v_operator  ! kept for module dependency; inlined in hot path
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

    complex(dp) :: V_mat(n+1,n+1), U_step(n+1,n+1), S_dir(n+1,n+1)
    real(dp) :: t_start, t_end, t_k, sinusoidal
    integer :: i, dim

    dim = n + 1
    t_start = T * real(p - 1, dp)
    t_end = T * real(p, dp)

    result = floquet_unitary

    ! Fast path: when h=0, V(t) = H0 for all t, so all Trotter steps
    ! produce the same exp(-i*dt*H0). Their product equals exp(-i*T_period*H0).
    if (abs(h) < tiny(1.0_dp)) then
      call expm_hermitian(H0, dim, t_end - t_start, U_step)
      result = matmul(U_step, result)
      return
    end if

    ! Precompute time-independent direction matrix once:
    ! S_dir = sin(theta)*cos(varphi)*Sx + sin(theta)*sin(varphi)*Sy + cos(theta)*Sz
    S_dir = sin(theta) * cos(varphi) * Sx &
          + sin(theta) * sin(varphi) * Sy &
          + cos(theta) * Sz

    do i = 0, steps - 1
      t_k = t_start + real(i, dp) * (t_end - t_start) / real(steps - 1, dp)
      sinusoidal = sin(t_k * omega + phi_0)
      V_mat = H0 + h * sinusoidal * S_dir
      call expm_hermitian(V_mat, dim, t_delta, U_step)
      result = matmul(U_step, result)
    end do
  end subroutine evolution_T_step

  !> Compute U^r via eigendecomposition: U^r = P * diag(lambda^r) * P^{-1}
  subroutine find_power_r(uf, r, dim, result)
    type(UF_type), intent(in) :: uf
    integer, intent(in) :: r, dim
    complex(dp), intent(out) :: result(dim, dim)

    complex(dp) :: scaled(dim, dim)
    integer :: i

    if (r <= 0) then
      call eye_complex(dim, result)
      return
    end if

    ! Scale columns of U by eigenvalue^r instead of forming a diagonal matrix
    ! This replaces matmul(U, matmul(diag, U_inv)) with one matmul
    do i = 1, dim
      scaled(:, i) = uf%U(:, i) * (uf%eigenvalues(i) ** r)
    end do

    result = matmul(scaled, uf%U_inv)
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
  !> Accepts precomputed spin operators and kick to avoid redundant construction.
  subroutine calculate_unitary_T(h, params, H0, Sz, Sx, Sy, kick, n, U_T)
    real(dp), intent(in) :: h
    type(SimulationParams_type), intent(in) :: params
    complex(dp), intent(in) :: H0(n+1, n+1)
    complex(dp), intent(in) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1)
    complex(dp), intent(in) :: kick(n+1,n+1)
    integer, intent(in) :: n
    complex(dp), intent(out) :: U_T(n+1, n+1)

    complex(dp) :: temp(n+1,n+1), exp_H0_T(n+1,n+1)
    real(dp) :: t_delta, omega
    integer :: p, dim

    dim = n + 1
    t_delta = params%T / real(params%steps_floquet_unitary, dp)
    omega = 2.0_dp * PI / (real(params%freq, dp) * params%T)

    call eye_complex(dim, U_T)

    ! Fast path: when h=0, exp(-i*T*H0) is the same for every period.
    ! Precompute once and reuse across all freq kicks.
    if (abs(h) < tiny(1.0_dp)) then
      call expm_hermitian(H0, dim, params%T, exp_H0_T)
      do p = 1, params%freq
        U_T = matmul(kick, matmul(exp_H0_T, U_T))
      end do
      return
    end if

    do p = 1, params%freq
      call evolution_T_step(U_T, h, params%T, params%varphi, params%theta, &
                            params%phi_0, H0, Sx, Sy, Sz, n, omega, p, t_delta, &
                            params%steps_floquet_unitary, temp)
      U_T = matmul(kick, temp)
    end do
  end subroutine calculate_unitary_T

  !> Calculate unitary at arbitrary discrete time.
  !> time is measured in kick periods: U(time) = U_F^r * extra kicks
  !> Accepts precomputed spin operators and kick to avoid redundant construction.
  subroutine calculate_unitary_at_time(h, time, params, H0, uf, Sz, Sx, Sy, kick, n, result)
    real(dp), intent(in) :: h
    integer, intent(in) :: time
    type(SimulationParams_type), intent(in) :: params
    complex(dp), intent(in) :: H0(n+1, n+1)
    type(UF_type), intent(in) :: uf
    complex(dp), intent(in) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1)
    complex(dp), intent(in) :: kick(n+1,n+1)
    integer, intent(in) :: n
    complex(dp), intent(out) :: result(n+1, n+1)

    complex(dp) :: temp(n+1,n+1)
    real(dp) :: t_delta, omega
    integer :: r, p, dim, nu

    dim = n + 1
    nu = params%freq

    t_delta = params%T / real(params%steps_floquet_unitary, dp)
    omega = 2.0_dp * PI / (real(nu, dp) * params%T)

    r = time / nu  ! integer division
    call find_power_r(uf, r, dim, result)

    ! Extra intervals beyond r*nu
    do p = r * nu + 1, time
      call evolution_T_step(result, h, params%T, params%varphi, params%theta, &
                            params%phi_0, H0, Sx, Sy, Sz, n, omega, p, t_delta, &
                            params%steps_floquet_unitary, temp)
      result = matmul(kick, temp)
    end do
  end subroutine calculate_unitary_at_time

end module lmg_evolution
