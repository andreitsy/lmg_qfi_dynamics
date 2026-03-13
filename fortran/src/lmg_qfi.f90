module lmg_qfi_mod
  use lmg_constants, only: dp, CZERO
  use lmg_types, only: QFIInformation_type, UF_type, SimulationParams_type
  use lmg_operators, only: create_spin_xyz_operators
  use lmg_evolution, only: calculate_unitary_at_time
  implicit none
  private
  public :: dketa_t, quantum_fisher_information, calculate_error_estimation, process_time_point

contains

  !> Finite difference derivative: dket = (ket_p - ket_m) / (2*delta)
  subroutine dketa_t(ket_p, ket_m, delta, dim, dket)
    complex(dp), intent(in) :: ket_p(dim), ket_m(dim)
    real(dp), intent(in) :: delta
    integer, intent(in) :: dim
    complex(dp), intent(out) :: dket(dim)

    dket = (ket_p - ket_m) / (2.0_dp * delta)
  end subroutine dketa_t

  !> QFI = 4 * ( <dket|dket> - |<ket|dket>|^2 )
  function quantum_fisher_information(dket, ket, dim) result(qfi)
    complex(dp), intent(in) :: dket(dim), ket(dim)
    integer, intent(in) :: dim
    real(dp) :: qfi

    complex(dp) :: term1, overlap
    real(dp) :: term2

    ! <dket|dket>
    term1 = dot_product(dket, dket)  ! Fortran dot_product conjugates first arg
    ! <ket|dket>
    overlap = dot_product(ket, dket)
    term2 = abs(overlap)**2

    qfi = 4.0_dp * real(term1 - term2, dp)
  end function quantum_fisher_information

  !> Error estimation: |<dket|ket> + <ket|dket>|
  function calculate_error_estimation(dket, ket, dim) result(error_est)
    complex(dp), intent(in) :: dket(dim), ket(dim)
    integer, intent(in) :: dim
    real(dp) :: error_est

    complex(dp) :: term

    term = dot_product(dket, ket) + dot_product(ket, dket)
    error_est = abs(term)
  end function calculate_error_estimation

  !> Process a single time point: compute QFI and magnetizations.
  subroutine process_time_point(time, params, H0, uf, uf_p, uf_m, &
                                 init_state, Sz, Sx, Sy, n, result)
    integer, intent(in) :: time, n
    type(SimulationParams_type), intent(in) :: params
    complex(dp), intent(in) :: H0(n+1,n+1)
    type(UF_type), intent(in) :: uf, uf_p, uf_m
    complex(dp), intent(in) :: init_state(n+1)
    complex(dp), intent(in) :: Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1)
    type(QFIInformation_type), intent(out) :: result

    complex(dp) :: U_t(n+1,n+1), U_tp(n+1,n+1), U_tm(n+1,n+1)
    complex(dp) :: ket(n+1), ket_p(n+1), ket_m(n+1), dket(n+1)
    complex(dp) :: temp_vec(n+1)
    real(dp) :: epsilon, qfi_raw
    integer :: dim

    dim = n + 1
    epsilon = params%h  ! will be overridden — use the FD epsilon
    ! The finite-difference epsilon from the Python code is 1e-(dps//2) ~ 1e-7
    ! We pass it via the EPSILON_FD constant
    block
      use lmg_constants, only: EPSILON_FD
      epsilon = EPSILON_FD
    end block

    ! Compute unitaries at this time
    call calculate_unitary_at_time(params%h, time, params, H0, uf, n, U_t)
    call calculate_unitary_at_time(params%h + epsilon, time, params, H0, uf_p, n, U_tp)
    call calculate_unitary_at_time(params%h - epsilon, time, params, H0, uf_m, n, U_tm)

    ! Evolve states
    ket = matmul(U_t, init_state)
    ket_p = matmul(U_tp, init_state)
    ket_m = matmul(U_tm, init_state)

    ! Compute derivative ket
    call dketa_t(ket_p, ket_m, epsilon, dim, dket)

    ! QFI
    qfi_raw = quantum_fisher_information(dket, ket, dim)

    ! Magnetizations: m_alpha = <ket|S_alpha|ket> / N
    temp_vec = matmul(Sx, ket)
    result%m_x = real(dot_product(ket, temp_vec), dp) / real(params%N, dp)

    temp_vec = matmul(Sy, ket)
    result%m_y = real(dot_product(ket, temp_vec), dp) / real(params%N, dp)

    temp_vec = matmul(Sz, ket)
    result%m_z = real(dot_product(ket, temp_vec), dp) / real(params%N, dp)

    result%qfi_raw = qfi_raw
    result%qfi = qfi_raw / (real(params%N, dp)**2 * real(time, dp)**2)
    result%time = time
  end subroutine process_time_point

end module lmg_qfi_mod
