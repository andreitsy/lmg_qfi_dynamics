program test_simulation
  use lmg_constants, only: dp, CZERO, CONE
  use lmg_types, only: SimulationParams_type, STATE_GS_PHYS, STATE_GS_CAT, &
                        STATE_PHYS, STATE_CAT_SUM
  use lmg_operators, only: create_hamiltonian_h0
  use lmg_simulation, only: generate_time_interval, build_initial_state
  use test_utils
  implicit none

  call test_suite_start('lmg_simulation')

  call test_time_interval_sorted()
  call test_time_interval_starts_at_one()
  call test_time_interval_unique()
  call test_initial_state_phys_normalized()
  call test_initial_state_cat_sum_normalized()
  call test_initial_state_gs_cat_normalized()
  call test_initial_state_gs_phys_normalized()

  call test_suite_end()
  if (tests_failed > 0) stop 1

contains

  subroutine test_time_interval_sorted()
    integer, allocatable :: tp(:)
    integer :: total, i

    call test_start('time_interval sorted')
    call generate_time_interval(50, 3, tp, total)
    do i = 2, total
      call assert_true(tp(i) >= tp(i-1), 'sorted')
    end do
    deallocate(tp)
    call test_pass()
  end subroutine

  subroutine test_time_interval_starts_at_one()
    integer, allocatable :: tp(:)
    integer :: total

    call test_start('time_interval starts at 1')
    call generate_time_interval(50, 3, tp, total)
    call assert_true(tp(1) == 1, 'starts at 1')
    deallocate(tp)
    call test_pass()
  end subroutine

  subroutine test_time_interval_unique()
    integer, allocatable :: tp(:)
    integer :: total, i

    call test_start('time_interval unique')
    call generate_time_interval(50, 3, tp, total)
    do i = 2, total
      call assert_true(tp(i) > tp(i-1), 'unique')
    end do
    deallocate(tp)
    call test_pass()
  end subroutine

  subroutine test_initial_state_phys_normalized()
    integer, parameter :: n = 5
    complex(dp) :: H0(n+1, n+1), init_state(n+1)
    real(dp) :: norm_sq

    call test_start('PHYS state normalized')
    call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
    call build_initial_state(STATE_PHYS, H0, n+1, n, init_state)
    norm_sq = real(dot_product(init_state, init_state), dp)
    call assert_close(norm_sq, 1.0_dp, 1e-12_dp, '|psi|^2=1')
    call test_pass()
  end subroutine

  subroutine test_initial_state_cat_sum_normalized()
    integer, parameter :: n = 5
    complex(dp) :: H0(n+1, n+1), init_state(n+1)
    real(dp) :: norm_sq

    call test_start('CAT_SUM state normalized')
    call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
    call build_initial_state(STATE_CAT_SUM, H0, n+1, n, init_state)
    norm_sq = real(dot_product(init_state, init_state), dp)
    call assert_close(norm_sq, 1.0_dp, 1e-12_dp, '|psi|^2=1')
    call test_pass()
  end subroutine

  subroutine test_initial_state_gs_cat_normalized()
    integer, parameter :: n = 5
    complex(dp) :: H0(n+1, n+1), init_state(n+1)
    real(dp) :: norm_sq

    call test_start('GS_CAT state normalized')
    call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
    call build_initial_state(STATE_GS_CAT, H0, n+1, n, init_state)
    norm_sq = real(dot_product(init_state, init_state), dp)
    call assert_close(norm_sq, 1.0_dp, 1e-12_dp, '|psi|^2=1')
    call test_pass()
  end subroutine

  subroutine test_initial_state_gs_phys_normalized()
    integer, parameter :: n = 5
    complex(dp) :: H0(n+1, n+1), init_state(n+1)
    real(dp) :: norm_sq

    call test_start('GS_PHYS state normalized')
    call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
    call build_initial_state(STATE_GS_PHYS, H0, n+1, n, init_state)
    norm_sq = real(dot_product(init_state, init_state), dp)
    call assert_close(norm_sq, 1.0_dp, 1e-12_dp, '|psi|^2=1')
    call test_pass()
  end subroutine

end program test_simulation
