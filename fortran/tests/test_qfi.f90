program test_qfi
  use lmg_constants, only: dp, CI, CZERO, CONE, PI
  use lmg_qfi_mod, only: dketa_t, quantum_fisher_information, calculate_error_estimation
  use lmg_operators, only: create_spin_xyz_operators
  use test_utils
  implicit none

  call test_suite_start('lmg_qfi')

  call test_dketa_basic()
  call test_dketa_zero_difference()
  call test_qfi_zero_stationary()
  call test_qfi_non_negative()
  call test_qfi_orthogonal()
  call test_qfi_parallel()
  call test_qfi_phase_invariance()
  call test_error_orthogonal()
  call test_error_parallel()

  call test_suite_end()
  if (tests_failed > 0) stop 1

contains

  subroutine test_dketa_basic()
    complex(dp) :: ket_p(3), ket_m(3), dket(3), expected(3)

    call test_start('dketa_t basic')

    ket_p = [(1.0_dp, 0.0_dp), (2.0_dp, 0.0_dp), (3.0_dp, 0.0_dp)]
    ket_m = [(0.0_dp, 0.0_dp), (1.0_dp, 0.0_dp), (2.0_dp, 0.0_dp)]
    expected = [(1.0_dp, 0.0_dp), (1.0_dp, 0.0_dp), (1.0_dp, 0.0_dp)]

    call dketa_t(ket_p, ket_m, 0.5_dp, 3, dket)

    call assert_close_complex(dket(1), expected(1), 1e-10_dp, 'dket(1)')
    call assert_close_complex(dket(2), expected(2), 1e-10_dp, 'dket(2)')
    call assert_close_complex(dket(3), expected(3), 1e-10_dp, 'dket(3)')
    call test_pass()
  end subroutine

  subroutine test_dketa_zero_difference()
    complex(dp) :: ket(3), dket(3)
    integer :: i

    call test_start('dketa_t zero when equal')

    ket = [(1.0_dp, 0.0_dp), (2.0_dp, 0.0_dp), (3.0_dp, 0.0_dp)]
    call dketa_t(ket, ket, 0.1_dp, 3, dket)

    do i = 1, 3
      call assert_close(abs(dket(i)), 0.0_dp, 1e-10_dp, 'dket=0')
    end do
    call test_pass()
  end subroutine

  subroutine test_qfi_zero_stationary()
    complex(dp) :: ket(3), dket(3)
    real(dp) :: qfi

    call test_start('QFI=0 for stationary')

    ket = [(1.0_dp, 0.0_dp), (0.0_dp, 0.0_dp), (0.0_dp, 0.0_dp)]
    dket = CZERO

    qfi = quantum_fisher_information(dket, ket, 3)
    call assert_close(qfi, 0.0_dp, 1e-10_dp, 'QFI=0')
    call test_pass()
  end subroutine

  subroutine test_qfi_non_negative()
    complex(dp) :: ket(2), dket(2)
    real(dp) :: qfi

    call test_start('QFI >= 0')

    ket = [(0.6_dp, 0.0_dp), (0.8_dp, 0.0_dp)]
    dket = [(0.1_dp, 0.0_dp), (-0.2_dp, 0.0_dp)]

    qfi = quantum_fisher_information(dket, ket, 2)
    call assert_true(qfi >= -1e-12_dp, 'QFI non-negative')
    call test_pass()
  end subroutine

  subroutine test_qfi_orthogonal()
    complex(dp) :: ket(2), dket(2)
    real(dp) :: qfi

    call test_start('QFI=4 orthogonal')

    ket = [(1.0_dp, 0.0_dp), (0.0_dp, 0.0_dp)]
    dket = [(0.0_dp, 0.0_dp), (1.0_dp, 0.0_dp)]

    qfi = quantum_fisher_information(dket, ket, 2)
    call assert_close(qfi, 4.0_dp, 1e-10_dp, 'QFI=4')
    call test_pass()
  end subroutine

  subroutine test_qfi_parallel()
    complex(dp) :: ket(2), dket(2)
    real(dp) :: qfi

    call test_start('QFI=0 parallel')

    ket = [(1.0_dp, 0.0_dp), (0.0_dp, 0.0_dp)]
    dket = [(0.5_dp, 0.0_dp), (0.0_dp, 0.0_dp)]

    qfi = quantum_fisher_information(dket, ket, 2)
    call assert_close(qfi, 0.0_dp, 1e-10_dp, 'QFI=0 parallel')
    call test_pass()
  end subroutine

  subroutine test_qfi_phase_invariance()
    complex(dp) :: ket(2), dket(2), phase
    real(dp) :: qfi_orig, qfi_phased

    call test_start('QFI phase invariant')

    ket = [(0.6_dp, 0.0_dp), (0.8_dp, 0.0_dp)]
    dket = [(0.1_dp, 0.0_dp), (-0.1_dp, 0.0_dp)]
    qfi_orig = quantum_fisher_information(dket, ket, 2)

    phase = exp(CI * PI / 4.0_dp)
    qfi_phased = quantum_fisher_information(phase * dket, phase * ket, 2)

    call assert_close(qfi_orig, qfi_phased, 1e-10_dp, 'QFI invariant')
    call test_pass()
  end subroutine

  subroutine test_error_orthogonal()
    complex(dp) :: ket(2), dket(2)
    real(dp) :: err

    call test_start('error=0 orthogonal')

    ket = [(1.0_dp, 0.0_dp), (0.0_dp, 0.0_dp)]
    dket = [(0.0_dp, 0.0_dp), (0.1_dp, 0.0_dp)]

    err = calculate_error_estimation(dket, ket, 2)
    call assert_close(err, 0.0_dp, 1e-10_dp, 'error=0')
    call test_pass()
  end subroutine

  subroutine test_error_parallel()
    complex(dp) :: ket(2), dket(2)
    real(dp) :: err

    call test_start('error=0.2 parallel')

    ket = [(1.0_dp, 0.0_dp), (0.0_dp, 0.0_dp)]
    dket = [(0.1_dp, 0.0_dp), (0.0_dp, 0.0_dp)]

    err = calculate_error_estimation(dket, ket, 2)
    call assert_close(err, 0.2_dp, 1e-10_dp, 'error=0.2')
    call test_pass()
  end subroutine

end program test_qfi
