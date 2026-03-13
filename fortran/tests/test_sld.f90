program test_sld
  use lmg_constants, only: dp, CI, CZERO, CONE
  use lmg_types, only: SLDInformation_type
  use lmg_sld, only: compute_sld_matrix, sld_expectation, sld_squared_expectation, compute_sld_info
  use lmg_qfi_mod, only: quantum_fisher_information
  use test_utils
  implicit none

  call test_suite_start('lmg_sld')

  call test_sld_matrix_hermitian()
  call test_sld_equation()
  call test_sld_matrix_zero_derivative()
  call test_sld_expectation_orthogonal()
  call test_sld_expectation_formula()
  call test_sld_expectation_nonzero_real()
  call test_sld_squared_matches_qfi()
  call test_sld_squared_non_negative()
  call test_sld_squared_zero_derivative()
  call test_sld_squared_orthogonal()
  call test_sld_squared_phase_invariance()
  call test_compute_sld_info_time()
  call test_compute_sld_info_normalization()
  call test_compute_sld_info_l_exp_near_zero()

  call test_suite_end()
  if (tests_failed > 0) stop 1

contains

  subroutine test_sld_matrix_hermitian()
    integer, parameter :: dim = 6
    complex(dp) :: ket(dim), dket(dim), L(dim, dim)

    call test_start('SLD matrix Hermitian')

    ket = CZERO; ket(1) = CONE
    dket = CZERO; dket(2) = (0.1_dp, 0.0_dp)

    call compute_sld_matrix(dket, ket, dim, L)
    call assert_matrix_hermitian(L, dim, 1e-12_dp, 'L Hermitian')
    call test_pass()
  end subroutine

  subroutine test_sld_equation()
    integer, parameter :: dim = 3
    complex(dp) :: ket(dim), dket(dim), L(dim, dim)
    complex(dp) :: rho(dim,dim), d_rho(dim,dim), lhs(dim,dim)
    integer :: i, j

    call test_start('SLD equation')

    ket = CZERO; ket(1) = CONE
    dket = CZERO; dket(2) = (0.3_dp, 0.0_dp)

    call compute_sld_matrix(dket, ket, dim, L)

    ! rho = |ket><ket|
    do j = 1, dim
      do i = 1, dim
        rho(i,j) = ket(i) * conjg(ket(j))
        d_rho(i,j) = dket(i) * conjg(ket(j)) + ket(i) * conjg(dket(j))
      end do
    end do

    ! lhs = (rho*L + L*rho) / 2
    lhs = 0.5_dp * (matmul(rho, L) + matmul(L, rho))
    call assert_matrix_close(lhs, d_rho, dim, 1e-12_dp, 'SLD equation')
    call test_pass()
  end subroutine

  subroutine test_sld_matrix_zero_derivative()
    integer, parameter :: dim = 4
    complex(dp) :: ket(dim), dket(dim), L(dim, dim), zero_mat(dim, dim)

    call test_start('SLD zero for zero deriv')

    ket = CZERO; ket(1) = CONE
    dket = CZERO
    zero_mat = CZERO

    call compute_sld_matrix(dket, ket, dim, L)
    call assert_matrix_close(L, zero_mat, dim, 1e-12_dp, 'L=0')
    call test_pass()
  end subroutine

  subroutine test_sld_expectation_orthogonal()
    integer, parameter :: dim = 6
    complex(dp) :: ket(dim), dket(dim)
    real(dp) :: result

    call test_start('<L>=0 orthogonal')

    ket = CZERO; ket(1) = CONE
    dket = CZERO; dket(2) = (0.5_dp, 0.0_dp)

    result = sld_expectation(dket, ket, dim)
    call assert_close(abs(result), 0.0_dp, 1e-12_dp, '<L>=0')
    call test_pass()
  end subroutine

  subroutine test_sld_expectation_formula()
    integer, parameter :: dim = 2
    complex(dp) :: ket(dim), dket(dim), alpha
    real(dp) :: result, expected

    call test_start('<L> = 4*Re(<ket|dket>)')

    ket = [(0.6_dp, 0.0_dp), (0.8_dp, 0.0_dp)]
    dket = [(0.1_dp, 0.0_dp), (0.2_dp, 0.0_dp)]

    result = sld_expectation(dket, ket, dim)
    alpha = dot_product(ket, dket)
    expected = 4.0_dp * real(alpha, dp)

    call assert_close(result, expected, 1e-12_dp, '<L> formula')
    call test_pass()
  end subroutine

  subroutine test_sld_expectation_nonzero_real()
    integer, parameter :: dim = 2
    complex(dp) :: ket(dim), dket(dim)
    real(dp) :: result

    call test_start('<L>=0.4 real overlap')

    ket = [CONE, CZERO]
    dket = [(0.1_dp, 0.0_dp), CZERO]

    result = sld_expectation(dket, ket, dim)
    call assert_close(result, 0.4_dp, 1e-12_dp, '<L>=0.4')
    call test_pass()
  end subroutine

  subroutine test_sld_squared_matches_qfi()
    integer, parameter :: dim = 6
    complex(dp) :: ket(dim), dket(dim)
    real(dp) :: L2, qfi

    call test_start('<L^2> = QFI')

    ket = CZERO; ket(1) = CONE
    dket = CZERO; dket(2) = (0.1_dp, 0.0_dp)

    L2 = sld_squared_expectation(dket, ket, dim)
    qfi = quantum_fisher_information(dket, ket, dim)

    call assert_close(L2, qfi, 1e-12_dp, '<L^2>=QFI')
    call test_pass()
  end subroutine

  subroutine test_sld_squared_non_negative()
    integer, parameter :: dim = 2
    complex(dp) :: ket(dim), dket(dim)
    real(dp) :: result

    call test_start('<L^2> >= 0')

    ket = [(0.6_dp, 0.0_dp), (0.8_dp, 0.0_dp)]
    dket = [(0.1_dp, 0.0_dp), (-0.2_dp, 0.0_dp)]

    result = sld_squared_expectation(dket, ket, dim)
    call assert_true(result >= -1e-12_dp, '<L^2> non-negative')
    call test_pass()
  end subroutine

  subroutine test_sld_squared_zero_derivative()
    integer, parameter :: dim = 2
    complex(dp) :: ket(dim), dket(dim)
    real(dp) :: result

    call test_start('<L^2>=0 zero deriv')

    ket = [CONE, CZERO]
    dket = CZERO

    result = sld_squared_expectation(dket, ket, dim)
    call assert_close(abs(result), 0.0_dp, 1e-12_dp, '<L^2>=0')
    call test_pass()
  end subroutine

  subroutine test_sld_squared_orthogonal()
    integer, parameter :: dim = 2
    complex(dp) :: ket(dim), dket(dim)
    real(dp) :: result

    call test_start('<L^2>=4 orthogonal')

    ket = [CONE, CZERO]
    dket = [CZERO, CONE]

    result = sld_squared_expectation(dket, ket, dim)
    call assert_close(result, 4.0_dp, 1e-12_dp, '<L^2>=4')
    call test_pass()
  end subroutine

  subroutine test_sld_squared_phase_invariance()
    integer, parameter :: dim = 3
    complex(dp) :: ket(dim), dket(dim), phase
    real(dp) :: L2_orig, L2_phased

    call test_start('<L^2> phase invariant')

    ket = CZERO
    ket(1) = cmplx(1.0_dp / sqrt(2.0_dp), 0.0_dp, dp)
    ket(2) = cmplx(1.0_dp / sqrt(2.0_dp), 0.0_dp, dp)
    dket = CZERO
    dket(2) = (0.1_dp, 0.0_dp)

    L2_orig = sld_squared_expectation(dket, ket, dim)

    phase = exp(CI * 1.047197551196598_dp)  ! pi/3
    L2_phased = sld_squared_expectation(phase * dket, phase * ket, dim)

    call assert_close(L2_orig, L2_phased, 1e-12_dp, '<L^2> invariant')
    call test_pass()
  end subroutine

  subroutine test_compute_sld_info_time()
    integer, parameter :: dim = 2
    complex(dp) :: ket(dim), dket(dim)
    type(SLDInformation_type) :: info

    call test_start('sld_info time field')

    ket = [CONE, CZERO]
    dket = [CZERO, (0.1_dp, 0.0_dp)]

    call compute_sld_info(dket, ket, dim, 2, 7, info)
    call assert_true(info%time == 7, 'time=7')
    call test_pass()
  end subroutine

  subroutine test_compute_sld_info_normalization()
    integer, parameter :: dim = 2
    complex(dp) :: ket(dim), dket(dim)
    type(SLDInformation_type) :: info
    real(dp) :: expected

    call test_start('sld_info normalization')

    ket = [CONE, CZERO]
    dket = [CZERO, (0.1_dp, 0.0_dp)]

    call compute_sld_info(dket, ket, dim, 2, 5, info)
    expected = info%L_squared_expectation / (4.0_dp * 25.0_dp)  ! N=2, t=5
    call assert_close(info%qfi_from_sld, expected, 1e-12_dp, 'qfi_from_sld')
    call test_pass()
  end subroutine

  subroutine test_compute_sld_info_l_exp_near_zero()
    integer, parameter :: dim = 2
    complex(dp) :: ket(dim), dket(dim)
    type(SLDInformation_type) :: info

    call test_start('sld_info <L>~0')

    ket = [CONE, CZERO]
    dket = [CZERO, (0.3_dp, 0.0_dp)]

    call compute_sld_info(dket, ket, dim, 2, 3, info)
    call assert_close(abs(info%L_expectation), 0.0_dp, 1e-12_dp, '<L>~0')
    call test_pass()
  end subroutine

end program test_sld
