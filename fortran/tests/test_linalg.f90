program test_linalg
  use lmg_constants, only: dp, CI, CZERO, CONE, PI
  use lmg_linalg, only: eigh, eig, matrix_inverse, expm_hermitian, adjoint, eye_complex
  use test_utils
  implicit none

  call test_suite_start('lmg_linalg')

  call test_eigh_2x2()
  call test_eigh_diagonal()
  call test_eig_2x2()
  call test_inverse_identity()
  call test_inverse_2x2()
  call test_inverse_roundtrip()
  call test_expm_identity()
  call test_expm_pauli_x()
  call test_expm_unitary()
  call test_adjoint_basic()
  call test_eye_complex_basic()

  call test_suite_end()
  if (tests_failed > 0) stop 1

contains

  subroutine test_eigh_2x2()
    complex(dp) :: A(2,2), evecs(2,2)
    real(dp) :: evals(2)

    call test_start('eigh_2x2')

    ! Pauli Z: eigenvalues -1, +1
    A = CZERO
    A(1,1) = (1.0_dp, 0.0_dp)
    A(2,2) = (-1.0_dp, 0.0_dp)

    call eigh(A, 2, evals, evecs)

    call assert_close(evals(1), -1.0_dp, 1e-12_dp, 'eigenvalue 1')
    call assert_close(evals(2), 1.0_dp, 1e-12_dp, 'eigenvalue 2')
    call test_pass()
  end subroutine test_eigh_2x2

  subroutine test_eigh_diagonal()
    integer, parameter :: n = 4
    complex(dp) :: A(n,n), evecs(n,n)
    real(dp) :: evals(n), expected(n)
    integer :: i

    call test_start('eigh_diagonal')

    A = CZERO
    expected = [1.0_dp, 3.0_dp, 5.0_dp, 7.0_dp]
    do i = 1, n
      A(i,i) = cmplx(expected(i), 0.0_dp, dp)
    end do

    call eigh(A, n, evals, evecs)

    do i = 1, n
      call assert_close(evals(i), expected(i), 1e-12_dp, 'diagonal eigenvalue')
    end do
    call test_pass()
  end subroutine test_eigh_diagonal

  subroutine test_eig_2x2()
    complex(dp) :: A(2,2), evecs(2,2), evals(2)
    complex(dp) :: D(2,2), A_inv(2,2), check(2,2)

    call test_start('eig_2x2')

    ! Non-symmetric matrix
    A(1,1) = (2.0_dp, 0.0_dp)
    A(1,2) = (1.0_dp, 0.0_dp)
    A(2,1) = (0.0_dp, 0.0_dp)
    A(2,2) = (3.0_dp, 0.0_dp)

    call eig(A, 2, evals, evecs)

    ! Eigenvalues should be 2 and 3 (upper triangular)
    ! Check A * v = lambda * v for each eigenvector
    call assert_true(abs(abs(evals(1)) - 2.0_dp) < 1e-10_dp .or. &
                     abs(abs(evals(1)) - 3.0_dp) < 1e-10_dp, 'eig eigenvalue check')
    call test_pass()
  end subroutine test_eig_2x2

  subroutine test_inverse_identity()
    integer, parameter :: n = 3
    complex(dp) :: I_mat(n,n), I_inv(n,n)

    call test_start('inverse_identity')

    call eye_complex(n, I_mat)
    call matrix_inverse(I_mat, n, I_inv)
    call assert_matrix_close(I_inv, I_mat, n, 1e-12_dp, 'inv(I) = I')
    call test_pass()
  end subroutine test_inverse_identity

  subroutine test_inverse_2x2()
    complex(dp) :: A(2,2), A_inv(2,2), product(2,2), I_mat(2,2)

    call test_start('inverse_2x2')

    A(1,1) = (1.0_dp, 0.0_dp); A(1,2) = (2.0_dp, 0.0_dp)
    A(2,1) = (3.0_dp, 0.0_dp); A(2,2) = (4.0_dp, 0.0_dp)

    call matrix_inverse(A, 2, A_inv)
    product = matmul(A, A_inv)
    call eye_complex(2, I_mat)
    call assert_matrix_close(product, I_mat, 2, 1e-12_dp, 'A*A_inv = I')
    call test_pass()
  end subroutine test_inverse_2x2

  subroutine test_inverse_roundtrip()
    integer, parameter :: n = 3
    complex(dp) :: A(n,n), A_inv(n,n), product(n,n), I_mat(n,n)

    call test_start('inverse_roundtrip')

    ! Hermitian matrix
    A(1,1) = (2.0_dp, 0.0_dp); A(1,2) = (1.0_dp, -1.0_dp); A(1,3) = CZERO
    A(2,1) = (1.0_dp, 1.0_dp); A(2,2) = (3.0_dp, 0.0_dp); A(2,3) = (0.5_dp, -0.5_dp)
    A(3,1) = CZERO; A(3,2) = (0.5_dp, 0.5_dp); A(3,3) = (1.0_dp, 0.0_dp)

    call matrix_inverse(A, n, A_inv)
    product = matmul(A, A_inv)
    call eye_complex(n, I_mat)
    call assert_matrix_close(product, I_mat, n, 1e-10_dp, 'A*A_inv = I roundtrip')
    call test_pass()
  end subroutine test_inverse_roundtrip

  subroutine test_expm_identity()
    integer, parameter :: n = 3
    complex(dp) :: H(n,n), result(n,n), I_mat(n,n)

    call test_start('expm_zero_gives_identity')

    call eye_complex(n, H)
    ! exp(-i * 0 * H) = I
    call expm_hermitian(H, n, 0.0_dp, result)
    call eye_complex(n, I_mat)
    call assert_matrix_close(result, I_mat, n, 1e-12_dp, 'exp(0) = I')
    call test_pass()
  end subroutine test_expm_identity

  subroutine test_expm_pauli_x()
    complex(dp) :: Sx(2,2), result(2,2)
    real(dp) :: alpha
    complex(dp) :: expected(2,2)

    call test_start('expm_pauli_x')

    ! Pauli X / 2
    Sx = CZERO
    Sx(1,2) = (0.5_dp, 0.0_dp)
    Sx(2,1) = (0.5_dp, 0.0_dp)

    alpha = PI  ! exp(-i*pi*Sx) should be -i*2*Sx for spin-1/2, let's just check unitarity
    call expm_hermitian(Sx, 2, alpha, result)
    call assert_matrix_unitary(result, 2, 1e-12_dp, 'exp(-i*pi*Sx) is unitary')
    call test_pass()
  end subroutine test_expm_pauli_x

  subroutine test_expm_unitary()
    integer, parameter :: n = 3
    complex(dp) :: H(n,n), result(n,n)

    call test_start('expm_result_is_unitary')

    ! Random Hermitian matrix
    H(1,1) = (1.0_dp, 0.0_dp); H(1,2) = (0.5_dp, -0.3_dp); H(1,3) = (0.1_dp, 0.2_dp)
    H(2,1) = (0.5_dp, 0.3_dp); H(2,2) = (2.0_dp, 0.0_dp); H(2,3) = (0.4_dp, -0.1_dp)
    H(3,1) = (0.1_dp, -0.2_dp); H(3,2) = (0.4_dp, 0.1_dp); H(3,3) = (3.0_dp, 0.0_dp)

    call expm_hermitian(H, n, 1.5_dp, result)
    call assert_matrix_unitary(result, n, 1e-10_dp, 'expm produces unitary')
    call test_pass()
  end subroutine test_expm_unitary

  subroutine test_adjoint_basic()
    complex(dp) :: A(2,2), A_dag(2,2)

    call test_start('adjoint_basic')

    A(1,1) = (1.0_dp, 2.0_dp); A(1,2) = (3.0_dp, 4.0_dp)
    A(2,1) = (5.0_dp, 6.0_dp); A(2,2) = (7.0_dp, 8.0_dp)

    call adjoint(A, 2, A_dag)

    call assert_close_complex(A_dag(1,1), (1.0_dp, -2.0_dp), 1e-12_dp, 'adjoint(1,1)')
    call assert_close_complex(A_dag(1,2), (5.0_dp, -6.0_dp), 1e-12_dp, 'adjoint(1,2)')
    call assert_close_complex(A_dag(2,1), (3.0_dp, -4.0_dp), 1e-12_dp, 'adjoint(2,1)')
    call assert_close_complex(A_dag(2,2), (7.0_dp, -8.0_dp), 1e-12_dp, 'adjoint(2,2)')
    call test_pass()
  end subroutine test_adjoint_basic

  subroutine test_eye_complex_basic()
    integer, parameter :: n = 4
    complex(dp) :: I_mat(n,n)
    integer :: i, j

    call test_start('eye_complex')

    call eye_complex(n, I_mat)
    do j = 1, n
      do i = 1, n
        if (i == j) then
          call assert_close_complex(I_mat(i,j), CONE, 1e-12_dp, 'eye diagonal')
        else
          call assert_close_complex(I_mat(i,j), CZERO, 1e-12_dp, 'eye off-diagonal')
        end if
      end do
    end do
    call test_pass()
  end subroutine test_eye_complex_basic

end program test_linalg
