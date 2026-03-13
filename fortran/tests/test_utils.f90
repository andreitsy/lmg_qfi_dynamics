module test_utils
  use lmg_constants, only: dp
  implicit none
  private
  public :: assert_close, assert_close_complex, assert_matrix_hermitian, &
            assert_matrix_unitary, assert_matrix_close, assert_true, &
            test_suite_start, test_suite_end, test_start, test_pass, &
            tests_passed, tests_failed, tests_total

  integer :: tests_passed = 0
  integer :: tests_failed = 0
  integer :: tests_total = 0

contains

  subroutine test_suite_start(name)
    character(len=*), intent(in) :: name
    write(*,'(A)') ''
    write(*,'(A,A)') '=== Test Suite: ', name
    tests_passed = 0
    tests_failed = 0
    tests_total = 0
  end subroutine test_suite_start

  subroutine test_suite_end()
    write(*,'(A)') ''
    write(*,'(A,I0,A,I0,A,I0)') 'Results: ', tests_passed, ' passed, ', &
      tests_failed, ' failed, ', tests_total, ' total'
    if (tests_failed > 0) then
      write(*,'(A)') 'SOME TESTS FAILED!'
    else
      write(*,'(A)') 'ALL TESTS PASSED.'
    end if
  end subroutine test_suite_end

  subroutine test_start(name)
    character(len=*), intent(in) :: name
    tests_total = tests_total + 1
    write(*,'(A,A,A)', advance='no') '  [', name, '] ... '
  end subroutine test_start

  subroutine test_pass()
    tests_passed = tests_passed + 1
    write(*,'(A)') 'PASS'
  end subroutine test_pass

  subroutine test_fail(msg)
    character(len=*), intent(in) :: msg
    tests_failed = tests_failed + 1
    write(*,'(A,A)') 'FAIL: ', msg
  end subroutine test_fail

  subroutine assert_true(condition, msg)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: msg
    if (.not. condition) then
      call test_fail(msg)
      stop 1
    end if
  end subroutine assert_true

  subroutine assert_close(a, b, tol, msg)
    real(dp), intent(in) :: a, b, tol
    character(len=*), intent(in) :: msg
    if (abs(a - b) > tol) then
      write(*,'(A,ES12.4,A,ES12.4,A,ES12.4)') 'FAIL: ', a, ' /= ', b, ' tol=', tol
      call test_fail(msg)
      stop 1
    end if
  end subroutine assert_close

  subroutine assert_close_complex(a, b, tol, msg)
    complex(dp), intent(in) :: a, b
    real(dp), intent(in) :: tol
    character(len=*), intent(in) :: msg
    if (abs(a - b) > tol) then
      call test_fail(msg)
      stop 1
    end if
  end subroutine assert_close_complex

  !> Check that a matrix is Hermitian: A = A^dagger
  subroutine assert_matrix_hermitian(A, n, tol, msg)
    complex(dp), intent(in) :: A(n,n)
    integer, intent(in) :: n
    real(dp), intent(in) :: tol
    character(len=*), intent(in) :: msg
    integer :: i, j

    do j = 1, n
      do i = 1, n
        if (abs(A(i,j) - conjg(A(j,i))) > tol) then
          write(*,'(A,I0,A,I0,A,2ES12.4,A,2ES12.4)') &
            'Non-Hermitian at (', i, ',', j, '): ', &
            real(A(i,j)), aimag(A(i,j)), ' vs ', real(conjg(A(j,i))), aimag(conjg(A(j,i)))
          call test_fail(msg)
          stop 1
        end if
      end do
    end do
  end subroutine assert_matrix_hermitian

  !> Check that a matrix is unitary: A * A^dagger = I
  subroutine assert_matrix_unitary(A, n, tol, msg)
    complex(dp), intent(in) :: A(n,n)
    integer, intent(in) :: n
    real(dp), intent(in) :: tol
    character(len=*), intent(in) :: msg
    complex(dp) :: product(n,n)
    integer :: i, j

    product = matmul(A, conjg(transpose(A)))
    do j = 1, n
      do i = 1, n
        if (i == j) then
          if (abs(product(i,j) - (1.0_dp, 0.0_dp)) > tol) then
            call test_fail(msg)
            stop 1
          end if
        else
          if (abs(product(i,j)) > tol) then
            call test_fail(msg)
            stop 1
          end if
        end if
      end do
    end do
  end subroutine assert_matrix_unitary

  !> Check that two matrices are close element-wise
  subroutine assert_matrix_close(A, B, n, tol, msg)
    complex(dp), intent(in) :: A(n,n), B(n,n)
    integer, intent(in) :: n
    real(dp), intent(in) :: tol
    character(len=*), intent(in) :: msg
    integer :: i, j

    do j = 1, n
      do i = 1, n
        if (abs(A(i,j) - B(i,j)) > tol) then
          write(*,'(A,I0,A,I0,A,2ES12.4,A,2ES12.4)') &
            'Mismatch at (', i, ',', j, '): ', &
            real(A(i,j)), aimag(A(i,j)), ' vs ', real(B(i,j)), aimag(B(i,j))
          call test_fail(msg)
          stop 1
        end if
      end do
    end do
  end subroutine assert_matrix_close

end module test_utils
