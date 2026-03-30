program test_simple_eigh
  use lmg_constants, only: dp, CZERO, CONE
  use lmg_linalg, only: eigh
  implicit none

  integer, parameter :: n = 3
  complex(dp) :: A(n,n), A_copy(n,n), eigvecs(n,n)
  real(dp) :: eigenvalues(n)
  complex(dp) :: Av(n), lv(n)
  real(dp) :: residual
  integer :: i

  ! Simple 3x3 Hermitian matrix
  A = CZERO
  A(1,1) = (2.0_dp, 0.0_dp)
  A(2,2) = (3.0_dp, 0.0_dp)
  A(3,3) = (1.0_dp, 0.0_dp)
  A(1,2) = (0.5_dp, -0.3_dp); A(2,1) = (0.5_dp, 0.3_dp)
  A(1,3) = (0.1_dp, 0.2_dp); A(3,1) = (0.1_dp, -0.2_dp)
  A(2,3) = (0.4_dp, -0.1_dp); A(3,2) = (0.4_dp, 0.1_dp)

  ! Save copy before eigh
  A_copy = A

  write(*,'(A)') '=== A before eigh ==='
  do i = 1, n
    write(*,'(3("(",F8.4,",",F8.4,") "))') A(i,1), A(i,2), A(i,3)
  end do

  call eigh(A, n, eigenvalues, eigvecs)

  write(*,'(A)') ''
  write(*,'(A)') '=== A after eigh ==='
  do i = 1, n
    write(*,'(3("(",F8.4,",",F8.4,") "))') A(i,1), A(i,2), A(i,3)
  end do

  write(*,'(A,ES12.4)') 'Max |A_after - A_before| = ', maxval(abs(A - A_copy))

  write(*,'(A)') ''
  write(*,'(A)') '=== Check H*v = lambda*v using SAVED copy of A ==='
  do i = 1, n
    Av = matmul(A_copy, eigvecs(:,i))
    lv = eigenvalues(i) * eigvecs(:,i)
    residual = maxval(abs(Av - lv))
    write(*,'("  eigvec ",I1,": lambda=",ES12.4,"  max|A*v - lambda*v| = ",ES12.4)') i, eigenvalues(i), residual
  end do

  write(*,'(A)') ''
  write(*,'(A)') '=== Check H*v = lambda*v using A (possibly modified) ==='
  do i = 1, n
    Av = matmul(A, eigvecs(:,i))
    lv = eigenvalues(i) * eigvecs(:,i)
    residual = maxval(abs(Av - lv))
    write(*,'("  eigvec ",I1,": lambda=",ES12.4,"  max|A*v - lambda*v| = ",ES12.4)') i, eigenvalues(i), residual
  end do

end program test_simple_eigh
