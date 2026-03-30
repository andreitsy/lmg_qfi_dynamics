program test_eigh_recon
  use lmg_constants, only: dp, CZERO, CONE
  use lmg_linalg, only: eigh
  use lmg_operators, only: create_hamiltonian_h0
  implicit none

  integer, parameter :: n = 6
  integer :: dim, i, j
  complex(dp) :: H0(n+1,n+1), eigvecs(n+1,n+1)
  complex(dp) :: Hv(n+1), lv(n+1)
  real(dp) :: eigenvalues(n+1), residual

  dim = n + 1
  call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
  call eigh(H0, dim, eigenvalues, eigvecs)

  write(*,'(A)') '=== Check H0*v_i = lambda_i*v_i ==='
  do i = 1, dim
    ! Hv = H0 * v_i
    Hv = matmul(H0, eigvecs(:,i))
    ! lv = lambda_i * v_i
    lv = eigenvalues(i) * eigvecs(:,i)
    residual = maxval(abs(Hv - lv))
    write(*,'("  eigvec ",I1,": lambda=",ES12.4,"  max|H*v - lambda*v| = ",ES12.4)') i, eigenvalues(i), residual
  end do

  ! Also try with a 3x3 Hermitian (from test_linalg)
  write(*,'(A)') ''
  write(*,'(A)') '=== 3x3 Hermitian reconstruction test ==='
  block
    complex(dp) :: A(3,3), evecs(3,3), Av(3), lv3(3)
    real(dp) :: evals(3), res3
    integer :: k

    A(1,1) = (2.0_dp, 0.0_dp); A(1,2) = (0.5_dp, -0.3_dp); A(1,3) = (0.1_dp, 0.2_dp)
    A(2,1) = (0.5_dp, 0.3_dp); A(2,2) = (3.0_dp, 0.0_dp); A(2,3) = (0.4_dp, -0.1_dp)
    A(3,1) = (0.1_dp, -0.2_dp); A(3,2) = (0.4_dp, 0.1_dp); A(3,3) = (1.0_dp, 0.0_dp)

    call eigh(A, 3, evals, evecs)
    do k = 1, 3
      Av = matmul(A, evecs(:,k))
      lv3 = evals(k) * evecs(:,k)
      res3 = maxval(abs(Av - lv3))
      write(*,'("  eigvec ",I1,": lambda=",ES12.4,"  max|A*v - lambda*v| = ",ES12.4)') k, evals(k), res3
    end do
  end block

end program test_eigh_recon
