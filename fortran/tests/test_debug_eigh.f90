program test_debug_eigh
  use lmg_constants, only: dp, CI, CZERO, CONE
  use lmg_linalg, only: eigh, expm_hermitian, adjoint
  use lmg_operators, only: create_hamiltonian_h0
  implicit none

  integer, parameter :: n = 6
  integer :: dim, i, j
  complex(dp) :: H0(n+1,n+1), eigvecs(n+1,n+1), exp_result(n+1,n+1)
  complex(dp) :: reconstructed(n+1,n+1), scaled(n+1,n+1), eigvecs_dag(n+1,n+1)
  real(dp) :: eigenvalues(n+1)

  dim = n + 1
  call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)

  ! Eigendecompose
  call eigh(H0, dim, eigenvalues, eigvecs)

  write(*,'(A)') '=== Eigenvalues ==='
  do i = 1, dim
    write(*,'(I2,": ",ES23.15E3)') i, eigenvalues(i)
  end do

  ! Verify: V * diag(lambda) * V^dag = H0
  call adjoint(eigvecs, dim, eigvecs_dag)
  do i = 1, dim
    scaled(:, i) = eigvecs(:, i) * eigenvalues(i)
  end do
  reconstructed = matmul(scaled, eigvecs_dag)

  write(*,'(A)') ''
  write(*,'(A)') '=== Max |V*D*V^dag - H0| ==='
  write(*,'(ES12.4)') maxval(abs(reconstructed - H0))

  ! Verify orthogonality: V^dag * V = I
  reconstructed = matmul(eigvecs_dag, eigvecs)
  write(*,'(A)') ''
  write(*,'(A)') '=== Max |V^dag*V - I| ==='
  do i = 1, dim
    reconstructed(i,i) = reconstructed(i,i) - CONE
  end do
  write(*,'(ES12.4)') maxval(abs(reconstructed))

  ! Print eigenvector 1 (first 4 components)
  write(*,'(A)') ''
  write(*,'(A)') '=== First eigenvector (v1, first 4 comps) ==='
  do i = 1, 4
    write(*,'("  v1(",I1,") = (",ES15.8E2,",",ES15.8E2,")")') i, real(eigvecs(i,1), dp), aimag(eigvecs(i,1))
  end do

  ! Now compute exp(-i*1.0*H0) manually
  write(*,'(A)') ''
  write(*,'(A)') '=== Manual exp(-i*H0) reconstruction ==='
  do i = 1, dim
    scaled(:, i) = eigvecs(:, i) * exp(-CI * 1.0_dp * eigenvalues(i))
  end do
  call adjoint(eigvecs, dim, eigvecs_dag)
  exp_result = matmul(scaled, eigvecs_dag)

  write(*,'(A)') 'exp(-i*H0)(1,2):'
  write(*,'("  (",ES15.8E2,",",ES15.8E2,")")') real(exp_result(1,2), dp), aimag(exp_result(1,2))

  ! Compare with expm_hermitian
  call expm_hermitian(H0, dim, 1.0_dp, exp_result)
  write(*,'(A)') 'expm_hermitian(H0,1.0)(1,2):'
  write(*,'("  (",ES15.8E2,",",ES15.8E2,")")') real(exp_result(1,2), dp), aimag(exp_result(1,2))

end program test_debug_eigh
