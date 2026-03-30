program test_eigh_detail
  use lmg_constants, only: dp, CZERO, CONE
  use lmg_linalg, only: eigh, adjoint
  implicit none

  integer, parameter :: n = 3
  complex(dp) :: A(n,n), eigvecs(n,n), vdv(n,n), vdag(n,n), scaled(n,n)
  real(dp) :: eigenvalues(n)
  integer :: i, j

  A = CZERO
  A(1,1) = (2.0_dp, 0.0_dp)
  A(2,2) = (3.0_dp, 0.0_dp)
  A(3,3) = (1.0_dp, 0.0_dp)
  A(1,2) = (0.5_dp, -0.3_dp); A(2,1) = (0.5_dp, 0.3_dp)
  A(1,3) = (0.1_dp, 0.2_dp); A(3,1) = (0.1_dp, -0.2_dp)
  A(2,3) = (0.4_dp, -0.1_dp); A(3,2) = (0.4_dp, 0.1_dp)

  call eigh(A, n, eigenvalues, eigvecs)

  write(*,'(A)') '=== Eigenvalues ==='
  do i = 1, n
    write(*,'(I2,": ",ES23.15)') i, eigenvalues(i)
  end do

  write(*,'(A)') ''
  write(*,'(A)') '=== Eigenvectors (columns) ==='
  do i = 1, n
    do j = 1, n
      write(*,'("  V(",I1,",",I1,") = (",ES15.8,",",ES15.8,")")') i, j, real(eigvecs(i,j),dp), aimag(eigvecs(i,j))
    end do
  end do

  ! Check V^dag * V = I
  call adjoint(eigvecs, n, vdag)
  vdv = matmul(vdag, eigvecs)
  write(*,'(A)') ''
  write(*,'(A)') '=== V^dag * V ==='
  do i = 1, n
    do j = 1, n
      write(*,'("(",F10.6,",",F10.6,") ")', advance='no') real(vdv(i,j),dp), aimag(vdv(i,j))
    end do
    write(*,*)
  end do

  ! Check V * D * V^dag = A
  do i = 1, n
    scaled(:,i) = eigvecs(:,i) * eigenvalues(i)
  end do
  vdv = matmul(scaled, vdag)
  write(*,'(A)') ''
  write(*,'(A)') '=== V * D * V^dag ==='
  do i = 1, n
    do j = 1, n
      write(*,'("(",F10.6,",",F10.6,") ")', advance='no') real(vdv(i,j),dp), aimag(vdv(i,j))
    end do
    write(*,*)
  end do

  write(*,'(A)') ''
  write(*,'(A)') '=== Original A ==='
  do i = 1, n
    do j = 1, n
      write(*,'("(",F10.6,",",F10.6,") ")', advance='no') real(A(i,j),dp), aimag(A(i,j))
    end do
    write(*,*)
  end do

  write(*,'(A,ES12.4)') 'Max |V*D*V^dag - A| = ', maxval(abs(vdv - A))

end program test_eigh_detail
