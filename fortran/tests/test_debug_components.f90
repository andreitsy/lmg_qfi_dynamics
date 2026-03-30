program test_debug_components
  use lmg_constants, only: dp, CI, CZERO, CONE, PI
  use lmg_linalg, only: expm_hermitian
  use lmg_operators, only: create_spin_xyz_operators, create_hamiltonian_h0, create_kick_operator
  implicit none

  integer, parameter :: n = 6
  integer :: dim, i, j
  complex(dp) :: H0(n+1,n+1), Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1)
  complex(dp) :: kick(n+1,n+1), exp_H0_T(n+1,n+1)

  dim = n + 1
  call create_spin_xyz_operators(n, Sz, Sx, Sy)
  call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)

  ! Print Sx(1:4, 1:4)
  write(*,'(A)') '=== Sx (first 4x4) ==='
  do i = 1, 4
    do j = 1, 4
      write(*,'("(",ES15.8E2,",",ES15.8E2,") ")', advance='no') real(Sx(i,j), dp), aimag(Sx(i,j))
    end do
    write(*,*)
  end do

  ! Compute and print kick = exp(-i*pi*Sx)
  call create_kick_operator(PI, Sx, n, kick)
  write(*,'(A)') ''
  write(*,'(A)') '=== kick = exp(-i*pi*Sx) (full matrix) ==='
  do i = 1, dim
    do j = 1, dim
      if (abs(kick(i,j)) > 1e-10_dp) then
        write(*,'("(",ES12.4E2,",",ES12.4E2,")")', advance='no') real(kick(i,j), dp), aimag(kick(i,j))
      else
        write(*,'("(    0       ,    0       )")', advance='no')
      end if
    end do
    write(*,*)
  end do

  ! Compute and print exp(-i*T*H0) first 3x3
  call expm_hermitian(H0, dim, 1.0_dp, exp_H0_T)
  write(*,'(A)') ''
  write(*,'(A)') '=== exp(-i*T*H0) (first 3x3) ==='
  do i = 1, 3
    do j = 1, 3
      write(*,'("(",ES15.8E2,",",ES15.8E2,") ")', advance='no') real(exp_H0_T(i,j), dp), aimag(exp_H0_T(i,j))
    end do
    write(*,*)
  end do
end program test_debug_components
