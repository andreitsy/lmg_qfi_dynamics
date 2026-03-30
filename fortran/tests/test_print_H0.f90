program test_print_H0
  use lmg_constants, only: dp
  use lmg_operators, only: create_hamiltonian_h0
  implicit none
  integer, parameter :: n = 6
  complex(dp) :: H0(n+1,n+1)
  integer :: i, j

  call create_hamiltonian_h0(1.0_dp, 0.4_dp, n, H0)
  write(*,'(A)') 'Fortran H0 (4x4):'
  do i = 1, 4
    do j = 1, 4
      write(*,'(F12.6)', advance='no') real(H0(i,j), dp)
    end do
    write(*,*)
  end do
end program test_print_H0
