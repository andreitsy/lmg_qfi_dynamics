program test_debug_UT
  use lmg_constants, only: dp, CI, CZERO, CONE, PI
  use lmg_types, only: UF_type, SimulationParams_type
  use lmg_linalg, only: eig, matrix_inverse, eye_complex, expm_hermitian
  use lmg_operators, only: create_spin_xyz_operators, create_hamiltonian_h0, create_kick_operator
  use lmg_evolution, only: evolution_T_step, find_power_r, build_uf, &
                            calculate_unitary_T, calculate_unitary_at_time
  implicit none

  integer, parameter :: n = 6
  integer :: dim, i
  type(SimulationParams_type) :: params
  complex(dp) :: H0(n+1,n+1), Sz(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1)
  complex(dp) :: kick(n+1,n+1), U_T(n+1,n+1)
  complex(dp) :: UT_r1(n+1,n+1)
  type(UF_type) :: uf
  complex(dp) :: init_state(n+1), ket_direct(n+1), ket_eig(n+1), temp_vec(n+1)
  real(dp) :: mx_direct, mx_eig, mz_direct, mz_eig

  dim = n + 1

  params%N = n
  params%J = 1.0_dp
  params%B = 0.4_dp
  params%T = 1.0_dp
  params%phi_kick_phase = PI
  params%h = 0.0_dp
  params%varphi = 0.0_dp
  params%theta = 0.0_dp
  params%phi_0 = 0.0_dp
  params%freq = 2
  params%steps_floquet_unitary = 50
  params%num_points = 500
  params%dps = 30

  ! Create operators
  call create_spin_xyz_operators(n, Sz, Sx, Sy)
  call create_hamiltonian_h0(params%J, params%B, n, H0)
  call create_kick_operator(params%phi_kick_phase, Sx, n, kick)

  ! Compute U_T
  call calculate_unitary_T(0.0_dp, params, H0, Sz, Sx, Sy, kick, n, U_T)

  ! Print first few elements of U_T
  write(*,'(A)') '=== U_T matrix (first 3x3 block) ==='
  do i = 1, min(3, dim)
    write(*,'(3("(",ES23.15E3,",",ES23.15E3,") "))') U_T(i,1), U_T(i,2), U_T(i,3)
  end do

  ! Eigendecompose U_T
  call build_uf(U_T, dim, uf)

  ! U_T^1 via eigendecomposition
  call find_power_r(uf, 1, dim, UT_r1)

  ! Print U_T^1 from eigendecomposition
  write(*,'(A)') ''
  write(*,'(A)') '=== U_T^1 from find_power_r (first 3x3 block) ==='
  do i = 1, min(3, dim)
    write(*,'(3("(",ES23.15E3,",",ES23.15E3,") "))') UT_r1(i,1), UT_r1(i,2), UT_r1(i,3)
  end do

  ! Check max difference
  write(*,'(A)') ''
  write(*,'(A,ES12.4)') 'Max |U_T - U_T^1_eig|: ', maxval(abs(U_T - UT_r1))

  ! Initial state: PHYS (first basis vector)
  init_state = CZERO
  init_state(1) = CONE

  ! Evolve with U_T directly (no eigendecomposition)
  ket_direct = matmul(U_T, init_state)

  ! Evolve with U_T^1 from eigendecomposition
  ket_eig = matmul(UT_r1, init_state)

  ! Magnetizations
  temp_vec = matmul(Sx, ket_direct)
  mx_direct = real(dot_product(ket_direct, temp_vec), dp) / real(n, dp)
  temp_vec = matmul(Sz, ket_direct)
  mz_direct = real(dot_product(ket_direct, temp_vec), dp) / real(n, dp)

  temp_vec = matmul(Sx, ket_eig)
  mx_eig = real(dot_product(ket_eig, temp_vec), dp) / real(n, dp)
  temp_vec = matmul(Sz, ket_eig)
  mz_eig = real(dot_product(ket_eig, temp_vec), dp) / real(n, dp)

  write(*,'(A)') ''
  write(*,'(A)') '=== Magnetization from direct U_T*|PHYS> ==='
  write(*,'(A,ES23.15E3)') 'm_x(direct): ', mx_direct
  write(*,'(A,ES23.15E3)') 'm_z(direct): ', mz_direct
  write(*,'(A)') ''
  write(*,'(A)') '=== Magnetization from eigendecomposed U_T^1*|PHYS> ==='
  write(*,'(A,ES23.15E3)') 'm_x(eig):    ', mx_eig
  write(*,'(A,ES23.15E3)') 'm_z(eig):    ', mz_eig

  ! Now test with r=2 (simulate time=2*nu=4 for nu=2)
  write(*,'(A)') ''
  write(*,'(A)') '=== r=1 state comparison ==='
  write(*,'(A,ES12.4)') 'Max |ket_direct - ket_eig|: ', maxval(abs(ket_direct - ket_eig))

  ! Clean up
  if (allocated(uf%eigenvalues)) deallocate(uf%eigenvalues)
  if (allocated(uf%U)) deallocate(uf%U)
  if (allocated(uf%U_inv)) deallocate(uf%U_inv)

end program test_debug_UT
