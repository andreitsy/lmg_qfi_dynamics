module lmg_simulation
  use lmg_constants, only: dp, PI, CZERO, EPSILON_FD
  use lmg_types, only: SimulationParams_type, UF_type, QFIInformation_type, &
                        STATE_GS_PHYS, STATE_GS_CAT, STATE_PHYS, STATE_CAT_SUM, STATE_LABELS
  use lmg_linalg, only: eigh, eye_complex
  use lmg_operators, only: create_spin_xyz_operators, create_hamiltonian_h0
  use lmg_evolution, only: calculate_unitary_T, build_uf
  use lmg_qfi_mod, only: process_time_point
  use lmg_io, only: save_to_file_qfi_dynamics, build_output_filename
  implicit none
  private
  public :: generate_time_interval, build_initial_state, simulation_with_AC_field, run_simulation

contains

  !> Generate time interval: 1..99 linearly, then logspace [2,4] and [4,max_degree]
  subroutine generate_time_interval(num_points, max_degree, time_points, total_points)
    integer, intent(in) :: num_points, max_degree
    integer, allocatable, intent(out) :: time_points(:)
    integer, intent(out) :: total_points

    integer :: i, n_linear, n_log1, n_log2, idx, val, prev_val
    integer, allocatable :: temp(:)
    real(dp) :: x

    if (max_degree <= 1) then
      write(*,*) 'ERROR: max_degree should be greater than 1!'
      stop 1
    end if

    n_linear = 99  ! 1..99
    n_log1 = num_points  ! logspace(2,4)
    n_log2 = num_points  ! logspace(4,max_degree)

    ! Over-allocate, then deduplicate
    allocate(temp(n_linear + n_log1 + n_log2))

    idx = 0
    ! Linear part: 1..99
    do i = 1, 99
      idx = idx + 1
      temp(idx) = i
    end do

    ! Logspace [2, 4), num_points points, endpoint=false
    do i = 0, n_log1 - 1
      x = 10.0_dp ** (2.0_dp + real(i, dp) * 2.0_dp / real(n_log1, dp))
      val = int(x)
      idx = idx + 1
      temp(idx) = val
    end do

    ! Logspace [4, max_degree], num_points points, endpoint=true
    do i = 0, n_log2 - 1
      x = 10.0_dp ** (4.0_dp + real(i, dp) * real(max_degree - 4, dp) / real(n_log2 - 1, dp))
      val = int(x)
      idx = idx + 1
      temp(idx) = val
    end do

    ! Sort and deduplicate
    call sort_integers(temp(1:idx), idx)
    total_points = 1
    prev_val = temp(1)
    do i = 2, idx
      if (temp(i) /= prev_val) then
        total_points = total_points + 1
        prev_val = temp(i)
      end if
    end do

    allocate(time_points(total_points))
    time_points(1) = temp(1)
    total_points = 1
    prev_val = temp(1)
    do i = 2, idx
      if (temp(i) /= prev_val) then
        total_points = total_points + 1
        time_points(total_points) = temp(i)
        prev_val = temp(i)
      end if
    end do

    deallocate(temp)
  end subroutine generate_time_interval

  !> Simple insertion sort for small integer arrays
  subroutine sort_integers(arr, n)
    integer, intent(inout) :: arr(n)
    integer, intent(in) :: n
    integer :: i, j, key

    do i = 2, n
      key = arr(i)
      j = i - 1
      do while (j >= 1 .and. arr(j) > key)
        arr(j+1) = arr(j)
        j = j - 1
      end do
      arr(j+1) = key
    end do
  end subroutine sort_integers

  !> Build initial state vector from state ID.
  subroutine build_initial_state(state_id, H0, dim, N, init_state)
    integer, intent(in) :: state_id, dim, N
    complex(dp), intent(in) :: H0(dim, dim)
    complex(dp), intent(out) :: init_state(dim)

    real(dp) :: eigenvalues(dim)
    complex(dp) :: eigvecs(dim, dim)
    integer :: sorted_idx(dim), i, j, min_idx
    real(dp) :: min_val, norm_val

    select case(state_id)
    case(STATE_GS_PHYS, STATE_GS_CAT)
      ! Eigendecompose H0
      call eigh(H0, dim, eigenvalues, eigvecs)

      ! Find sorted indices (ascending eigenvalue)
      do i = 1, dim
        sorted_idx(i) = i
      end do
      do i = 1, dim - 1
        min_idx = i
        min_val = eigenvalues(sorted_idx(i))
        do j = i + 1, dim
          if (eigenvalues(sorted_idx(j)) < min_val) then
            min_idx = j
            min_val = eigenvalues(sorted_idx(j))
          end if
        end do
        if (min_idx /= i) then
          j = sorted_idx(i)
          sorted_idx(i) = sorted_idx(min_idx)
          sorted_idx(min_idx) = j
        end if
      end do

      if (state_id == STATE_GS_CAT) then
        ! Ground state of H0
        init_state = eigvecs(:, sorted_idx(1))
      else
        ! GS_PHYS: ground state + first excited state
        init_state = eigvecs(:, sorted_idx(1)) + eigvecs(:, sorted_idx(2))
      end if

    case(STATE_PHYS)
      ! Fock state |N/2, N/2> = |0> (top spin state, first basis vector)
      init_state = CZERO
      init_state(1) = (1.0_dp, 0.0_dp)

    case(STATE_CAT_SUM)
      ! Superposition of top and bottom Fock states
      init_state = CZERO
      init_state(1) = (1.0_dp, 0.0_dp)
      init_state(dim) = (1.0_dp, 0.0_dp)

    case default
      write(*,*) 'ERROR: Unknown state ID:', state_id
      stop 1
    end select

    ! Normalize
    norm_val = sqrt(real(dot_product(init_state, init_state), dp))
    init_state = init_state / norm_val
  end subroutine build_initial_state

  !> Main simulation loop over time points.
  subroutine simulation_with_AC_field(params, time_points, num_time_points, &
                                       init_state, n, results)
    type(SimulationParams_type), intent(in) :: params
    integer, intent(in) :: time_points(:)
    integer, intent(in) :: num_time_points, n
    complex(dp), intent(in) :: init_state(n+1)
    type(QFIInformation_type), intent(out) :: results(num_time_points)

    complex(dp) :: H0(n+1, n+1), Sz(n+1, n+1), Sx(n+1, n+1), Sy(n+1, n+1)
    complex(dp) :: U_T(n+1, n+1), U_Tp(n+1, n+1), U_Tm(n+1, n+1)
    type(UF_type) :: uf, uf_p, uf_m
    real(dp) :: epsilon
    integer :: i, dim

    dim = n + 1
    epsilon = EPSILON_FD

    call create_hamiltonian_h0(params%J, params%B, n, H0)
    call create_spin_xyz_operators(n, Sz, Sx, Sy)

    ! Compute Floquet unitaries for h, h+eps, h-eps
    call calculate_unitary_T(params%h, params, H0, n, U_T)
    call build_uf(U_T, dim, uf)

    call calculate_unitary_T(params%h + epsilon, params, H0, n, U_Tp)
    call build_uf(U_Tp, dim, uf_p)

    call calculate_unitary_T(params%h - epsilon, params, H0, n, U_Tm)
    call build_uf(U_Tm, dim, uf_m)

    ! Parallel loop over time points
    !$OMP PARALLEL DO SCHEDULE(DYNAMIC) DEFAULT(SHARED) PRIVATE(i)
    do i = 1, num_time_points
      call process_time_point(time_points(i), params, H0, uf, uf_p, uf_m, &
                               init_state, Sz, Sx, Sy, n, results(i))
    end do
    !$OMP END PARALLEL DO

    ! Clean up UF allocations
    if (allocated(uf%eigenvalues)) deallocate(uf%eigenvalues)
    if (allocated(uf%U)) deallocate(uf%U)
    if (allocated(uf%U_inv)) deallocate(uf%U_inv)
    if (allocated(uf_p%eigenvalues)) deallocate(uf_p%eigenvalues)
    if (allocated(uf_p%U)) deallocate(uf_p%U)
    if (allocated(uf_p%U_inv)) deallocate(uf_p%U_inv)
    if (allocated(uf_m%eigenvalues)) deallocate(uf_m%eigenvalues)
    if (allocated(uf_m%U)) deallocate(uf_m%U)
    if (allocated(uf_m%U_inv)) deallocate(uf_m%U_inv)
  end subroutine simulation_with_AC_field

  !> Full simulation orchestration: for each state, compute and save results.
  subroutine run_simulation(params, state_ids, num_states)
    type(SimulationParams_type), intent(in) :: params
    integer, intent(in) :: state_ids(:)
    integer, intent(in) :: num_states

    complex(dp), allocatable :: H0(:,:), init_state(:)
    real(dp), allocatable :: eigenvalues(:)
    complex(dp), allocatable :: eigvecs(:,:)
    type(QFIInformation_type), allocatable :: results(:)
    integer, allocatable :: time_points(:)
    integer :: total_points, dim, s, last_time_degree
    real(dp) :: gap
    character(len=256) :: output_file
    character(len=10) :: slabel

    dim = params%N + 1
    allocate(H0(dim, dim), init_state(dim))
    allocate(eigenvalues(dim), eigvecs(dim, dim))

    call create_hamiltonian_h0(params%J, params%B, params%N, H0)
    call eigh(H0, dim, eigenvalues, eigvecs)

    ! Compute energy gap for time scale
    gap = eigenvalues(2) - eigenvalues(1)  ! ZHEEV returns sorted ascending
    last_time_degree = int(log10(abs(4.0_dp * PI / gap))) + 1
    if (last_time_degree < 2) last_time_degree = 2

    call generate_time_interval(params%num_points, last_time_degree, time_points, total_points)

    write(*,'(A,I0,A,I0)') 'Time interval: ', time_points(1), ' to ', time_points(total_points)
    write(*,'(A,I0,A)') 'Total time points: ', total_points, ''

    allocate(results(total_points))

    do s = 1, num_states
      slabel = STATE_LABELS(state_ids(s))
      write(*,'(A,A)') 'Running simulation for state: ', trim(slabel)

      call build_initial_state(state_ids(s), H0, dim, params%N, init_state)
      call simulation_with_AC_field(params, time_points, total_points, init_state, params%N, results)

      call build_output_filename(trim(slabel), params%N, params%B, trim(params%output_dir), output_file)
      call save_to_file_qfi_dynamics(results, total_points, trim(output_file))
      write(*,'(A,A)') 'Results saved to: ', trim(output_file)
    end do

    deallocate(H0, init_state, eigenvalues, eigvecs, results, time_points)
  end subroutine run_simulation

end module lmg_simulation
