program lmg_qfi_sim
  use lmg_types, only: SimulationParams_type, STATE_GS_PHYS, STATE_GS_CAT, &
                        STATE_PHYS, STATE_CAT_SUM
  use lmg_config, only: set_default_params, read_config_file, parse_command_line, &
                         state_from_string
  use lmg_simulation, only: run_simulation
  implicit none

  type(SimulationParams_type) :: params
  integer :: ierr, i, nargs, state_id
  character(len=256) :: arg, val, config_file
  integer :: state_ids(4), num_states
  logical :: has_init_state, config_found

  ! Set defaults
  call set_default_params(params)

  ! Look for --config argument first
  config_file = 'parameters.ini'
  nargs = command_argument_count()
  i = 1
  do while (i <= nargs)
    call get_command_argument(i, arg)
    if (trim(arg) == '--config') then
      i = i + 1
      call get_command_argument(i, config_file)
    end if
    i = i + 1
  end do

  ! Read config file
  inquire(file=trim(config_file), exist=config_found)
  if (config_found) then
    call read_config_file(trim(config_file), params, ierr)
    if (ierr /= 0) then
      write(*,*) 'Warning: Could not read config file: ', trim(config_file)
    end if
  end if

  ! Parse command line (overrides config)
  call parse_command_line(params)

  ! Determine initial states to simulate
  has_init_state = .false.
  num_states = 0
  i = 1
  do while (i <= nargs)
    call get_command_argument(i, arg)
    if (trim(arg) == '--init-state') then
      i = i + 1
      call get_command_argument(i, val)
      state_id = state_from_string(trim(val))
      if (state_id > 0) then
        num_states = num_states + 1
        state_ids(num_states) = state_id
        has_init_state = .true.
      else
        write(*,*) 'ERROR: Unknown initial state: ', trim(val)
        stop 1
      end if
    end if
    i = i + 1
  end do

  ! Default: run all states
  if (.not. has_init_state) then
    num_states = 4
    state_ids(1) = STATE_GS_PHYS
    state_ids(2) = STATE_GS_CAT
    state_ids(3) = STATE_PHYS
    state_ids(4) = STATE_CAT_SUM
  end if

  ! Print parameters
  write(*,'(A)') '=== LMG QFI Dynamics Simulation ==='
  write(*,'(A,I0)')    '  N     = ', params%N
  write(*,'(A,F8.4)')  '  J     = ', params%J
  write(*,'(A,F8.4)')  '  B     = ', params%B
  write(*,'(A,F8.4)')  '  T     = ', params%T
  write(*,'(A,F8.4)')  '  phi   = ', params%phi_kick_phase
  write(*,'(A,F8.4)')  '  h     = ', params%h
  write(*,'(A,I0)')    '  freq  = ', params%freq
  write(*,'(A,I0)')    '  steps = ', params%steps_floquet_unitary
  write(*,'(A)')       '===================================='

  ! Ensure output directory exists
  call execute_command_line('mkdir -p ' // trim(params%output_dir), exitstat=ierr)

  ! Run simulation
  call run_simulation(params, state_ids, num_states)

  write(*,'(A)') 'Simulation complete.'
end program lmg_qfi_sim
