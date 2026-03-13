module lmg_config
  use lmg_constants, only: dp, PI
  use lmg_types, only: SimulationParams_type, STATE_GS_PHYS, STATE_GS_CAT, &
                        STATE_PHYS, STATE_CAT_SUM, STATE_LABELS
  implicit none
  private
  public :: set_default_params, read_config_file, parse_command_line
  public :: state_from_string, state_label

contains

  !> Set default parameter values
  subroutine set_default_params(params)
    type(SimulationParams_type), intent(out) :: params

    params%N = 10
    params%J = 1.0_dp
    params%B = 0.4_dp
    params%T = 1.0_dp
    params%phi_kick_phase = PI
    params%h = 0.0_dp
    params%varphi = 0.0_dp
    params%theta = 0.0_dp
    params%phi_0 = 0.0_dp
    params%freq = 2
    params%steps_floquet_unitary = 10
    params%num_points = 200
    params%dps = 15
    params%output_dir = 'results'
  end subroutine set_default_params

  !> Read parameters from INI config file.
  !> Simple parser: looks for key = value lines in [Simulation] and [Files] sections.
  subroutine read_config_file(filename, params, ierr)
    character(len=*), intent(in) :: filename
    type(SimulationParams_type), intent(inout) :: params
    integer, intent(out) :: ierr

    integer :: unit_num, ios
    character(len=512) :: line, section, key, val
    integer :: eq_pos, semi_pos

    ierr = 0
    section = ''

    open(newunit=unit_num, file=filename, status='old', action='read', iostat=ios)
    if (ios /= 0) then
      ierr = -1
      return
    end if

    do
      read(unit_num, '(A)', iostat=ios) line
      if (ios /= 0) exit

      ! Strip leading/trailing whitespace
      line = adjustl(line)

      ! Skip empty and comment lines
      if (len_trim(line) == 0) cycle
      if (line(1:1) == ';' .or. line(1:1) == '#') cycle

      ! Section header
      if (line(1:1) == '[') then
        eq_pos = index(line, ']')
        if (eq_pos > 2) then
          section = line(2:eq_pos-1)
        end if
        cycle
      end if

      ! Key = value
      eq_pos = index(line, '=')
      if (eq_pos < 2) cycle

      key = adjustl(trim(line(1:eq_pos-1)))
      val = adjustl(line(eq_pos+1:))

      ! Strip inline comments
      semi_pos = index(val, ';')
      if (semi_pos > 0) val = val(1:semi_pos-1)
      semi_pos = index(val, '#')
      if (semi_pos > 0) val = val(1:semi_pos-1)
      val = trim(val)

      if (trim(section) == 'Simulation') then
        call parse_simulation_key(key, val, params)
      else if (trim(section) == 'Files') then
        call parse_files_key(key, val, params)
      end if
    end do

    close(unit_num)
  end subroutine read_config_file

  subroutine parse_simulation_key(key, val, params)
    character(len=*), intent(in) :: key, val
    type(SimulationParams_type), intent(inout) :: params
    character(len=256) :: k
    integer :: ios

    k = trim(key)

    select case(k)
    case('J', 'j')
      read(val, *, iostat=ios) params%J
    case('N', 'n')
      read(val, *, iostat=ios) params%N
    case('B', 'b')
      read(val, *, iostat=ios) params%B
    case('T', 't')
      read(val, *, iostat=ios) params%T
    case('phi-kick-phase')
      params%phi_kick_phase = parse_float_or_pi(val)
    case('h')
      read(val, *, iostat=ios) params%h
    case('frequency')
      read(val, *, iostat=ios) params%freq
    case('phi-0')
      params%phi_0 = parse_float_or_pi(val)
    case('varphi')
      params%varphi = parse_float_or_pi(val)
    case('theta')
      params%theta = parse_float_or_pi(val)
    case('num-points')
      read(val, *, iostat=ios) params%num_points
    case('steps-floquet-unitary')
      read(val, *, iostat=ios) params%steps_floquet_unitary
    case('dps')
      read(val, *, iostat=ios) params%dps
    end select
  end subroutine parse_simulation_key

  subroutine parse_files_key(key, val, params)
    character(len=*), intent(in) :: key, val
    type(SimulationParams_type), intent(inout) :: params

    if (trim(key) == 'output-dir') then
      params%output_dir = trim(val)
    end if
  end subroutine parse_files_key

  !> Parse a string that can be "pi" or a numeric value
  function parse_float_or_pi(val) result(res)
    character(len=*), intent(in) :: val
    real(dp) :: res
    integer :: ios

    if (trim(val) == 'pi') then
      res = PI
    else
      read(val, *, iostat=ios) res
      if (ios /= 0) res = 0.0_dp
    end if
  end function parse_float_or_pi

  !> Convert string to state integer ID
  function state_from_string(str) result(state_id)
    character(len=*), intent(in) :: str
    integer :: state_id

    select case(trim(str))
    case('GS_phys', 'GS_PHYS')
      state_id = STATE_GS_PHYS
    case('GS_cat', 'GS_CAT')
      state_id = STATE_GS_CAT
    case('Phys', 'PHYS')
      state_id = STATE_PHYS
    case('CatSum', 'CAT_SUM')
      state_id = STATE_CAT_SUM
    case default
      state_id = -1
    end select
  end function state_from_string

  !> Get label for a state ID
  function state_label(state_id) result(label)
    integer, intent(in) :: state_id
    character(len=10) :: label

    if (state_id >= 1 .and. state_id <= 4) then
      label = STATE_LABELS(state_id)
    else
      label = 'UNKNOWN'
    end if
  end function state_label

  !> Parse command line arguments (overrides config file values)
  subroutine parse_command_line(params)
    type(SimulationParams_type), intent(inout) :: params

    integer :: i, nargs
    character(len=256) :: arg, val

    nargs = command_argument_count()
    i = 1
    do while (i <= nargs)
      call get_command_argument(i, arg)
      select case(trim(arg))
      case('--system-size')
        i = i + 1
        call get_command_argument(i, val)
        read(val, *) params%N
      case('--x-coupling')
        i = i + 1
        call get_command_argument(i, val)
        read(val, *) params%B
      case('--config')
        ! Already handled before this call
        i = i + 1  ! skip value
      case('--init-state')
        ! Handled separately in main
        i = i + 1  ! skip value
      end select
      i = i + 1
    end do
  end subroutine parse_command_line

end module lmg_config
