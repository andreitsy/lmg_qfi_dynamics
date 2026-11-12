!> Run configuration: defaults, `parameters.ini` and command-line options.
!>
!> The solver reads the same `parameters.ini` as the Python scripts
!> ([Simulation] and [Files] sections, same key names). Precedence:
!> built-in defaults < ini file < command line.
module lmg_config
   use, intrinsic :: iso_fortran_env, only: error_unit
   use lmg_kinds, only: wp, pi
   use lmg_operators, only: lmg_model
   use lmg_states, only: n_state_kinds, state_from_name
   implicit none
   private

   public :: run_config, init_config, load_ini_file, apply_setting
   public :: parse_command_line, print_usage, parse_real, parse_int, validate_config

   type :: run_config
      type(lmg_model) :: model
      integer :: num_points = 200                    !< points per logarithmic segment
      integer :: dps = 15                            !< read for compatibility; precision is fixed
      integer :: max_degree = 0                      !< last decade; <= 0: from the gap of H0
      integer :: nthreads = 0                        !< <= 0: OpenMP default
      logical :: states(n_state_kinds) = .true.      !< initial states to simulate
      logical :: gs_up = .true.                      !< GS_phys with <Sz> > 0 (else < 0)
      logical :: has_b_init = .false.                !< initial eigenstates prepared at b_init /= B
      real(wp) :: b_init = 0.0_wp
      logical :: show_help = .false.
      character(len=:), allocatable :: output_dir
      character(len=:), allocatable :: prefix
      character(len=:), allocatable :: times_file
      character(len=:), allocatable :: config_file
   end type run_config

contains

   subroutine init_config(cfg)
      type(run_config), intent(out) :: cfg

      cfg%output_dir = 'results'
      cfg%prefix = 'fortran.'
      cfg%times_file = ''
      cfg%config_file = ''
   end subroutine init_config

   pure function lower(s) result(t)
      character(len=*), intent(in) :: s
      character(len=len(s)) :: t
      integer :: i, c

      do i = 1, len(s)
         c = iachar(s(i:i))
         if (c >= iachar('A') .and. c <= iachar('Z')) c = c + 32
         t(i:i) = achar(c)
      end do
   end function lower

   !> Parse an integer; `ok` is false for anything but an optionally signed
   !> string of digits.
   subroutine parse_int(text, value, ok)
      character(len=*), intent(in) :: text
      integer, intent(out)         :: value
      logical, intent(out)         :: ok
      character(len=:), allocatable :: s
      integer :: ios

      value = 0
      ok = .false.
      s = trim(adjustl(text))
      if (len(s) == 0) return
      if (verify(s, '+-0123456789') /= 0) return
      read (s, *, iostat=ios) value
      ok = ios == 0
   end subroutine parse_int

   !> Parse a real number in working precision. Accepts decimal literals,
   !> the constant `pi` and products/quotients of them, e.g. `0.4`, `pi`,
   !> `pi/2`, `2*pi`, `-0.5*pi`. Decimal literals are converted directly to
   !> quadruple precision (no detour through double precision).
   subroutine parse_real(text, value, ok)
      character(len=*), intent(in) :: text
      real(wp), intent(out)        :: value
      logical, intent(out)         :: ok

      character(len=:), allocatable :: s, tok
      character(len=1) :: op
      real(wp) :: term, sgn
      integer :: i, pos, next, ios

      value = 0.0_wp
      ok = .false.
      s = ''
      do i = 1, len_trim(text)
         if (text(i:i) /= ' ' .and. text(i:i) /= achar(9)) s = s//text(i:i)
      end do
      s = lower(s)
      if (len(s) == 0) return

      sgn = 1.0_wp
      if (s(1:1) == '-') then
         sgn = -1.0_wp
         s = s(2:)
      else if (s(1:1) == '+') then
         s = s(2:)
      end if
      if (len(s) == 0) return

      value = 1.0_wp
      op = '*'
      pos = 1
      do
         next = scan(s(pos:), '*/')
         if (next == 0) then
            tok = s(pos:)
         else
            tok = s(pos:pos + next - 2)
         end if
         if (len(tok) == 0) return
         if (tok == 'pi') then
            term = pi
         else
            if (verify(tok, '0123456789.+-edq') /= 0) return
            if (scan(tok, '0123456789') == 0) return
            read (tok, *, iostat=ios) term
            if (ios /= 0) return
         end if
         if (op == '*') then
            value = value*term
         else
            if (term == 0.0_wp) return
            value = value/term
         end if
         if (next == 0) exit
         op = s(pos + next - 1:pos + next - 1)
         pos = pos + next
         if (pos > len(s)) return
      end do
      value = sgn*value
      ok = .true.
   end subroutine parse_real

   !> Apply one `key = value` setting. `base_dir` is prepended to relative
   !> paths (used for values that come from an ini file).
   !> ierr: 0 ok, 1 unknown key, 2 bad value.
   subroutine apply_setting(cfg, key_in, value_in, base_dir, ierr)
      type(run_config), intent(inout) :: cfg
      character(len=*), intent(in)    :: key_in, value_in, base_dir
      integer, intent(out)            :: ierr

      character(len=:), allocatable :: key, val
      logical :: ok
      integer :: i

      ierr = 0
      ok = .true.
      key = lower(trim(adjustl(key_in)))
      do i = 1, len(key)
         if (key(i:i) == '_') key(i:i) = '-'
      end do
      val = trim(adjustl(value_in))

      select case (key)
      case ('n', 'system-size')
         call parse_int(val, cfg%model%nspins, ok)
      case ('j')
         call parse_real(val, cfg%model%j, ok)
      case ('b', 'x-coupling')
         call parse_real(val, cfg%model%b, ok)
      case ('b-init')
         call parse_real(val, cfg%b_init, ok)
         cfg%has_b_init = ok
      case ('t')
         call parse_real(val, cfg%model%t, ok)
      case ('phi-kick-phase')
         call parse_real(val, cfg%model%phi_kick, ok)
      case ('h')
         call parse_real(val, cfg%model%h, ok)
      case ('frequency', 'nu')
         call parse_int(val, cfg%model%nu, ok)
      case ('phi-0')
         call parse_real(val, cfg%model%phi0, ok)
      case ('varphi')
         call parse_real(val, cfg%model%varphi, ok)
      case ('theta')
         call parse_real(val, cfg%model%theta, ok)
      case ('steps-floquet-unitary', 'steps')
         call parse_int(val, cfg%model%steps, ok)
      case ('num-points')
         call parse_int(val, cfg%num_points, ok)
      case ('dps')
         call parse_int(val, cfg%dps, ok)
      case ('max-degree')
         call parse_int(val, cfg%max_degree, ok)
      case ('threads')
         call parse_int(val, cfg%nthreads, ok)
      case ('output-dir')
         cfg%output_dir = resolve(val)
         ok = len(val) > 0
      case ('times-file')
         cfg%times_file = resolve(val)
      case ('prefix')
         cfg%prefix = val
      case ('log-file')
         continue      ! Python-only option: the solver logs to standard output
      case ('gs-phys-orientation')
         select case (lower(val))
         case ('up', '+', '+1', '1')
            cfg%gs_up = .true.
         case ('down', '-', '-1')
            cfg%gs_up = .false.
         case default
            ok = .false.
         end select
      case ('init-state')
         call select_states(cfg, val, ok)
      case default
         ierr = 1
         return
      end select
      if (.not. ok) ierr = 2

   contains

      function resolve(path) result(full)
         character(len=*), intent(in) :: path
         character(len=:), allocatable :: full

         if (len(path) == 0 .or. len_trim(base_dir) == 0) then
            full = path
         else if (path(1:1) == '/') then
            full = path
         else
            full = trim(base_dir)//'/'//path
         end if
      end function resolve

   end subroutine apply_setting

   !> Restrict the run to a comma-separated list of initial states.
   subroutine select_states(cfg, list, ok)
      type(run_config), intent(inout) :: cfg
      character(len=*), intent(in)    :: list
      logical, intent(out)            :: ok
      logical :: chosen(n_state_kinds)
      integer :: pos, next, kind

      ok = .false.
      chosen = .false.
      pos = 1
      do while (pos <= len(list))
         next = index(list(pos:), ',')
         if (next == 0) next = len(list) - pos + 2
         kind = state_from_name(trim(adjustl(list(pos:pos + next - 2))))
         if (kind == 0) return
         chosen(kind) = .true.
         pos = pos + next
      end do
      if (.not. any(chosen)) return
      cfg%states = chosen
      ok = .true.
   end subroutine select_states

   !> Read an ini file in the format of Python's configparser. Unknown keys
   !> are reported and ignored. ierr /= 0 if the file cannot be read or a
   !> value is malformed.
   subroutine load_ini_file(path, cfg, ierr)
      character(len=*), intent(in)    :: path
      type(run_config), intent(inout) :: cfg
      integer, intent(out)            :: ierr

      character(len=4096) :: line
      character(len=:), allocatable :: base_dir, key, val
      integer :: unit, ios, pos, slash, stat, lineno

      ierr = 0
      slash = index(path, '/', back=.true.)
      if (slash > 1) then
         base_dir = path(1:slash - 1)
      else if (slash == 1) then
         base_dir = '/'
      else
         base_dir = ''
      end if

      open (newunit=unit, file=path, status='old', action='read', iostat=ios)
      if (ios /= 0) then
         write (error_unit, '(a)') 'error: cannot open configuration file '//trim(path)
         ierr = 1
         return
      end if
      lineno = 0
      do
         read (unit, '(a)', iostat=ios) line
         if (ios /= 0) exit
         lineno = lineno + 1
         line = adjustl(line)
         if (len_trim(line) == 0) cycle
         if (line(1:1) == ';' .or. line(1:1) == '#' .or. line(1:1) == '[') cycle
         pos = scan(line, '=:')
         if (pos <= 1) cycle
         key = trim(line(1:pos - 1))
         val = trim(adjustl(line(pos + 1:)))
         call apply_setting(cfg, key, val, base_dir, stat)
         if (stat == 1) then
            write (error_unit, '(a,i0,a)') 'warning: '//trim(path)//' line ', lineno, &
               ': unknown key "'//key//'" ignored'
         else if (stat == 2) then
            write (error_unit, '(a,i0,a)') 'error: '//trim(path)//' line ', lineno, &
               ': invalid value "'//val//'" for "'//key//'"'
            ierr = 2
         end if
      end do
      close (unit)
   end subroutine load_ini_file

   !> Command line. `--config` is honoured first (default: ./parameters.ini
   !> if it exists), then the remaining options override it in order.
   subroutine parse_command_line(cfg, ierr)
      type(run_config), intent(inout) :: cfg
      integer, intent(out)            :: ierr

      character(len=:), allocatable :: arg, opt, val, key
      integer :: nargs, i, eq, stat
      logical :: has_val, exists, first_state

      ierr = 0
      nargs = command_argument_count()

      ! Pass 1: configuration file and help.
      i = 1
      do while (i <= nargs)
         call get_arg(i, arg)
         if (arg == '--help' .or. arg == '-h') then
            cfg%show_help = .true.
            return
         else if (arg == '--config') then
            if (i == nargs) then
               write (error_unit, '(a)') 'error: --config needs a file name'
               ierr = 1
               return
            end if
            call get_arg(i + 1, cfg%config_file)
            i = i + 1
         else if (len(arg) > 9) then
            if (arg(1:9) == '--config=') cfg%config_file = arg(10:)
         end if
         i = i + 1
      end do
      if (len(cfg%config_file) == 0) then
         inquire (file='parameters.ini', exist=exists)
         if (exists) cfg%config_file = 'parameters.ini'
      end if
      if (len(cfg%config_file) > 0) then
         call load_ini_file(cfg%config_file, cfg, stat)
         if (stat /= 0) then
            ierr = 1
            return
         end if
      end if

      ! Pass 2: everything else.
      first_state = .true.
      i = 1
      do while (i <= nargs)
         call get_arg(i, arg)
         i = i + 1
         if (len(arg) < 3) then
            call bad('unexpected argument "'//arg//'"')
            return
         end if
         if (arg(1:2) /= '--') then
            call bad('unexpected argument "'//arg//'"')
            return
         end if
         eq = index(arg, '=')
         has_val = eq > 0
         if (has_val) then
            opt = arg(3:eq - 1)
            val = arg(eq + 1:)
         else
            opt = arg(3:)
            if (i > nargs) then
               call bad('option --'//opt//' needs a value')
               return
            end if
            call get_arg(i, val)
            i = i + 1
         end if

         key = ''
         select case (opt)
         case ('config')
            cycle
         case ('set')
            eq = index(val, '=')
            if (eq <= 1) then
               call bad('--set expects KEY=VALUE, got "'//val//'"')
               return
            end if
            key = val(1:eq - 1)
            val = val(eq + 1:)
         case ('system-size', 'x-coupling', 'steps', 'num-points', 'max-degree', 'times-file', &
               'output-dir', 'prefix', 'threads', 'gs-phys-orientation', 'init-state')
            key = opt
         case default
            call bad('unknown option --'//opt)
            return
         end select

         if (key == 'init-state' .or. key == 'init_state') then
            ! the first --init-state replaces the default (all states), later ones add
            call add_states(val, first_state, stat)
            first_state = .false.
         else
            call apply_setting(cfg, key, val, '', stat)
         end if
         if (stat == 1) then
            call bad('unknown setting "'//key//'"')
            return
         else if (stat == 2) then
            call bad('invalid value "'//val//'" for '//key)
            return
         end if
      end do

   contains

      subroutine get_arg(pos, text)
         integer, intent(in) :: pos
         character(len=:), allocatable, intent(out) :: text
         integer :: length

         call get_command_argument(pos, length=length)
         allocate (character(len=length) :: text)
         if (length > 0) call get_command_argument(pos, value=text)
      end subroutine get_arg

      subroutine add_states(list, replace, stat)
         character(len=*), intent(in) :: list
         logical, intent(in)          :: replace
         integer, intent(out)         :: stat
         logical :: previous(n_state_kinds), ok

         previous = cfg%states
         call select_states(cfg, list, ok)
         stat = 0
         if (.not. ok) then
            stat = 2
         else if (.not. replace) then
            cfg%states = cfg%states .or. previous
         end if
      end subroutine add_states

      subroutine bad(text)
         character(len=*), intent(in) :: text

         write (error_unit, '(a)') 'error: '//text//' (see --help)'
         ierr = 1
      end subroutine bad

   end subroutine parse_command_line

   !> Sanity checks; ierr /= 0 and a message on stderr if the run cannot start.
   subroutine validate_config(cfg, ierr)
      type(run_config), intent(in) :: cfg
      integer, intent(out)         :: ierr

      ierr = 0
      if (cfg%model%nspins < 1) call fail('the number of spins N must be at least 1')
      if (cfg%model%nu < 1) call fail('frequency (nu) must be a positive integer')
      if (cfg%model%steps < 2) call fail('steps-floquet-unitary must be at least 2')
      if (cfg%model%t <= 0.0_wp) call fail('the period T must be positive')
      if (cfg%num_points < 0) call fail('num-points must not be negative')
      if (len(cfg%output_dir) == 0) call fail('output-dir must not be empty')

   contains

      subroutine fail(text)
         character(len=*), intent(in) :: text

         write (error_unit, '(a)') 'error: '//text
         ierr = 1
      end subroutine fail

   end subroutine validate_config

   subroutine print_usage(unit)
      integer, intent(in) :: unit

      write (unit, '(a)') 'lmg_qfi - QFI dynamics of the kicked LMG model (quadruple precision, OpenMP)'
      write (unit, '(a)') ''
      write (unit, '(a)') 'usage: lmg_qfi [options]'
      write (unit, '(a)') ''
      write (unit, '(a)') 'Parameters are read from parameters.ini (same file and keys as the Python'
      write (unit, '(a)') 'scripts) and can be overridden on the command line.'
      write (unit, '(a)') ''
      write (unit, '(a)') '  --config FILE            ini file (default: ./parameters.ini if present)'
      write (unit, '(a)') '  --system-size N          number of spins'
      write (unit, '(a)') '  --x-coupling B           transverse field'
      write (unit, '(a)') '  --init-state NAME[,..]   Phys, GS_phys, GS_cat, CatSum (default: all four)'
      write (unit, '(a)') '  --steps K                Trotter steps per period (steps-floquet-unitary)'
      write (unit, '(a)') '  --num-points P           points per logarithmic segment of the time grid'
      write (unit, '(a)') '  --max-degree D           last decade of the time grid (default: from the gap)'
      write (unit, '(a)') '  --times-file FILE        take the times from the first column of FILE'
      write (unit, '(a)') '                           (e.g. a result CSV) instead of the default grid'
      write (unit, '(a)') '  --output-dir DIR         where the CSV files are written'
      write (unit, '(a)') '  --prefix STR             file name prefix (default: "fortran.")'
      write (unit, '(a)') '  --gs-phys-orientation S  up (default, <Sz> > 0) or down for GS_phys'
      write (unit, '(a)') '  --threads T              OpenMP threads (default: OMP_NUM_THREADS)'
      write (unit, '(a)') '  --set KEY=VALUE          any ini key, e.g. --set theta=pi/2 --set h=0.01;'
      write (unit, '(a)') '                           b-init=B0 prepares GS_phys/GS_cat with the field B0'
      write (unit, '(a)') '  --help                   this text'
      write (unit, '(a)') ''
      write (unit, '(a)') 'Output: <output-dir>/<prefix><STATE>_N=<N>_B=<B>.csv with the columns'
      write (unit, '(a)') 'time,m_x,m_y,m_z,qfi,qfi_raw (qfi = qfi_raw/(N*time)**2).'
   end subroutine print_usage

end module lmg_config
