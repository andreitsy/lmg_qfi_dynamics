!> lmg_qfi: quantum Fisher information dynamics of the kicked LMG model.
!>
!> Fortran/OpenMP port of quantum_fisher_information_simulation_mpmath.py in
!> quadruple precision. See fortran/README.md.
program lmg_qfi
   use, intrinsic :: iso_fortran_env, only: output_unit, error_unit
   use lmg_kinds, only: wp, dp, ik
   use lmg_config, only: run_config, init_config, parse_command_line, validate_config, print_usage
   use lmg_states, only: n_state_kinds, ground_doublet, initial_state, state_cli_name, state_enum_name
   use lmg_timegrid, only: max_time_degree, python_time_grid, read_time_grid
   use lmg_dynamics, only: qfi_point, qfi_diagnostics, qfi_dynamics, wall_time
   use lmg_io, only: write_csv, format_real, format_time, ensure_directory, result_file_name
   !$ use omp_lib, only: omp_get_max_threads, omp_set_num_threads
   implicit none

   type(run_config) :: cfg
   type(qfi_diagnostics) :: diag
   type(qfi_point), allocatable :: res(:, :)
   complex(wp), allocatable :: psi0(:, :), v0(:), v1(:)
   integer(ik), allocatable :: times(:)
   integer, allocatable :: kinds(:)
   character(len=:), allocatable :: path
   character(len=512) :: message
   real(wp) :: e0, e1, gap, t_start, accuracy, accuracy_qfi, b_init
   integer(ik) :: tmax
   integer :: ierr, n, ns, is, ikind, degree, nthreads

   t_start = wall_time()
   call init_config(cfg)
   call parse_command_line(cfg, ierr)
   if (ierr /= 0) stop 2
   if (cfg%show_help) then
      call print_usage(output_unit)
      stop
   end if
   call validate_config(cfg, ierr)
   if (ierr /= 0) stop 2

   nthreads = 1
   !$ if (cfg%nthreads > 0) call omp_set_num_threads(cfg%nthreads)
   !$ nthreads = omp_get_max_threads()

   n = cfg%model%nspins + 1
   write (output_unit, '(a)') 'lmg_qfi: QFI dynamics of the kicked LMG model'
   if (len(cfg%config_file) > 0) write (output_unit, '(2x,a)') 'configuration   '//cfg%config_file
   write (output_unit, '(2x,a,i0)') 'N               ', cfg%model%nspins
   write (output_unit, '(2x,a)') 'J, B, T         '//format_real(cfg%model%j, 12)//', '// &
      format_real(cfg%model%b, 12)//', '//format_real(cfg%model%t, 12)
   write (output_unit, '(2x,a)') 'kick angle      '//format_real(cfg%model%phi_kick, 12)
   write (output_unit, '(2x,a)') 'probe h, phi0   '//format_real(cfg%model%h, 12)//', '// &
      format_real(cfg%model%phi0, 12)
   write (output_unit, '(2x,a)') 'theta, varphi   '//format_real(cfg%model%theta, 12)//', '// &
      format_real(cfg%model%varphi, 12)
   write (output_unit, '(2x,a,i0,a,i0)') 'nu, steps       ', cfg%model%nu, ', ', cfg%model%steps
   write (output_unit, '(2x,a,i0,a,i0,a)') 'precision       ', precision(1.0_wp), &
      ' decimal digits (', digits(1.0_wp), '-bit significand)'
   write (output_unit, '(2x,a,i0)') 'OpenMP threads  ', nthreads

   ! Ground-state doublet of H0: initial states and the time range.
   allocate (v0(n), v1(n))
   b_init = cfg%model%b
   if (cfg%has_b_init) then
      b_init = cfg%b_init
      write (output_unit, '(2x,a)') 'B of GS states  '//format_real(b_init, 34)
   end if
   call ground_doublet(cfg%model%nspins, cfg%model%j, b_init, e0, e1, v0, v1)
   gap = e1 - e0
   write (output_unit, '(2x,a)') 'gap E1 - E0     '//format_real(gap, 12)

   if (len(cfg%times_file) > 0) then
      call read_time_grid(cfg%times_file, times, ierr, message)
      if (ierr /= 0) then
         write (error_unit, '(a)') 'error: '//trim(message)
         stop 1
      end if
      write (output_unit, '(2x,a)') 'time grid       '//cfg%times_file
   else
      degree = cfg%max_degree
      if (degree <= 0) then
         if (.not. gap > 0.0_wp) then
            write (error_unit, '(a)') 'error: the gap of H0 is not resolved in this precision; '// &
               'set the time range with --max-degree'
            stop 1
         end if
         degree = max_time_degree(gap)
      end if
      if (degree <= 1 .or. degree > 38) then
         write (error_unit, '(a,i0,a)') 'error: max_degree = ', degree, &
            ' is outside 2..38; choose the last decade with --max-degree'
         stop 1
      end if
      call python_time_grid(cfg%num_points, degree, times)
      write (output_unit, '(2x,a,i0,a,i0)') 'time grid       num-points = ', cfg%num_points, &
         ', max degree = ', degree
   end if
   tmax = 0_ik
   if (size(times) > 0) tmax = maxval(times)
   write (output_unit, '(2x,a,i0,a)') 'time points     ', size(times), ' (largest t = '//format_time(tmax)//')'

   ns = count(cfg%states)
   allocate (kinds(ns), psi0(n, ns), res(size(times), ns))
   is = 0
   do ikind = 1, n_state_kinds
      if (.not. cfg%states(ikind)) cycle
      is = is + 1
      kinds(is) = ikind
      call initial_state(ikind, cfg%model%nspins, v0, v1, cfg%gs_up, psi0(:, is))
   end do

   write (output_unit, '(a)') 'stages (wall-clock):'
   call qfi_dynamics(cfg%model, times, psi0, res, diag, log_unit=output_unit)

   ! Accuracy report. Every period adds the rounding error of U_F again, so
   ! the error grows linearly with time: about (N+1) eps t m for the state,
   ! with m distinct Trotter exponentials per period (m = 1 for h = 0, where
   ! a period is one closed-form exponential; about `steps` otherwise). The
   ! measured loss of unitarity and of the norm is used instead when it is
   ! larger. The estimate for F_h, which can be less accurate than the state
   ! (see evaluate_point), is evaluated point by point in qfi_dynamics.
   accuracy = real(n, wp)*epsilon(1.0_wp)*max(1.0_wp, real(tmax, wp))*max(1.0_wp, diag%exp_per_period)
   write (output_unit, '(a)') 'accuracy:'
   call report('unitarity defect of U_F', diag%unitarity_uf)
   write (message, '(a,i0,a)') 'unitarity defect of U_F**(2**', max(diag%nlev - 1, 0), ')'
   call report(trim(message), diag%unitarity_top)
   call report('max | <psi|psi> - 1 |', diag%max_norm_error)
   call report('max | 2 Re <psi|dpsi> | / ||dpsi||', diag%max_overlap_error)
   call report('max ||dpsi|| / ||dpsi_perp||', diag%max_amplification)
   write (message, '(a,i0,a)') 'rounding estimate (N+1) eps t_max m, m=', nint(diag%exp_per_period), ':'
   call report(trim(message), accuracy)
   accuracy_qfi = diag%qfi_error
   if (accuracy > 0.0_wp) accuracy_qfi = accuracy_qfi*max(1.0_wp, max(diag%unitarity_top, diag%max_norm_error)/accuracy)
   accuracy = max(accuracy, diag%unitarity_top, diag%max_norm_error)
   call report('=> psi, m_x, m_y, m_z accurate to about', accuracy)
   call report('=> qfi, qfi_raw: relative accuracy about', accuracy_qfi)
   if (max(accuracy, accuracy_qfi) > 1.0e-6_wp) then
      write (error_unit, '(a)') 'warning: the largest times exhaust quadruple precision; results near t_max '// &
         'are accurate to about the figures printed under "accuracy" only.'
   end if

   call ensure_directory(cfg%output_dir, ierr)
   if (ierr /= 0) then
      write (error_unit, '(a)') 'error: cannot create output directory '//cfg%output_dir
      stop 1
   end if
   write (output_unit, '(a)') 'results:'
   do is = 1, ns
      path = result_file_name(cfg%output_dir, cfg%prefix, state_enum_name(kinds(is)), &
                              cfg%model%nspins, cfg%model%b)
      call write_csv(path, res(:, is), ierr)
      if (ierr /= 0) then
         write (error_unit, '(a)') 'error: cannot write '//path
         stop 1
      end if
      write (output_unit, '(2x,a)') state_cli_name(kinds(is))//repeat(' ', 9 - len(state_cli_name(kinds(is))))//path
   end do
   write (output_unit, '(a,f0.2,a)') 'total wall-clock time: ', real(wall_time() - t_start, dp), ' s'

contains

   subroutine report(label, value)
      character(len=*), intent(in) :: label
      real(wp), intent(in)         :: value
      character(len=44) :: padded

      padded = label
      write (output_unit, '(2x,a,es10.2)') padded, real(value, dp)
   end subroutine report

end program lmg_qfi
