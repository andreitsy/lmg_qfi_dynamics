!> QFI dynamics: |psi(t)>, d|psi(t)>/dh, magnetisation and quantum Fisher
!> information for a list of stroboscopic times and initial states.
module lmg_dynamics
   use, intrinsic :: iso_fortran_env, only: int64
   use lmg_kinds, only: wp, dp, ik
   use lmg_linalg, only: unitarity_defect
   use lmg_operators, only: lmg_model, ladder_coefficients, magnetization
   use lmg_floquet, only: floquet_ops, power_table, build_floquet, build_power_table, propagate
   implicit none
   private

   public :: qfi_point, qfi_diagnostics, qfi_dynamics, evaluate_point, wall_time

   !> One row of the output table.
   type :: qfi_point
      integer(ik) :: time = 0_ik
      real(wp) :: mx = 0.0_wp        !< <Sx>/N
      real(wp) :: my = 0.0_wp        !< <Sy>/N
      real(wp) :: mz = 0.0_wp        !< <Sz>/N
      real(wp) :: qfi = 0.0_wp       !< F_h/(N t)^2
      real(wp) :: qfi_raw = 0.0_wp   !< F_h
   end type qfi_point

   !> Accuracy indicators and wall-clock timings of a run.
   type :: qfi_diagnostics
      integer  :: nlev = 0                      !< stored powers U_F**(2**k)
      real(wp) :: exp_per_period = 1.0_wp       !< distinct Trotter exponentials per kicked period
      real(wp) :: unitarity_uf = 0.0_wp         !< max |U_F^H U_F - 1|
      real(wp) :: unitarity_top = 0.0_wp        !< same for the highest stored power
      real(wp) :: max_norm_error = 0.0_wp       !< max |<psi|psi> - 1| over all points
      real(wp) :: max_overlap_error = 0.0_wp    !< max |2 Re<psi|dpsi>| / ||dpsi||
      real(wp) :: max_amplification = 0.0_wp    !< max ||dpsi|| / ||dpsi_perp|| (see evaluate_point)
      real(wp) :: qfi_error = 0.0_wp            !< estimated max relative error of F_h
      real(wp) :: t_floquet = 0.0_wp
      real(wp) :: t_powers = 0.0_wp
      real(wp) :: t_points = 0.0_wp
   end type qfi_diagnostics

contains

   !> Wall-clock time in seconds.
   function wall_time() result(t)
      real(wp) :: t
      integer(int64) :: count, rate

      call system_clock(count, rate)
      t = real(count, wp)/real(rate, wp)
   end function wall_time

   !> Observables of one evolved state.
   !>
   !> F_h = 4 (<dpsi|dpsi> - |<psi|dpsi>|^2), evaluated as 4 ||dpsi - <psi|dpsi> psi||^2,
   !> which is the same number for a normalised state but free of cancellation.
   !>
   !> F_h only depends on the component dpsi_perp of dpsi orthogonal to psi.
   !> An error of dpsi relative to ||dpsi|| therefore becomes a relative error
   !> of F_h larger by 2 ||dpsi|| / ||dpsi_perp||; this `amplification` is
   !> returned (0 if F_h is below 1e-30 of 4 ||dpsi||^2, i.e. zero to working
   !> precision, where a relative error is not meaningful).
   pure subroutine evaluate_point(nspins, c, t, psi, dpsi, point, norm_error, overlap_error, amplification)
      integer, intent(in)          :: nspins
      real(wp), intent(in)         :: c(:)
      integer(ik), intent(in)      :: t
      complex(wp), intent(in)      :: psi(:), dpsi(:)
      type(qfi_point), intent(out) :: point
      real(wp), intent(out)        :: norm_error, overlap_error, amplification

      complex(wp) :: overlap, d
      real(wp) :: perp2, dnorm2
      integer :: i

      overlap = dot_product(psi, dpsi)
      perp2 = 0.0_wp
      dnorm2 = 0.0_wp
      do i = 1, size(psi)
         d = dpsi(i) - overlap*psi(i)
         perp2 = perp2 + real(d, wp)**2 + aimag(d)**2
         dnorm2 = dnorm2 + real(dpsi(i), wp)**2 + aimag(dpsi(i))**2
      end do

      point%time = t
      point%qfi_raw = 4.0_wp*perp2
      if (t > 0_ik) then
         point%qfi = point%qfi_raw/(real(nspins, wp)*real(t, wp))**2
      else
         point%qfi = 0.0_wp
      end if
      call magnetization(nspins, c, psi, point%mx, point%my, point%mz)

      norm_error = abs(real(dot_product(psi, psi), wp) - 1.0_wp)
      overlap_error = 0.0_wp
      if (dnorm2 > 0.0_wp) overlap_error = abs(2.0_wp*real(overlap, wp))/sqrt(dnorm2)
      amplification = 0.0_wp
      if (perp2 > 1.0e-30_wp*dnorm2) amplification = sqrt(dnorm2/perp2)
   end subroutine evaluate_point

   !> Run the whole calculation for normalised initial states psi0(:, s).
   !>
   !> 1. Floquet operators P_1..P_nu and dP/dh       (threads: Trotter steps)
   !> 2. U_F**(2**k) and derivatives by squaring     (threads: matrix products)
   !> 3. all (time, state) pairs                     (threads: time points)
   subroutine qfi_dynamics(mdl, times, psi0, res, diag, log_unit)
      type(lmg_model), intent(in)        :: mdl
      integer(ik), intent(in)            :: times(:)
      complex(wp), intent(in)            :: psi0(:, :)
      type(qfi_point), intent(out)       :: res(:, :)
      type(qfi_diagnostics), intent(out) :: diag
      integer, intent(in), optional      :: log_unit

      type(floquet_ops) :: fl
      type(power_table) :: pt
      real(wp), allocatable :: c(:)
      real(wp) :: t0, t1, max_norm, max_overlap, max_amp, max_qfi_err
      integer(ik) :: tmax
      integer :: n, nt, ns, it, lu

      lu = -1
      if (present(log_unit)) lu = log_unit
      n = mdl%nspins + 1
      nt = size(times)
      ns = size(psi0, 2)
      if (size(psi0, 1) /= n) error stop 'qfi_dynamics: initial states have the wrong dimension'
      if (size(res, 1) /= nt .or. size(res, 2) /= ns) error stop 'qfi_dynamics: result array has the wrong shape'

      allocate (c(mdl%nspins))
      call ladder_coefficients(mdl%nspins, c)
      tmax = 0_ik
      if (nt > 0) tmax = maxval(times)

      t0 = wall_time()
      call build_floquet(mdl, fl, derivative=.true.)
      t1 = wall_time()
      diag%t_floquet = t1 - t0
      diag%unitarity_uf = unitarity_defect(fl%u(:, :, mdl%nu))
      diag%exp_per_period = real(fl%nexp, wp)/real(mdl%nu, wp)
      call log_line('Floquet operator and its h-derivative', diag%t_floquet)

      t0 = wall_time()
      call build_power_table(fl, tmax/int(mdl%nu, ik), pt)
      t1 = wall_time()
      diag%t_powers = t1 - t0
      diag%nlev = pt%nlev
      if (pt%nlev > 0) then
         diag%unitarity_top = unitarity_defect(pt%a(:, :, pt%nlev))
      else
         diag%unitarity_top = diag%unitarity_uf
      end if
      call log_line('repeated squaring of the Floquet operator', diag%t_powers)

      t0 = wall_time()
      max_norm = 0.0_wp
      max_overlap = 0.0_wp
      max_amp = 0.0_wp
      max_qfi_err = 0.0_wp
      !$omp parallel do default(shared) private(it) schedule(dynamic, 1) &
      !$omp    reduction(max: max_norm, max_overlap, max_amp, max_qfi_err)
      do it = 1, nt
         call time_point(it, max_norm, max_overlap, max_amp, max_qfi_err)
      end do
      !$omp end parallel do
      t1 = wall_time()
      diag%t_points = t1 - t0
      diag%max_norm_error = max_norm
      diag%max_overlap_error = max_overlap
      diag%max_amplification = max_amp
      diag%qfi_error = max_qfi_err
      call log_line('time points', diag%t_points)

   contains

      !> All initial states at time index `it`.
      !>
      !> Error model of F_h at time t (calibrated against mpmath, see README):
      !>   2 (N+1) eps t (amplification + m).
      !> A rounding error of about (N+1) eps per period shifts the
      !> quasi-energies; the resulting phase error is amplified like any error
      !> of dpsi. The additional error of the m distinct Trotter exponentials
      !> per period (m = 1 for h = 0) grows the state error to (N+1) eps t m
      !> but was found not to be amplified.
      subroutine time_point(it, max_norm, max_overlap, max_amp, max_qfi_err)
         integer, intent(in)     :: it
         real(wp), intent(inout) :: max_norm, max_overlap, max_amp, max_qfi_err
         complex(wp) :: psi(n), dpsi(n)
         real(wp) :: norm_error, overlap_error, amp, err_t
         integer :: is

         err_t = real(n, wp)*epsilon(1.0_wp)*max(1.0_wp, real(times(it), wp))
         do is = 1, ns
            call propagate(fl, pt, times(it), psi0(:, is), psi, dpsi)
            call evaluate_point(mdl%nspins, c, times(it), psi, dpsi, res(it, is), norm_error, overlap_error, amp)
            max_norm = max(max_norm, norm_error)
            max_overlap = max(max_overlap, overlap_error)
            max_amp = max(max_amp, amp)
            max_qfi_err = max(max_qfi_err, 2.0_wp*err_t*(amp + max(1.0_wp, diag%exp_per_period)))
         end do
      end subroutine time_point

      subroutine log_line(what, seconds)
         character(len=*), intent(in) :: what
         real(wp), intent(in)         :: seconds

         if (lu < 0) return
         write (lu, '(2x,a,t48,f10.2,a)') what, real(seconds, dp), ' s'
         flush (lu)
      end subroutine log_line

   end subroutine qfi_dynamics

end module lmg_dynamics
