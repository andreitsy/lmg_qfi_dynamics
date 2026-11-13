!> Self-contained test suite of the Fortran solver (`make test`).
!>
!> Every check compares against something independent of the code path
!> under test: operator identities, closed-form results, finite differences
!> in h, step-by-step propagation, or values produced by NumPy.
program test_lmg
   use, intrinsic :: iso_fortran_env, only: output_unit
   use lmg_kinds, only: wp, dp, ik, pi, czero, ci
   use lmg_linalg, only: matmul_omp, matvec, adjoint, identity_matrix, herm_eig, max_abs, unitarity_defect
   use lmg_operators, only: lmg_model, ladder_coefficients, spin_operators, hamiltonian_h0, &
                            probe_operator, magnetization, apply_mom_observable, obs_x, obs_y
   use lmg_floquet, only: floquet_ops, power_table, build_floquet, build_power_table, propagate
   use lmg_states, only: ground_doublet, initial_state, state_from_name, state_enum_name, &
                         st_phys, st_gs_phys, st_gs_cat, st_cat_sum
   use lmg_timegrid, only: max_time_degree, python_time_grid, logspace_int, read_time_grid
   use lmg_dynamics, only: qfi_point, qfi_diagnostics, qfi_dynamics, method_of_moments
   use lmg_config, only: run_config, init_config, apply_setting, parse_real, parse_int
   use lmg_io, only: format_time, format_real, format_fixed2, result_file_name, write_csv
   implicit none

   integer :: nchecks = 0, nfailed = 0
   integer :: seed = 20261111

   call test_spin_algebra()
   call test_herm_eig()
   call test_kick_operator()
   call test_floquet_derivative()
   call test_propagation()
   call test_closed_form_qfi()
   call test_qfi_against_finite_difference()
   call test_method_of_moments()
   call test_ground_doublet()
   call test_time_grid()
   call test_config_and_io()

   write (output_unit, '(a,i0,a,i0,a)') 'test_lmg: ', nchecks - nfailed, ' of ', nchecks, ' checks passed'
   if (nfailed > 0) error stop 1

contains

   ! ------------------------------------------------------------------
   ! helpers
   ! ------------------------------------------------------------------

   subroutine check(condition, name)
      logical, intent(in)          :: condition
      character(len=*), intent(in) :: name

      nchecks = nchecks + 1
      if (condition) then
         write (output_unit, '(2x,a,1x,a)') 'ok  ', name
      else
         nfailed = nfailed + 1
         write (output_unit, '(2x,a,1x,a)') 'FAIL', name
      end if
   end subroutine check

   !> Passes if 0 <= value <= tol (NaN fails).
   subroutine check_small(value, tol, name)
      real(wp), intent(in)         :: value, tol
      character(len=*), intent(in) :: name
      character(len=160) :: text

      write (text, '(a,a,es9.2,a,es9.2,a)') name, '  [', real(value, dp), ' <= ', real(tol, dp), ']'
      call check(value >= 0.0_wp .and. value <= tol, trim(text))
   end subroutine check_small

   !> Deterministic pseudo-random number in (-1, 1).
   function rnd() result(x)
      real(wp) :: x

      seed = modulo(seed*1103515245 + 12345, 2147483647)
      x = 2.0_wp*real(seed, wp)/2147483647.0_wp - 1.0_wp
   end function rnd

   function model(nspins, j, b, h, nu, steps, theta, varphi, phi0, phi_kick) result(mdl)
      integer, intent(in)  :: nspins, nu, steps
      real(wp), intent(in) :: j, b, h, theta, varphi, phi0, phi_kick
      type(lmg_model) :: mdl

      mdl%nspins = nspins
      mdl%j = j
      mdl%b = b
      mdl%h = h
      mdl%nu = nu
      mdl%steps = steps
      mdl%theta = theta
      mdl%varphi = varphi
      mdl%phi0 = phi0
      mdl%phi_kick = phi_kick
      mdl%t = 1.0_wp
   end function model

   ! ------------------------------------------------------------------
   ! tests
   ! ------------------------------------------------------------------

   subroutine test_spin_algebra()
      integer, parameter :: nspins = 7, n = nspins + 1
      complex(wp) :: sx(n, n), sy(n, n), sz(n, n), a(n, n), b(n, n), c(n, n), sa(n, n)
      real(wp) :: s

      write (output_unit, '(a)') 'spin operators'
      call spin_operators(nspins, sx, sy, sz)
      call matmul_omp(sx, sy, a)
      call matmul_omp(sy, sx, b)
      call check_small(max_abs(a - b - ci*sz), 1.0e-31_wp, '[Sx, Sy] = i Sz')
      call matmul_omp(sy, sz, a)
      call matmul_omp(sz, sy, b)
      call check_small(max_abs(a - b - ci*sx), 1.0e-31_wp, '[Sy, Sz] = i Sx')
      call matmul_omp(sx, sx, a)
      call matmul_omp(sy, sy, b)
      call matmul_omp(sz, sz, c)
      s = 0.5_wp*real(nspins, wp)
      call check_small(max_abs(a + b + c - s*(s + 1.0_wp)*identity_matrix(n)), 1.0e-30_wp, &
                       'Sx^2 + Sy^2 + Sz^2 = S(S+1)')
      call check_small(max_abs(sx - adjoint(sx)) + max_abs(sy - adjoint(sy)), 0.0_wp, 'Sx, Sy Hermitian')
      call check(real(sz(1, 1), wp) == s .and. real(sz(n, n), wp) == -s, 'Sz = diag(S, ..., -S)')

      call probe_operator(nspins, 0.7_wp, 0.3_wp, sa)
      a = sin(0.7_wp)*cos(0.3_wp)*sx + sin(0.7_wp)*sin(0.3_wp)*sy + cos(0.7_wp)*sz
      call check_small(max_abs(sa - a), 1.0e-32_wp, 'S_alpha for a tilted probe direction')

      call hamiltonian_h0(nspins, 1.3_wp, 0.4_wp, a)
      call matmul_omp(sz, sz, b)
      c = -(2.0_wp*1.3_wp/real(nspins, wp))*b - 2.0_wp*0.4_wp*sx
      call check_small(max_abs(a - c), 1.0e-31_wp, 'H0 = -(2J/N) Sz^2 - 2B Sx')
   end subroutine test_spin_algebra

   subroutine test_herm_eig()
      integer, parameter :: n = 24
      complex(wp) :: a(n, n), v(n, n), av(n, n), vd(n, n), q(n, n), d(n, n), t(n, n)
      real(wp) :: w(n), scale
      integer :: i, k
      logical :: ascending

      write (output_unit, '(a)') 'Hermitian eigen-solver'
      do k = 1, n
         do i = 1, k
            a(i, k) = cmplx(rnd(), rnd(), wp)
            a(k, i) = conjg(a(i, k))
         end do
         a(k, k) = cmplx(real(a(k, k), wp), 0.0_wp, wp)
      end do
      call herm_eig(a, w, v)
      call matmul_omp(a, v, av)
      do k = 1, n
         vd(:, k) = v(:, k)*w(k)
      end do
      scale = max_abs(a)
      call check_small(max_abs(av - vd)/scale, 1.0e-31_wp, 'A V = V diag(w), random complex matrix')
      call check_small(unitarity_defect(v), 1.0e-31_wp, 'V^H V = 1')
      ascending = .true.
      do k = 2, n
         ascending = ascending .and. w(k) >= w(k - 1)
      end do
      call check(ascending, 'eigenvalues in ascending order')

      ! Degenerate spectrum: A = Q D Q^H with D = diag(1, 1, 1, 2, 2, 5, ...)
      q = v
      d = czero
      do k = 1, n
         d(k, k) = cmplx(real(min(k/3, 4), wp), 0.0_wp, wp)
      end do
      call matmul_omp(q, d, t)
      call matmul_omp(t, adjoint(q), a)
      call herm_eig(a, w, v)
      call matmul_omp(a, v, av)
      do k = 1, n
         vd(:, k) = v(:, k)*w(k)
      end do
      call check_small(max_abs(av - vd), 1.0e-30_wp, 'A V = V diag(w), degenerate spectrum')
      call check_small(unitarity_defect(v), 1.0e-31_wp, 'orthonormal eigenvectors within degenerate subspaces')
      call check_small(abs(w(1)) + abs(w(n) - 4.0_wp), 1.0e-30_wp, 'degenerate eigenvalues recovered')
   end subroutine test_herm_eig

   subroutine test_kick_operator()
      integer, parameter :: nspins = 6, n = nspins + 1
      type(lmg_model) :: mdl
      type(floquet_ops) :: fl
      complex(wp) :: expected(n, n), sx(n, n), sy(n, n), sz(n, n), t(n, n)
      integer :: i

      write (output_unit, '(a)') 'kick operator'
      ! J = B = h = 0: one period is the kick alone.
      mdl = model(nspins, 0.0_wp, 0.0_wp, 0.0_wp, 1, 5, 0.0_wp, 0.0_wp, 0.0_wp, pi)
      call build_floquet(mdl, fl, derivative=.false.)
      ! exp(-i pi Sx)|S, m> = (-i)^N |S, -m>
      expected = czero
      do i = 1, n
         expected(n + 1 - i, i) = (-ci)**nspins
      end do
      call check_small(max_abs(fl%u(:, :, 1) - expected), 1.0e-31_wp, 'exp(-i pi Sx) flips m -> -m')

      mdl%phi_kick = 0.37_wp
      call build_floquet(mdl, fl, derivative=.false.)
      call spin_operators(nspins, sx, sy, sz)
      ! K^H Sz K = cos(phi) Sz + sin(phi) Sy for a rotation about x
      call matmul_omp(sz, fl%u(:, :, 1), t)
      call matmul_omp(adjoint(fl%u(:, :, 1)), t, expected)
      call check_small(max_abs(expected - cos(0.37_wp)*sz - sin(0.37_wp)*sy), 1.0e-31_wp, &
                       'kick rotates the spin about x')
   end subroutine test_kick_operator

   !> dP_m/dh against central differences, both for the general path (every
   !> Trotter step has its own Hamiltonian) and for h = 0, where a whole
   !> period is evaluated in closed form in the eigenbasis of H0.
   subroutine test_floquet_derivative()
      integer, parameter :: nspins = 6
      real(wp), parameter :: eps = 1.0e-11_wp
      type(lmg_model) :: mdl
      type(floquet_ops) :: fl, flp, flm
      real(wp) :: err, defect, h
      integer :: m, pass

      write (output_unit, '(a)') 'Floquet operators and their h-derivative'
      do pass = 1, 2
         h = merge(0.05_wp, 0.0_wp, pass == 1)
         mdl = model(nspins, 1.0_wp, 0.4_wp, h, 3, 7, 0.7_wp, 0.3_wp, 0.2_wp, pi)
         call build_floquet(mdl, fl, derivative=.true.)
         call build_floquet(mdl, flp, derivative=.false., h=h + eps)
         call build_floquet(mdl, flm, derivative=.false., h=h - eps)
         err = 0.0_wp
         defect = 0.0_wp
         do m = 1, mdl%nu
            err = max(err, max_abs(fl%du(:, :, m) - (flp%u(:, :, m) - flm%u(:, :, m))/(2.0_wp*eps)))
            defect = max(defect, unitarity_defect(fl%u(:, :, m)))
         end do
         if (pass == 1) then
            call check_small(defect, 1.0e-30_wp, 'P_m unitary (h = 0.05, tilted probe)')
            call check_small(err, 1.0e-18_wp, 'dP_m/dh = central difference (h = 0.05)')
         else
            call check_small(defect, 1.0e-30_wp, 'P_m unitary (h = 0)')
            call check_small(err, 1.0e-18_wp, 'dP_m/dh = central difference (h = 0, closed-form period)')
         end if
      end do
   end subroutine test_floquet_derivative

   !> Binary powering against period-by-period propagation.
   subroutine test_propagation()
      integer, parameter :: nspins = 5, n = nspins + 1, tlast = 45
      type(lmg_model) :: mdl
      type(floquet_ops) :: fl
      type(power_table) :: pt
      complex(wp) :: psi0(n), psi(n), dpsi(n), ref(n), dref(n), w1(n), w2(n), w3(n), psi_only(n)
      complex(wp), allocatable :: wper(:, :, :), dwper(:, :, :)
      real(wp) :: err, derr, err_plain
      integer :: t, i, p, m

      write (output_unit, '(a)') 'propagation'
      mdl = model(nspins, 1.0_wp, 0.4_wp, 0.03_wp, 3, 6, 0.5_wp, 0.2_wp, 0.1_wp, pi)
      call build_floquet(mdl, fl, derivative=.true.)
      call build_power_table(fl, int(tlast, ik)/int(mdl%nu, ik), pt)
      do i = 1, n
         psi0(i) = cmplx(rnd(), rnd(), wp)
      end do
      psi0 = psi0/sqrt(sum(abs(psi0)**2))

      ! Single kicked periods W_p = P_p P_{p-1}^H and dW_p from the product rule.
      allocate (wper(n, n, mdl%nu), dwper(n, n, mdl%nu))
      wper(:, :, 1) = fl%u(:, :, 1)
      dwper(:, :, 1) = fl%du(:, :, 1)
      do p = 2, mdl%nu
         call matmul_omp(fl%u(:, :, p), adjoint(fl%u(:, :, p - 1)), wper(:, :, p))
         ! dP_p = dW_p P_{p-1} + W_p dP_{p-1}  =>  dW_p = (dP_p - W_p dP_{p-1}) P_{p-1}^H
         block
            complex(wp) :: t1(n, n), t2(n, n)
            call matmul_omp(wper(:, :, p), fl%du(:, :, p - 1), t1)
            t2 = fl%du(:, :, p) - t1
            call matmul_omp(t2, adjoint(fl%u(:, :, p - 1)), dwper(:, :, p))
         end block
      end do

      err = 0.0_wp
      derr = 0.0_wp
      err_plain = 0.0_wp
      ref = psi0
      dref = czero
      do t = 1, tlast
         m = modulo(t - 1, mdl%nu) + 1
         call matvec(wper(:, :, m), ref, w1)
         call matvec(dwper(:, :, m), ref, w2)
         call matvec(wper(:, :, m), dref, w3)
         ref = w1
         dref = w2 + w3
         call propagate(fl, pt, int(t, ik), psi0, psi, dpsi)
         call propagate(fl, pt, int(t, ik), psi0, psi_only)
         err = max(err, maxval(abs(psi - ref)))
         derr = max(derr, maxval(abs(dpsi - dref)))
         err_plain = max(err_plain, maxval(abs(psi_only - psi)))
      end do
      call check_small(err, 1.0e-29_wp, 'psi(t), t = 1..45: binary powering = stepwise evolution')
      call check_small(derr, 1.0e-27_wp, 'dpsi(t)/dh, t = 1..45: binary powering = stepwise product rule')
      call check_small(err_plain, 0.0_wp, 'propagation without derivative gives the same state')
      call propagate(fl, pt, 0_ik, psi0, psi, dpsi)
      call check_small(maxval(abs(psi - psi0)) + maxval(abs(dpsi)), 0.0_wp, 't = 0 is the identity')
   end subroutine test_propagation

   !> J = B = 0: H(t) = h s(t) Sz commutes with itself, the kick flips Sz
   !> every period while the drive changes sign every period, hence
   !>   F_h(t) = 4 t^2 C^2 Var(Sz),  C = (T/steps) cot(pi / (2 (steps - 1)))
   !> for nu = 2, phi0 = 0. The cat state has Var(Sz) = N^2/4, the polarised
   !> state Var(Sz) = 0.
   subroutine test_closed_form_qfi()
      integer, parameter :: nspins = 8, n = nspins + 1, steps = 10
      type(lmg_model) :: mdl
      type(qfi_diagnostics) :: diag
      type(qfi_point) :: res(6, 2)
      complex(wp) :: psi0(n, 2), v(n)
      integer(ik) :: times(6)
      real(wp) :: c1, expected, err, mag, zero_qfi
      integer :: it

      write (output_unit, '(a)') 'closed-form QFI (J = B = 0)'
      mdl = model(nspins, 0.0_wp, 0.0_wp, 0.3_wp, 2, steps, 0.0_wp, 0.0_wp, 0.0_wp, pi)
      v = czero
      call initial_state(st_cat_sum, nspins, v, v, .true., psi0(:, 1))
      call initial_state(st_phys, nspins, v, v, .true., psi0(:, 2))
      times = [1_ik, 2_ik, 3_ik, 10_ik, 1001_ik, 1000000000001_ik]
      call qfi_dynamics(mdl, times, psi0, res, diag)

      c1 = (1.0_wp/real(steps, wp))/tan(pi/(2.0_wp*real(steps - 1, wp)))
      err = 0.0_wp
      mag = 0.0_wp
      zero_qfi = 0.0_wp
      do it = 1, size(times)
         expected = (real(nspins, wp)*real(times(it), wp)*c1)**2
         err = max(err, abs(res(it, 1)%qfi_raw/expected - 1.0_wp))
         err = max(err, abs(res(it, 1)%qfi/c1**2 - 1.0_wp))
         mag = max(mag, abs(res(it, 1)%mx), abs(res(it, 1)%my), abs(res(it, 1)%mz))
         zero_qfi = max(zero_qfi, res(it, 2)%qfi)
         mag = max(mag, abs(abs(res(it, 2)%mz) - 0.5_wp))
      end do
      call check_small(err, 1.0e-19_wp, 'cat state: F_h = (N t C)^2 up to t = 1e12 + 1')
      call check_small(zero_qfi, 1.0e-40_wp, 'polarised state: F_h = 0')
      call check_small(mag, 1.0e-19_wp, 'magnetisations of both states')
      call check(res(1, 2)%mz < 0.0_wp .and. res(2, 2)%mz > 0.0_wp, 'the kick flips m_z every period')
      ! cat state: dpsi is orthogonal to psi (amplification 1); polarised
      ! state: F_h = 0 is excluded from the relative error estimate
      call check_small(abs(diag%max_amplification - 1.0_wp), 1.0e-20_wp, 'QFI error amplification ||dpsi||/||dpsi_perp||')
      call check(diag%qfi_error > 0.0_wp .and. diag%qfi_error < 1.0e-18_wp, 'QFI error estimate finite and small')
   end subroutine test_closed_form_qfi

   !> Method of moments: observables, MoM = (d<O>/dh)^2 / Var(O) against central
   !> differences of <O> in h, MoM <= F_h, and the eigenstate (Var = 0) limit.
   subroutine test_method_of_moments()
      integer, parameter :: nspins = 6, n = nspins + 1, nt = 6
      real(wp), parameter :: eps = 1.0e-12_wp
      type(lmg_model) :: mdl
      type(floquet_ops) :: flp, flm
      type(power_table) :: ptp, ptm
      type(qfi_diagnostics) :: diag
      type(qfi_point) :: res(nt, 4)
      complex(wp) :: psi0(n, 4), v0(n), v1(n), kp(n), km(n), psi(n), w(n), sx(n, n), sy(n, n), sz(n, n)
      complex(wp) :: x(n, n), y(n, n), e(n)
      integer(ik) :: times(nt)
      real(wp) :: e0, e1, op, om, mean, var, mom_fd, mom(3), err, bound, sat, herm
      integer :: it, is, k, i, pass

      write (output_unit, '(a)') 'method of moments'
      call spin_operators(nspins, sx, sy, sz)
      do i = 1, n
         e = czero
         e(i) = (1.0_wp, 0.0_wp)
         call apply_mom_observable(nspins, obs_x, e, x(:, i))
         call apply_mom_observable(nspins, obs_y, e, y(:, i))
      end do
      call check_small(max_abs(y - ci*matmul(sz, x)), 0.0_wp, 'Y = i Sz X')
      herm = max_abs(x - adjoint(x)) + max_abs(y - adjoint(y)) + max_abs(matmul(sz, x) + matmul(x, sz))
      call check_small(herm, 0.0_wp, 'X, Y Hermitian and {Sz, X} = 0')
      call check_small(max_abs(matmul(x, x) - identity_matrix(n)), 0.0_wp, 'X^2 = 1')

      times = [1_ik, 3_ik, 10_ik, 25_ik, 64_ik, 101_ik]
      do pass = 1, 2
         if (pass == 1) then
            mdl = model(nspins, 1.0_wp, 0.4_wp, 0.02_wp, 2, 6, 0.4_wp, 0.9_wp, 0.3_wp, pi)
         else
            mdl = model(nspins, 1.0_wp, 0.4_wp, 0.0_wp, 2, 10, 0.0_wp, 0.0_wp, 0.0_wp, pi)
         end if
         call ground_doublet(nspins, mdl%j, mdl%b, e0, e1, v0, v1)
         call initial_state(st_phys, nspins, v0, v1, .true., psi0(:, 1))
         call initial_state(st_gs_phys, nspins, v0, v1, .true., psi0(:, 2))
         call initial_state(st_gs_cat, nspins, v0, v1, .true., psi0(:, 3))
         call initial_state(st_cat_sum, nspins, v0, v1, .true., psi0(:, 4))
         call qfi_dynamics(mdl, times, psi0, res, diag)
         call build_floquet(mdl, flp, derivative=.false., h=mdl%h + eps)
         call build_floquet(mdl, flm, derivative=.false., h=mdl%h - eps)
         call build_power_table(flp, maxval(times)/int(mdl%nu, ik), ptp)
         call build_power_table(flm, maxval(times)/int(mdl%nu, ik), ptm)
         err = 0.0_wp
         bound = 0.0_wp
         sat = 0.0_wp
         do is = 1, 4
            do it = 1, nt
               call propagate(flp, ptp, times(it), psi0(:, is), kp)
               call propagate(flm, ptm, times(it), psi0(:, is), km)
               psi = 0.5_wp*(kp + km)
               mom = [res(it, is)%mom_sz, res(it, is)%mom_x, res(it, is)%mom_y] &
                     *(real(nspins, wp)*real(times(it), wp))**2
               do k = 1, 3
                  bound = max(bound, mom(k)/res(it, is)%qfi_raw - 1.0_wp)
                  call apply_mom_observable(nspins, k, psi, w)
                  mean = real(dot_product(psi, w), wp)
                  var = real(dot_product(w, w), wp) - mean**2
                  if (var < 1.0e-6_wp) cycle      ! eigenstate: checked below
                  call apply_mom_observable(nspins, k, kp, w)
                  op = real(dot_product(kp, w), wp)
                  call apply_mom_observable(nspins, k, km, w)
                  om = real(dot_product(km, w), wp)
                  mom_fd = ((op - om)/(2.0_wp*eps))**2/var
                  err = max(err, abs(mom(k) - mom_fd)/max(mom_fd, 1.0e-6_wp))
               end do
               ! At h = 0 parity is conserved and the z probe flips it: for the parity
               ! eigenstates GS_cat and CatSum, measuring X saturates F_h.
               if (pass == 2 .and. is >= 3) sat = max(sat, abs(mom(2)/res(it, is)%qfi_raw - 1.0_wp))
            end do
         end do
         if (pass == 1) then
            call check_small(err, 1.0e-15_wp, 'MoM(Sz, X, Y) = central differences (h = 0.02, tilted probe)')
            call check_small(max(bound, 0.0_wp), 1.0e-25_wp, 'MoM <= F_h (h = 0.02)')
         else
            call check_small(err, 1.0e-15_wp, 'MoM(Sz, X, Y) = central differences (h = 0)')
            call check_small(max(bound, 0.0_wp), 1.0e-25_wp, 'MoM <= F_h (h = 0)')
            call check_small(sat, 1.0e-25_wp, 'eigenstate limit: MoM(X) = F_h for parity eigenstates (h = 0)')
         end if
      end do

      ! two-level check of the eigenstate limit: psi(h) = cos h |S> + sin h |-S>, O = X
      ! at h = 0: <X> = sin 2h, MoM -> 4 = F_h.
      psi = czero
      psi(1) = (1.0_wp, 0.0_wp)
      w = czero
      w(n) = (1.0_wp, 0.0_wp)
      call check_small(abs(method_of_moments(nspins, obs_x, psi, w) - 4.0_wp), 1.0e-30_wp, &
                       'two-level example, regular formula')
      psi = (psi + w)/sqrt(2.0_wp)                ! X eigenstate
      w = czero
      w(1) = (1.0_wp, 0.0_wp)/sqrt(2.0_wp)
      w(n) = -(1.0_wp, 0.0_wp)/sqrt(2.0_wp)       ! derivative points to the other parity
      call check_small(abs(method_of_moments(nspins, obs_x, psi, w) - 4.0_wp), 1.0e-30_wp, &
                       'two-level example, eigenstate limit')
   end subroutine test_method_of_moments

   !> Whole pipeline against the definition used by the Python code:
   !> F_h = 4 (<d|d> - |<psi|d>|^2) with d = (psi(h+eps) - psi(h-eps))/(2 eps).
   subroutine test_qfi_against_finite_difference()
      integer, parameter :: nspins = 6, n = nspins + 1, nt = 8
      real(wp), parameter :: eps = 1.0e-12_wp
      type(lmg_model) :: mdl
      type(floquet_ops) :: flp, flm
      type(power_table) :: ptp, ptm
      type(qfi_diagnostics) :: diag
      type(qfi_point) :: res(nt, 4)
      complex(wp) :: psi0(n, 4), v0(n), v1(n), psi(n), kp(n), km(n), d(n)
      integer(ik) :: times(nt)
      real(wp) :: e0, e1, qfd, err, c(nspins), mx, my, mz, merr, h
      integer :: it, is, pass

      write (output_unit, '(a)') 'QFI against central differences in h'
      times = [1_ik, 2_ik, 3_ik, 7_ik, 10_ik, 25_ik, 64_ik, 101_ik]
      call ladder_coefficients(nspins, c)
      do pass = 1, 2
         h = merge(0.02_wp, 0.0_wp, pass == 1)
         if (pass == 1) then
            mdl = model(nspins, 1.0_wp, 0.4_wp, h, 2, 6, 0.4_wp, 0.9_wp, 0.3_wp, pi)
         else
            mdl = model(nspins, 1.0_wp, 0.4_wp, h, 2, 10, 0.0_wp, 0.0_wp, 0.0_wp, pi)
         end if
         call ground_doublet(nspins, mdl%j, mdl%b, e0, e1, v0, v1)
         call initial_state(st_phys, nspins, v0, v1, .true., psi0(:, 1))
         call initial_state(st_gs_phys, nspins, v0, v1, .true., psi0(:, 2))
         call initial_state(st_gs_cat, nspins, v0, v1, .true., psi0(:, 3))
         call initial_state(st_cat_sum, nspins, v0, v1, .true., psi0(:, 4))
         call qfi_dynamics(mdl, times, psi0, res, diag)

         call build_floquet(mdl, flp, derivative=.false., h=h + eps)
         call build_floquet(mdl, flm, derivative=.false., h=h - eps)
         call build_power_table(flp, maxval(times)/int(mdl%nu, ik), ptp)
         call build_power_table(flm, maxval(times)/int(mdl%nu, ik), ptm)
         err = 0.0_wp
         merr = 0.0_wp
         do is = 1, 4
            do it = 1, nt
               call propagate(flp, ptp, times(it), psi0(:, is), kp)
               call propagate(flm, ptm, times(it), psi0(:, is), km)
               d = (kp - km)/(2.0_wp*eps)
               psi = 0.5_wp*(kp + km)      ! psi(h) up to O(eps^2)
               qfd = 4.0_wp*(real(dot_product(d, d), wp) - abs(dot_product(psi, d))**2)
               err = max(err, abs(res(it, is)%qfi_raw - qfd)/max(abs(qfd), 1.0e-6_wp))
               call magnetization(nspins, c, psi, mx, my, mz)
               merr = max(merr, abs(mx - res(it, is)%mx), abs(my - res(it, is)%my), abs(mz - res(it, is)%mz))
            end do
         end do
         if (pass == 1) then
            call check_small(err, 1.0e-15_wp, 'F_h, 4 states x 8 times (h = 0.02, tilted probe)')
            call check_small(merr, 1.0e-18_wp, 'magnetisations (h = 0.02)')
         else
            call check_small(err, 1.0e-15_wp, 'F_h, 4 states x 8 times (h = 0, probe along z)')
            call check_small(merr, 1.0e-18_wp, 'magnetisations (h = 0)')
         end if
         call check_small(diag%max_norm_error, 1.0e-28_wp, 'states stay normalised')
      end do
   end subroutine test_qfi_against_finite_difference

   subroutine test_ground_doublet()
      type(lmg_model) :: mdl
      complex(wp), allocatable :: h0(:, :), v(:, :), v0(:), v1(:), psi(:), hv(:)
      real(wp), allocatable :: w(:), c(:)
      real(wp) :: e0, e1, mx, my, mz, sym
      integer :: nspins, n, i, pass

      write (output_unit, '(a)') 'ground-state doublet and initial states'
      do pass = 1, 2
         nspins = merge(12, 7, pass == 1)
         n = nspins + 1
         if (allocated(h0)) deallocate (h0, v, v0, v1, psi, hv, w, c)
         allocate (h0(n, n), v(n, n), v0(n), v1(n), psi(n), hv(n), w(n), c(nspins))
         mdl = model(nspins, 1.0_wp, 0.4_wp, 0.0_wp, 2, 10, 0.0_wp, 0.0_wp, 0.0_wp, pi)
         call hamiltonian_h0(nspins, mdl%j, mdl%b, h0)
         call herm_eig(h0, w, v)
         call ground_doublet(nspins, mdl%j, mdl%b, e0, e1, v0, v1)
         call check_small(abs(e0 - w(1)) + abs(e1 - w(2)), 1.0e-30_wp, &
                          'two lowest levels agree with the full diagonalisation')
         call matvec(h0, v0, hv)
         call check_small(maxval(abs(hv - e0*v0)), 1.0e-31_wp, 'H0 |g> = E0 |g>')
         call matvec(h0, v1, hv)
         call check_small(maxval(abs(hv - e1*v1)), 1.0e-31_wp, 'H0 |e> = E1 |e>')
         sym = 0.0_wp
         do i = 1, n
            sym = max(sym, abs(v0(i) - v0(n + 1 - i)), abs(v1(i) + v1(n + 1 - i)))
         end do
         call check_small(sym, 0.0_wp, 'ground state even, first excited state odd under m -> -m')
         call check_small(abs(dot_product(v0, v1)) + abs(dot_product(v0, v0) - 1.0_wp), 1.0e-32_wp, &
                          'doublet orthonormal')

         call ladder_coefficients(nspins, c)
         call initial_state(st_gs_phys, nspins, v0, v1, .true., psi)
         call magnetization(nspins, c, psi, mx, my, mz)
         call check(mz > 0.3_wp .and. abs(my) < 1.0e-30_wp, 'GS_phys "up" has <Sz> > 0')
         call initial_state(st_gs_phys, nspins, v0, v1, .false., psi)
         call magnetization(nspins, c, psi, mx, my, mz)
         call check(mz < -0.3_wp, 'GS_phys "down" has <Sz> < 0')
         call initial_state(st_cat_sum, nspins, v0, v1, .true., psi)
         call check_small(abs(psi(1) - sqrt(0.5_wp)) + abs(psi(n) - sqrt(0.5_wp)) &
                          + abs(sum(abs(psi)**2) - 1.0_wp), 1.0e-33_wp, 'CatSum = (|S,S> + |S,-S>)/sqrt(2)')
      end do
      call check(state_from_name('gs_PHYS') == st_gs_phys .and. state_from_name('CatSum') == st_cat_sum &
                 .and. state_from_name('GS_CAT') == st_gs_cat .and. state_from_name('nope') == 0, &
                 'state names are case-insensitive')
      call check(state_enum_name(st_cat_sum) == 'CAT_SUM', 'enum name used in file names')
   end subroutine test_ground_doublet

   !> Values printed by NumPy: [int(x) for x in np.logspace(...)].
   subroutine test_time_grid()
      integer(ik) :: a(50), b(50), d(50)
      integer(ik), allocatable :: grid(:), from_file(:)
      character(len=256) :: message
      integer :: unit, ierr

      write (output_unit, '(a)') 'time grid'
      call logspace_int(2.0_dp, 4.0_dp, 50, .false., a)
      call check(all(a(1:12) == [100_ik, 109_ik, 120_ik, 131_ik, 144_ik, 158_ik, 173_ik, 190_ik, &
                                 208_ik, 229_ik, 251_ik, 275_ik]) .and. a(50) == 9120_ik, &
                 'np.logspace(2, 4, 50, endpoint=False)')
      call logspace_int(4.0_dp, 18.0_dp, 50, .true., b)
      call check(all(b(1:8) == [10000_ik, 19306_ik, 37275_ik, 71968_ik, 138949_ik, 268269_ik, &
                                517947_ik, 1000000_ik]), 'np.logspace(4, 18, 50): first decades')
      call check(all(b(43:50) == [10000000000000000_ik, 19306977288832456_ik, 37275937203149224_ik, &
                                  71968567300114696_ik, 138949549437313600_ik, 268269579527971648_ik, &
                                  517947467923122304_ik, 1000000000000000000_ik]), &
                 'np.logspace(4, 18, 50): times beyond 2**53 are the integer values of the doubles')
      call logspace_int(4.0_dp, 23.0_dp, 50, .true., d)
      call check(d(50) == 99999999999999991611392_ik, 'int(1e23) = 99999999999999991611392')
      call logspace_int(4.0_dp, 4.0_dp, 50, .true., d)
      call check(all(d == 10000_ik), 'degenerate range repeats its end point')

      call python_time_grid(50, 18, grid)
      call check(size(grid) == 199 .and. grid(1) == 1_ik .and. grid(99) == 99_ik .and. grid(100) == 100_ik &
                 .and. grid(150) == 10000_ik .and. grid(199) == 1000000000000000000_ik, &
                 'generate_time_interval(50, 18)')
      call check(max_time_degree(1.31246436297e-5_wp) == 6 .and. max_time_degree(4.2e-11_wp) == 12 &
                 .and. max_time_degree(3.0e-22_wp) == 23, 'last decade from the gap')

      open (newunit=unit, file='test_lmg_times.tmp', status='replace', action='write')
      write (unit, '(a)') 'time,m_x,m_y'
      write (unit, '(a)') '1,0.5,0.25'
      write (unit, '(a)') '  17 , 1.0'
      write (unit, '(a)') ''
      write (unit, '(a)') '99999999999999991611392,0.1,0.2'
      close (unit)
      call read_time_grid('test_lmg_times.tmp', from_file, ierr, message)
      call check(ierr == 0 .and. size(from_file) == 3, 'time grid read from a CSV file')
      if (size(from_file) == 3) then
         call check(all(from_file == [1_ik, 17_ik, 99999999999999991611392_ik]), 'header skipped, 128-bit times kept')
      end if
      open (newunit=unit, file='test_lmg_times.tmp', status='old')
      close (unit, status='delete')
      call read_time_grid('test_lmg_missing.tmp', from_file, ierr, message)
      call check(ierr /= 0, 'missing time grid file is reported')
   end subroutine test_time_grid

   subroutine test_config_and_io()
      type(run_config) :: cfg
      type(qfi_point) :: pts(1)
      character(len=200) :: line
      real(wp) :: x
      integer :: k, ierr, unit
      logical :: ok

      write (output_unit, '(a)') 'configuration and output'
      call parse_real('0.4', x, ok)
      call check(ok .and. x == 0.4_wp, 'decimal literal parsed in quadruple precision')
      call parse_real(' pi ', x, ok)
      call check(ok .and. x == pi, 'pi')
      call parse_real('-0.5*pi', x, ok)
      call check(ok .and. abs(x + 0.5_wp*pi) < 1.0e-33_wp, '-0.5*pi')
      call parse_real('pi/2', x, ok)
      call check(ok .and. abs(x - 0.5_wp*pi) < 1.0e-33_wp, 'pi/2')
      call parse_real('1e-3', x, ok)
      call check(ok .and. abs(x - 1.0e-3_wp) < 1.0e-36_wp, 'exponent notation')
      call parse_real('abc', x, ok)
      call check(.not. ok, 'garbage rejected')
      call parse_real('2*', x, ok)
      call check(.not. ok, 'dangling operator rejected')
      call parse_int('12x', k, ok)
      call check(.not. ok, 'malformed integer rejected')

      call init_config(cfg)
      call apply_setting(cfg, 'steps-floquet-unitary', '250', '', ierr)
      call check(ierr == 0 .and. cfg%model%steps == 250, 'ini key steps-floquet-unitary')
      call apply_setting(cfg, 'phi-kick-phase', 'pi', '', ierr)
      call check(ierr == 0 .and. cfg%model%phi_kick == pi, 'ini key phi-kick-phase = pi')
      call apply_setting(cfg, 'Phi_0', '0.25', '', ierr)
      call check(ierr == 0 .and. cfg%model%phi0 == 0.25_wp, 'keys are case-insensitive, _ = -')
      call apply_setting(cfg, 'output-dir', 'results', '/some/dir', ierr)
      call check(ierr == 0 .and. cfg%output_dir == '/some/dir/results', 'relative paths resolved against the ini file')
      call apply_setting(cfg, 'init-state', 'GS_phys,CatSum', '', ierr)
      call check(ierr == 0 .and. all(cfg%states .eqv. [.false., .true., .false., .true.]), 'init-state list')
      call apply_setting(cfg, 'no-such-key', '1', '', ierr)
      call check(ierr == 1, 'unknown key reported')
      call apply_setting(cfg, 'n', 'ten', '', ierr)
      call check(ierr == 2, 'bad value reported')

      call check(format_time(99999999999999991611392_ik) == '99999999999999991611392', '128-bit time formatting')
      call check(format_fixed2(0.4_wp) == '0.40', 'B label like Python''s :.2f')
      call check(result_file_name('out', 'fortran.', 'GS_PHYS', 50, 0.4_wp) == 'out/fortran.GS_PHYS_N=50_B=0.40.csv', &
                 'result file name')
      call check(format_real(0.25_wp, 17) == '0.25000000000000000', 'real formatting')

      pts(1)%time = 12_ik
      pts(1)%mx = 0.125_wp
      pts(1)%qfi_raw = 3.0_wp
      call write_csv('test_lmg_out.tmp', pts, ierr)
      call check(ierr == 0, 'CSV file written')
      open (newunit=unit, file='test_lmg_out.tmp', status='old', action='read')
      read (unit, '(a)') line
      call check(trim(line) == 'time,m_x,m_y,m_z,qfi,qfi_raw,mom_sz,mom_x,mom_y', 'CSV header')
      read (unit, '(a)') line
      call check(line(1:3) == '12,' .and. index(line, ',0.12500000000000000,') == 3, 'CSV row')
      close (unit, status='delete')
   end subroutine test_config_and_io

end program test_lmg
