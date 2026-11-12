!> Floquet propagators of the kicked LMG model and their exact derivatives
!> with respect to the probe amplitude h.
!>
!> One kicked period p (p = 1..nu) is integrated exactly like the Python
!> reference (`evalution_T_step`): `steps` matrix exponentials of length
!> dt = T/steps, sampled at t_i = T (p-1) + (i-1) T/(steps-1), followed by
!> the kick,
!>
!>   W_p = K * E_steps * ... * E_1,     E_i = exp(-i dt (H0 + h s_i S_alpha)),
!>   s_i = sin(omega t_i + phi0),       K = exp(-i phi Sx).
!>
!> The evolution to the stroboscopic time t = r*nu + m is
!>   U(t) = P_m * U_F**r,   P_m = W_m ... W_1,   U_F = P_nu.
!> Because omega*nu*T = 2 pi, the steps of every later cycle coincide with
!> those of the first one, so only P_1..P_nu are ever needed.
!>
!> Instead of the finite difference in h used by the Python code, the
!> derivative dU(t)/dh is propagated analytically (product rule plus the
!> Daleckii-Krein formula for the derivative of a matrix exponential). This
!> removes the subtraction of two nearly equal states and is what makes
!> quadruple precision sufficient where the finite difference needs ~60 digits.
module lmg_floquet
   use, intrinsic :: iso_fortran_env, only: int64
   use lmg_kinds, only: wp, ik, twopi, czero
   use lmg_linalg, only: matmul_omp, matvec, adjoint, herm_eig
   use lmg_operators, only: lmg_model, hamiltonian_h0, probe_operator, spin_operators
   !$ use omp_lib, only: omp_in_parallel
   implicit none
   private

   public :: floquet_ops, build_floquet
   public :: power_table, build_power_table
   public :: propagate, drive_samples

   !> Evolution operators over the first m kicked periods of a drive cycle.
   type :: floquet_ops
      integer :: n = 0                          !< Hilbert-space dimension N+1
      integer :: nu = 0                         !< periods per drive cycle
      integer :: nexp = 0                       !< distinct Trotter exponentials in U_F
      logical :: has_derivative = .false.
      complex(wp), allocatable :: u(:, :, :)    !< u(:,:,m) = P_m, m = 1..nu; u(:,:,nu) = U_F
      complex(wp), allocatable :: du(:, :, :)   !< du(:,:,m) = dP_m/dh
   end type floquet_ops

   !> Repeated squares of the Floquet operator: a(:,:,k) = U_F**(2**(k-1)).
   type :: power_table
      integer :: n = 0
      integer :: nlev = 0
      logical :: has_derivative = .false.
      complex(wp), allocatable :: a(:, :, :)
      complex(wp), allocatable :: da(:, :, :)   !< da(:,:,k) = d a(:,:,k)/dh
   end type power_table

   !> Eigen-decomposition of the step Hamiltonian H0 + c S_alpha, reused by
   !> consecutive Trotter steps that share the same c = h s_i.
   type :: eig_cache
      logical :: valid = .false.
      logical :: has_b = .false.
      real(wp) :: c = 0.0_wp
      real(wp), allocatable :: lam(:)
      complex(wp), allocatable :: q(:, :), qh(:, :), bt(:, :)
   end type eig_cache

contains

   !> Build P_1..P_nu (and optionally their h-derivatives) for the model.
   !> `h` overrides the probe amplitude stored in the model.
   subroutine build_floquet(mdl, fl, derivative, h)
      type(lmg_model), intent(in)    :: mdl
      type(floquet_ops), intent(out) :: fl
      logical, intent(in)            :: derivative
      real(wp), intent(in), optional :: h

      complex(wp), allocatable :: h0(:, :), sa(:, :), kick(:, :), w(:, :), dw(:, :), t1(:, :), t2(:, :)
      type(eig_cache) :: cache
      real(wp) :: hval
      integer :: n, p, nexp_p

      if (mdl%nspins < 1) error stop 'build_floquet: the number of spins must be positive'
      if (mdl%nu < 1) error stop 'build_floquet: frequency (nu) must be a positive integer'
      if (mdl%steps < 2) error stop 'build_floquet: steps-floquet-unitary must be at least 2'

      hval = mdl%h
      if (present(h)) hval = h
      n = mdl%nspins + 1
      allocate (h0(n, n), sa(n, n), kick(n, n), w(n, n), dw(n, n), t1(n, n), t2(n, n))
      call hamiltonian_h0(mdl%nspins, mdl%j, mdl%b, h0)
      call probe_operator(mdl%nspins, mdl%theta, mdl%varphi, sa)
      call kick_operator(mdl%nspins, mdl%phi_kick, kick)

      fl%n = n
      fl%nu = mdl%nu
      fl%has_derivative = derivative
      allocate (fl%u(n, n, mdl%nu))
      if (derivative) then
         allocate (fl%du(n, n, mdl%nu))
      else
         allocate (fl%du(0, 0, 0))
      end if

      fl%nexp = 0
      do p = 1, mdl%nu
         call kicked_period(mdl, hval, p, h0, sa, kick, derivative, cache, w, dw, nexp_p)
         fl%nexp = fl%nexp + nexp_p
         if (p == 1) then
            fl%u(:, :, 1) = w
            if (derivative) fl%du(:, :, 1) = dw
         else
            call matmul_omp(w, fl%u(:, :, p - 1), fl%u(:, :, p))
            if (derivative) then
               call matmul_omp(dw, fl%u(:, :, p - 1), t1)
               call matmul_omp(w, fl%du(:, :, p - 1), t2)
               fl%du(:, :, p) = t1 + t2
            end if
         end if
      end do
   end subroutine build_floquet

   !> Samples s_i = sin(omega t_i + phi0) of the probe over period p.
   !> omega t_i = 2 pi [(p-1)(steps-1) + (i-1)] / [nu (steps-1)] is reduced
   !> modulo 2 pi in exact integer arithmetic.
   pure subroutine drive_samples(mdl, p, s)
      type(lmg_model), intent(in) :: mdl
      integer, intent(in)         :: p
      real(wp), intent(out)       :: s(:)
      integer(int64) :: den, q
      integer :: i

      den = int(mdl%nu, int64)*int(mdl%steps - 1, int64)
      do i = 1, mdl%steps
         q = modulo(int(p - 1, int64)*int(mdl%steps - 1, int64) + int(i - 1, int64), den)
         s(i) = sin(twopi*real(q, wp)/real(den, wp) + mdl%phi0)
      end do
   end subroutine drive_samples

   !> Kick K = exp(-i phi Sx). The spectrum of Sx is known exactly
   !> (m = -S..S), only the eigenvectors are computed numerically.
   subroutine kick_operator(nspins, phi, kick)
      integer, intent(in)      :: nspins
      real(wp), intent(in)     :: phi
      complex(wp), intent(out) :: kick(:, :)
      complex(wp), allocatable :: sx(:, :), sy(:, :), sz(:, :), q(:, :), tmp(:, :)
      real(wp), allocatable :: lam(:)
      real(wp) :: ang
      integer :: n, a

      n = nspins + 1
      allocate (sx(n, n), sy(n, n), sz(n, n), q(n, n), tmp(n, n), lam(n))
      call spin_operators(nspins, sx, sy, sz)
      call herm_eig(sx, lam, q)
      do a = 1, n
         ang = phi*(real(a - 1, wp) - 0.5_wp*real(nspins, wp))
         tmp(:, a) = q(:, a)*cmplx(cos(ang), -sin(ang), wp)
      end do
      call matmul_omp(tmp, adjoint(q), kick)
   end subroutine kick_operator

   !> W_p and dW_p/dh for one kicked period.
   !>
   !> Consecutive steps with the same h*s_i form a "run" that is evaluated in
   !> closed form in the eigenbasis of its Hamiltonian (for h = 0 the whole
   !> period is a single run). The runs are split into contiguous chunks that
   !> the OpenMP threads multiply independently; the ordered chunk products
   !> are merged at the end.
   !> `cache` carries the last eigen-decomposition from one period to the next
   !> (for h = 0 every period uses the one of H0). `nexp` returns the number
   !> of runs, i.e. of distinct matrix exponentials in W_p.
   subroutine kicked_period(mdl, h, p, h0, sa, kick, derivative, cache, w, dw, nexp)
      type(lmg_model), intent(in)    :: mdl
      real(wp), intent(in)           :: h
      integer, intent(in)            :: p
      complex(wp), intent(in)        :: h0(:, :), sa(:, :), kick(:, :)
      logical, intent(in)            :: derivative
      type(eig_cache), intent(inout) :: cache
      complex(wp), intent(out)       :: w(:, :), dw(:, :)
      integer, intent(out)           :: nexp

      type(eig_cache), allocatable :: chunk_cache(:)

      real(wp), allocatable :: s(:), c(:)
      integer, allocatable :: first(:)
      complex(wp), allocatable :: xc(:, :, :), dxc(:, :, :), x(:, :), dx(:, :), t1(:, :), t2(:, :), t3(:, :)
      real(wp) :: dt
      integer, parameter :: max_chunks = 64
      integer :: n, steps, i, nruns, nchunks, ic, r1, r2

      n = size(h0, 1)
      steps = mdl%steps
      dt = mdl%t/real(steps, wp)
      allocate (s(steps), c(steps), first(steps + 1))
      call drive_samples(mdl, p, s)
      c = h*s

      nruns = 1
      first(1) = 1
      do i = 2, steps
         if (c(i) /= c(i - 1)) then
            nruns = nruns + 1
            first(nruns) = i
         end if
      end do
      first(nruns + 1) = steps + 1
      nexp = nruns

      ! The number of chunks is fixed, not tied to the number of threads, so
      ! that the grouping of the matrix products (and with it every rounding
      ! error) is the same for any thread count.
      nchunks = min(nruns, max_chunks)
      allocate (xc(n, n, nchunks), dxc(n, n, nchunks))

      if (nchunks == 1) then
         ! Not inside a parallel region: the kernels below thread themselves.
         call chunk_product(dt, c, s, first, 1, nruns, h0, sa, derivative, cache, xc(:, :, 1), dxc(:, :, 1))
      else
         allocate (chunk_cache(nchunks))
         !$omp parallel do default(shared) private(ic, r1, r2) schedule(dynamic, 1)
         do ic = 1, nchunks
            r1 = ((ic - 1)*nruns)/nchunks + 1
            r2 = (ic*nruns)/nchunks
            call chunk_product(dt, c, s, first, r1, r2, h0, sa, derivative, chunk_cache(ic), &
                               xc(:, :, ic), dxc(:, :, ic))
         end do
         !$omp end parallel do
      end if

      allocate (x(n, n), dx(n, n), t1(n, n), t2(n, n), t3(n, n))
      x = xc(:, :, 1)
      dx = dxc(:, :, 1)
      do ic = 2, nchunks
         call matmul_omp(xc(:, :, ic), x, t1)
         if (derivative) then
            call matmul_omp(dxc(:, :, ic), x, t2)
            call matmul_omp(xc(:, :, ic), dx, t3)
            dx = t2 + t3
         end if
         x = t1
      end do

      call matmul_omp(kick, x, w)
      if (derivative) then
         call matmul_omp(kick, dx, dw)
      else
         dw = czero
      end if
   end subroutine kicked_period

   !> Ordered product (and derivative) of the runs r1..r2 of one period.
   subroutine chunk_product(dt, c, s, first, r1, r2, h0, sa, derivative, cache, x, dx)
      real(wp), intent(in)           :: dt, c(:), s(:)
      integer, intent(in)            :: first(:), r1, r2
      complex(wp), intent(in)        :: h0(:, :), sa(:, :)
      logical, intent(in)            :: derivative
      type(eig_cache), intent(inout) :: cache
      complex(wp), intent(out)       :: x(:, :), dx(:, :)

      complex(wp), allocatable :: r(:, :), dr(:, :), t1(:, :), t2(:, :), t3(:, :)
      integer :: n, ir, i1, i2

      n = size(h0, 1)
      allocate (r(n, n), dr(n, n), t1(n, n), t2(n, n), t3(n, n))
      dx = czero
      do ir = r1, r2
         i1 = first(ir)
         i2 = first(ir + 1) - 1
         call run_propagator(dt, c(i1), s(i1:i2), h0, sa, derivative, cache, r, dr)
         if (ir == r1) then
            x = r
            if (derivative) dx = dr
         else
            call matmul_omp(r, x, t1)
            if (derivative) then
               call matmul_omp(dr, x, t2)
               call matmul_omp(r, dx, t3)
               dx = t2 + t3
            end if
            x = t1
         end if
      end do
   end subroutine chunk_product

   !> Propagator R = E**k of a run of k identical steps E = exp(-i dt M),
   !> M = H0 + c S_alpha = Q Lambda Q^H, and its derivative
   !>
   !>   dR/dh = sum_j s_j E**(k-j) L E**(j-1),
   !>   L = Q [ (Q^H (-i dt S_alpha) Q) o F ] Q^H        (Daleckii-Krein),
   !>   F_ab = exp(-i dt (l_a + l_b)/2) sinc(dt (l_a - l_b)/2),
   !>
   !> so that in the eigenbasis (dR)_ab = B_ab F_ab G_ab with
   !>   G_ab = sum_j s_j x_a**(k-j) x_b**(j-1),   x_a = exp(-i dt l_a),
   !> evaluated with Horner's rule in z = x_b/x_a. The sinc form of the
   !> divided difference has no cancellation for close eigenvalues.
   subroutine run_propagator(dt, cval, s, h0, sa, derivative, cache, r, dr)
      real(wp), intent(in)           :: dt, cval, s(:)
      complex(wp), intent(in)        :: h0(:, :), sa(:, :)
      logical, intent(in)            :: derivative
      type(eig_cache), intent(inout) :: cache
      complex(wp), intent(out)       :: r(:, :), dr(:, :)

      complex(wp), allocatable :: tmp(:, :), tmp2(:, :), x1(:), xh(:), xk1(:)
      complex(wp) :: z, acc, f
      real(wp) :: ang, half
      integer :: n, k, a, b, jj
      !$ logical :: threaded

      n = size(h0, 1)
      k = size(s)
      allocate (tmp(n, n), tmp2(n, n))

      if (.not. cache%valid .or. cache%c /= cval) then
         if (.not. allocated(cache%lam)) allocate (cache%lam(n), cache%q(n, n), cache%qh(n, n), cache%bt(n, n))
         tmp = h0 + cval*sa
         call herm_eig(tmp, cache%lam, cache%q)
         cache%qh = adjoint(cache%q)
         cache%valid = .true.
         cache%has_b = .false.
         cache%c = cval
      end if

      do a = 1, n
         ang = real(k, wp)*dt*cache%lam(a)
         tmp(:, a) = cache%q(:, a)*cmplx(cos(ang), -sin(ang), wp)
      end do
      call matmul_omp(tmp, cache%qh, r)
      if (.not. derivative) return

      if (.not. cache%has_b) then
         call matmul_omp(cache%qh, sa, tmp)        ! S_alpha is tridiagonal: O(n^2)
         call matmul_omp(tmp, cache%q, cache%bt)
         cache%bt = cmplx(0.0_wp, -dt, wp)*cache%bt
         cache%has_b = .true.
      end if

      allocate (x1(n), xh(n), xk1(n))
      do a = 1, n
         ang = dt*cache%lam(a)
         x1(a) = cmplx(cos(ang), -sin(ang), wp)
         xh(a) = cmplx(cos(0.5_wp*ang), -sin(0.5_wp*ang), wp)
         xk1(a) = cmplx(cos(real(k - 1, wp)*ang), -sin(real(k - 1, wp)*ang), wp)
      end do

      !$ threaded = .not. omp_in_parallel()
      !$ threaded = threaded .and. k > 1
      !$omp parallel do default(none) shared(n, k, dt, s, cache, x1, xh, xk1, tmp) &
      !$omp    private(a, b, jj, half, f, z, acc) schedule(static) if (threaded)
      do b = 1, n
         do a = 1, n
            half = 0.5_wp*dt*(cache%lam(a) - cache%lam(b))
            f = sinc(half)*xh(a)*xh(b)
            if (k == 1) then
               acc = cmplx(s(1), 0.0_wp, wp)
            else
               z = x1(b)*conjg(x1(a))
               acc = cmplx(s(k), 0.0_wp, wp)
               do jj = k - 1, 1, -1
                  acc = acc*z + s(jj)
               end do
               acc = acc*xk1(a)
            end if
            tmp(a, b) = cache%bt(a, b)*f*acc
         end do
      end do
      !$omp end parallel do

      call matmul_omp(cache%q, tmp, tmp2)
      call matmul_omp(tmp2, cache%qh, dr)
   end subroutine run_propagator

   !> sin(x)/x.
   pure function sinc(x) result(y)
      real(wp), intent(in) :: x
      real(wp) :: y

      if (x == 0.0_wp) then
         y = 1.0_wp
      else
         y = sin(x)/x
      end if
   end function sinc

   !> Table of U_F**(2**(k-1)), k = 1..nlev, large enough for exponents up to
   !> rmax, with derivatives when the Floquet data carries them:
   !>   d(A^2)/dh = dA A + A dA.
   subroutine build_power_table(fl, rmax, pt)
      type(floquet_ops), intent(in)  :: fl
      integer(ik), intent(in)        :: rmax
      type(power_table), intent(out) :: pt

      complex(wp), allocatable :: t1(:, :), t2(:, :)
      integer(ik) :: v
      integer :: n, k, nlev

      n = fl%n
      nlev = 0
      v = rmax
      do while (v > 0_ik)
         nlev = nlev + 1
         v = v/2_ik
      end do

      pt%n = n
      pt%nlev = nlev
      pt%has_derivative = fl%has_derivative
      allocate (pt%a(n, n, nlev))
      if (fl%has_derivative) then
         allocate (pt%da(n, n, nlev), t1(n, n), t2(n, n))
      else
         allocate (pt%da(0, 0, 0))
      end if
      if (nlev == 0) return

      pt%a(:, :, 1) = fl%u(:, :, fl%nu)
      if (pt%has_derivative) pt%da(:, :, 1) = fl%du(:, :, fl%nu)
      do k = 2, nlev
         call matmul_omp(pt%a(:, :, k - 1), pt%a(:, :, k - 1), pt%a(:, :, k))
         if (pt%has_derivative) then
            call matmul_omp(pt%da(:, :, k - 1), pt%a(:, :, k - 1), t1)
            call matmul_omp(pt%a(:, :, k - 1), pt%da(:, :, k - 1), t2)
            pt%da(:, :, k) = t1 + t2
         end if
      end do
   end subroutine build_power_table

   !> |psi(t)> = U(t)|psi0> and, when `dpsi` is present, d|psi(t)>/dh.
   !>
   !> U_F**r is applied through the binary expansion of r = t/nu, the
   !> remaining m = mod(t, nu) kicked periods through P_m. Thread safe.
   subroutine propagate(fl, pt, t, psi0, psi, dpsi)
      type(floquet_ops), intent(in)      :: fl
      type(power_table), intent(in)      :: pt
      integer(ik), intent(in)            :: t
      complex(wp), intent(in)            :: psi0(:)
      complex(wp), intent(out)           :: psi(:)
      complex(wp), intent(out), optional :: dpsi(:)

      complex(wp) :: w1(size(psi0)), w2(size(psi0)), w3(size(psi0))
      integer(ik) :: r
      integer :: m, k
      logical :: deriv, dzero

      if (t < 0_ik) error stop 'propagate: negative time'
      deriv = present(dpsi)
      if (deriv .and. .not. (fl%has_derivative .and. pt%has_derivative)) &
         error stop 'propagate: derivative requested but not available'

      r = t/int(fl%nu, ik)
      m = int(mod(t, int(fl%nu, ik)))
      if (shiftr(r, pt%nlev) /= 0_ik) error stop 'propagate: time exceeds the power table'

      psi = psi0
      dzero = .true.
      if (deriv) dpsi = czero

      do k = 1, pt%nlev
         if (.not. btest(r, k - 1)) cycle
         if (deriv) then
            call step_with_derivative(pt%a(:, :, k), pt%da(:, :, k))
         else
            call matvec(pt%a(:, :, k), psi, w1)
            psi = w1
         end if
      end do
      if (m > 0) then
         if (deriv) then
            call step_with_derivative(fl%u(:, :, m), fl%du(:, :, m))
         else
            call matvec(fl%u(:, :, m), psi, w1)
            psi = w1
         end if
      end if

   contains

      !> (psi, dpsi) <- (A psi, dA psi + A dpsi)
      subroutine step_with_derivative(a, da)
         complex(wp), intent(in) :: a(:, :), da(:, :)

         call matvec(a, psi, w1)
         call matvec(da, psi, w2)
         if (.not. dzero) then
            call matvec(a, dpsi, w3)
            w2 = w2 + w3
         end if
         psi = w1
         dpsi = w2
         dzero = .false.
      end subroutine step_with_derivative

   end subroutine propagate

end module lmg_floquet
