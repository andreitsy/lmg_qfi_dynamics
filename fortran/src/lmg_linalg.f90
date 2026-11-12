!> Dense linear-algebra kernels in working (quadruple) precision.
!>
!> LAPACK/BLAS are not available in binary128, so the few kernels the solver
!> needs are implemented here: an OpenMP matrix product, a matrix-vector
!> product and a Jacobi eigen-solver for Hermitian matrices.
module lmg_linalg
   use lmg_kinds, only: wp, czero, cone
   !$ use omp_lib, only: omp_in_parallel
   implicit none
   private

   public :: matmul_omp, matvec, adjoint, identity_matrix
   public :: herm_eig, max_abs, unitarity_defect

contains

   !> C = A*B.
   !>
   !> The loop over the columns of C is shared between the OpenMP threads
   !> unless the caller is already inside a parallel region. C must not alias
   !> A or B. Exact zeros of B are skipped, so products whose right factor is
   !> diagonal or tridiagonal cost O(n^2) instead of O(n^3).
   subroutine matmul_omp(a, b, c)
      complex(wp), intent(in)  :: a(:, :), b(:, :)
      complex(wp), intent(out) :: c(:, :)
      integer :: i, j, k, n, l, m
      complex(wp) :: t
      !$ logical :: threaded

      n = size(a, 1); l = size(a, 2); m = size(b, 2)
      if (size(b, 1) /= l .or. size(c, 1) /= n .or. size(c, 2) /= m) &
         error stop 'matmul_omp: non-conforming shapes'
      !$ threaded = .not. omp_in_parallel()
      !$ threaded = threaded .and. m >= 8
      !$omp parallel do default(none) shared(a, b, c, n, l, m) private(i, j, k, t) &
      !$omp    schedule(dynamic, 1) if (threaded)
      do j = 1, m
         c(:, j) = czero
         do k = 1, l
            t = b(k, j)
            if (t == czero) cycle
            do i = 1, n
               c(i, j) = c(i, j) + a(i, k)*t
            end do
         end do
      end do
      !$omp end parallel do
   end subroutine matmul_omp

   !> y = A*x (y must not alias x). Exact zeros of x are skipped.
   pure subroutine matvec(a, x, y)
      complex(wp), intent(in)  :: a(:, :), x(:)
      complex(wp), intent(out) :: y(:)
      integer :: i, j
      complex(wp) :: t

      y = czero
      do j = 1, size(a, 2)
         t = x(j)
         if (t == czero) cycle
         do i = 1, size(a, 1)
            y(i) = y(i) + a(i, j)*t
         end do
      end do
   end subroutine matvec

   !> Conjugate transpose.
   pure function adjoint(a) result(ah)
      complex(wp), intent(in) :: a(:, :)
      complex(wp) :: ah(size(a, 2), size(a, 1))
      integer :: i, j

      do j = 1, size(a, 2)
         do i = 1, size(a, 1)
            ah(j, i) = conjg(a(i, j))
         end do
      end do
   end function adjoint

   !> n-by-n identity.
   pure function identity_matrix(n) result(e)
      integer, intent(in) :: n
      complex(wp) :: e(n, n)
      integer :: i

      e = czero
      do i = 1, n
         e(i, i) = cone
      end do
   end function identity_matrix

   !> Largest modulus of the entries of a matrix.
   pure function max_abs(a) result(m)
      complex(wp), intent(in) :: a(:, :)
      real(wp) :: m

      m = 0.0_wp
      if (size(a) > 0) m = maxval(abs(a))
   end function max_abs

   !> max |(U^H U - 1)_ij|: how far U is from being unitary.
   function unitarity_defect(u) result(defect)
      complex(wp), intent(in) :: u(:, :)
      real(wp) :: defect
      complex(wp) :: s
      integer :: i, j, n

      n = size(u, 2)
      defect = 0.0_wp
      !$omp parallel do default(none) shared(u, n) private(i, j, s) &
      !$omp    schedule(dynamic, 4) reduction(max: defect)
      do j = 1, n
         do i = 1, j
            s = dot_product(u(:, i), u(:, j))
            if (i == j) s = s - cone
            defect = max(defect, abs(s))
         end do
      end do
      !$omp end parallel do
   end function unitarity_defect

   !> Eigen-decomposition A = V diag(w) V^H of a Hermitian matrix with the
   !> Jacobi method.
   !>
   !> Eigenvalues are returned in ascending order; the columns of V are the
   !> corresponding eigenvectors, orthonormal to working precision even when
   !> eigenvalues are (nearly) degenerate. Only the Hermitian part of the
   !> input is used.
   !>
   !> A sweep visits every index pair once in round-robin ("tournament")
   !> order. Each of its rounds consists of disjoint pairs, whose plane
   !> rotations commute; they are applied by the OpenMP threads in parallel,
   !> first to the columns of A and V, then to the rows of A. The sequence of
   !> rotations, and with it the result, is independent of the thread count.
   subroutine herm_eig(a_in, w, v, sweeps)
      complex(wp), intent(in)  :: a_in(:, :)
      real(wp), intent(out)    :: w(:)
      complex(wp), intent(out) :: v(:, :)
      integer, intent(out), optional :: sweeps

      integer, parameter :: max_sweeps = 200
      complex(wp), allocatable :: a(:, :), col(:), cu(:), su(:)
      real(wp), allocatable :: cc(:), ss(:), dnew_p(:), dnew_q(:)
      integer, allocatable :: perm(:), pp(:), qq(:)
      logical, allocatable :: active(:)
      complex(wp) :: apq, u, x, y
      real(wp) :: g, dpp, dqq, theta, t, c, s, anorm, tol, rot_tol, offmax, wtmp
      integer :: n, m, npairs, p, q, k, i, sweep, round, imin, last
      logical :: converged
      !$ logical :: threaded

      n = size(a_in, 1)
      if (size(a_in, 2) /= n .or. size(w) /= n .or. size(v, 1) /= n .or. size(v, 2) /= n) &
         error stop 'herm_eig: non-conforming shapes'

      m = n + mod(n, 2)            ! an odd dimension is padded with an idle index
      npairs = m/2
      allocate (a(n, n), col(n), cu(npairs), su(npairs), cc(npairs), ss(npairs))
      allocate (dnew_p(npairs), dnew_q(npairs), perm(m), pp(npairs), qq(npairs), active(npairs))
      do q = 1, n
         do p = 1, n
            a(p, q) = 0.5_wp*(a_in(p, q) + conjg(a_in(q, p)))
         end do
      end do
      v = identity_matrix(n)
      do k = 1, m
         perm(k) = k
      end do

      ! Convergence: every off-diagonal element below eps*||A||_F/n. Smaller
      ! elements are not rotated. Jacobi converges quadratically, so the
      ! tight tolerance costs at most one extra sweep.
      anorm = sqrt(sum(real(a, wp)**2 + aimag(a)**2))
      tol = epsilon(1.0_wp)*anorm/real(n, wp)
      rot_tol = tol/16.0_wp
      !$ threaded = .not. omp_in_parallel()
      !$ threaded = threaded .and. n >= 16

      converged = .false.
      do sweep = 1, max_sweeps
         offmax = 0.0_wp
         do q = 2, n
            do p = 1, q - 1
               offmax = max(offmax, abs(a(p, q)))
            end do
         end do
         if (offmax <= tol) then
            converged = .true.
            if (present(sweeps)) sweeps = sweep - 1
            exit
         end if

         do round = 1, m - 1
            ! Rotation R = diag(1, conjg(u)) [[c, s], [-s, c]], u = a_pq/|a_pq|,
            ! annihilates a_pq:  A <- R^H A R,  V <- V R.
            do k = 1, npairs
               p = min(perm(k), perm(m + 1 - k))
               q = max(perm(k), perm(m + 1 - k))
               pp(k) = p
               qq(k) = q
               active(k) = .false.
               if (q > n) cycle
               apq = a(p, q)
               g = abs(apq)
               if (g <= rot_tol) cycle
               dpp = real(a(p, p), wp)
               dqq = real(a(q, q), wp)
               theta = 0.5_wp*(dqq - dpp)/g
               t = 1.0_wp/(abs(theta) + sqrt(1.0_wp + theta*theta))
               if (theta < 0.0_wp) t = -t
               c = 1.0_wp/sqrt(1.0_wp + t*t)
               s = t*c
               u = apq/g
               cc(k) = c
               ss(k) = s
               cu(k) = c*conjg(u)
               su(k) = s*conjg(u)
               dnew_p(k) = dpp - t*g
               dnew_q(k) = dqq + t*g
               active(k) = .true.
            end do

            if (any(active)) then
               !$omp parallel default(none) shared(a, v, n, npairs, pp, qq, active, cc, ss, cu, su) &
               !$omp    private(k, i, p, q, x, y) if (threaded)
               !$omp do schedule(dynamic, 1)
               do k = 1, npairs
                  if (.not. active(k)) cycle
                  p = pp(k)
                  q = qq(k)
                  do i = 1, n
                     x = a(i, p)
                     y = a(i, q)
                     a(i, p) = cc(k)*x - su(k)*y
                     a(i, q) = ss(k)*x + cu(k)*y
                  end do
                  do i = 1, n
                     x = v(i, p)
                     y = v(i, q)
                     v(i, p) = cc(k)*x - su(k)*y
                     v(i, q) = ss(k)*x + cu(k)*y
                  end do
               end do
               !$omp end do
               !$omp do schedule(dynamic, 1)
               do k = 1, npairs
                  if (.not. active(k)) cycle
                  p = pp(k)
                  q = qq(k)
                  do i = 1, n
                     x = a(p, i)
                     y = a(q, i)
                     a(p, i) = cc(k)*x - conjg(su(k))*y
                     a(q, i) = ss(k)*x + conjg(cu(k))*y
                  end do
               end do
               !$omp end do
               !$omp end parallel
               do k = 1, npairs
                  if (.not. active(k)) cycle
                  a(pp(k), pp(k)) = cmplx(dnew_p(k), 0.0_wp, wp)
                  a(qq(k), qq(k)) = cmplx(dnew_q(k), 0.0_wp, wp)
                  a(pp(k), qq(k)) = czero
                  a(qq(k), pp(k)) = czero
               end do
            end if

            ! Next round: keep the first entry, rotate the others by one place.
            last = perm(m)
            do k = m, 3, -1
               perm(k) = perm(k - 1)
            end do
            perm(2) = last
         end do
      end do
      if (.not. converged) error stop 'herm_eig: Jacobi iteration did not converge'

      do p = 1, n
         w(p) = real(a(p, p), wp)
      end do
      do p = 1, n - 1
         imin = p
         do q = p + 1, n
            if (w(q) < w(imin)) imin = q
         end do
         if (imin /= p) then
            wtmp = w(p); w(p) = w(imin); w(imin) = wtmp
            col = v(:, p); v(:, p) = v(:, imin); v(:, imin) = col
         end if
      end do
   end subroutine herm_eig

end module lmg_linalg
