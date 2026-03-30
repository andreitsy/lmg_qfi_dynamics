module lmg_linalg
  !> Pure Fortran linear algebra at arbitrary precision.
  !> No LAPACK dependency — works with any real kind (double, quad, etc.).
  !>
  !> Implementations:
  !>   eigh           — Jacobi eigenvalue algorithm for Hermitian matrices
  !>   eig            — QR algorithm (Hessenberg + shifted QR + Schur eigenvectors)
  !>   matrix_inverse — Gauss-Jordan elimination with partial pivoting
  !>   expm_hermitian — exp(-i*alpha*H) via Hermitian eigendecomposition
  use lmg_constants, only: dp, CI, CZERO, CONE
  implicit none
  private
  public :: eigh, eig, matrix_inverse, expm_hermitian, adjoint, eye_complex, matmul_alloc

contains

  ! ================================================================
  !  HERMITIAN EIGENDECOMPOSITION — Jacobi method
  ! ================================================================

  !> Hermitian eigendecomposition via the classical Jacobi algorithm.
  !> Returns eigenvalues in ascending order, eigenvectors as columns.
  subroutine eigh(A, n, eigenvalues, eigenvectors)
    integer, intent(in) :: n
    complex(dp), intent(in) :: A(n,n)
    real(dp), intent(out) :: eigenvalues(n)
    complex(dp), intent(out) :: eigenvectors(n,n)

    complex(dp) :: W(n,n)
    real(dp) :: off_norm, diag_norm, tol
    integer :: max_sweeps, sweep, p, q

    W = A
    call eye_complex(n, eigenvectors)

    max_sweeps = 100
    tol = epsilon(1.0_dp)

    do sweep = 1, max_sweeps
      ! Compute off-diagonal Frobenius norm
      call compute_offdiag_norm(W, n, off_norm)
      call compute_diag_norm(W, n, diag_norm)
      if (off_norm <= tol * max(diag_norm, 1.0_dp)) exit

      ! Sweep through all upper-triangular pairs
      do p = 1, n - 1
        do q = p + 1, n
          if (abs(W(p,q)) > tol * max(abs(W(p,p)), abs(W(q,q)), 1.0_dp) * 1e-2_dp) then
            call jacobi_rotate(W, eigenvectors, n, p, q)
          end if
        end do
      end do
    end do

    ! Extract eigenvalues
    call extract_and_sort_eigh(W, eigenvectors, n, eigenvalues)
  end subroutine eigh

  !> Apply one Jacobi rotation to zero W(p,q) in a Hermitian matrix.
  !> Uses the standard two-step approach: phase removal + real Givens rotation.
  !> G = D * R where D = diag(..., e^{-imu}, ...) at position q removes the phase,
  !> and R = [c, s; -s, c] is the standard real Givens rotation.
  subroutine jacobi_rotate(W, V, n, p, q)
    complex(dp), intent(inout) :: W(n,n), V(n,n)
    integer, intent(in) :: n, p, q

    complex(dp) :: Wpq, phase_conj, tmp_p, tmp_q
    real(dp) :: app, aqq, beta_abs, tau, t, c, s
    integer :: k

    Wpq = W(p,q)
    beta_abs = abs(Wpq)
    if (beta_abs < tiny(1.0_dp)) return

    ! Phase: Wpq = beta_abs * exp(i*mu), phase_conj = exp(-i*mu)
    phase_conj = conjg(Wpq) / beta_abs

    app = real(W(p,p), dp)
    aqq = real(W(q,q), dp)

    ! Compute rotation angle for real Givens after phase removal
    if (abs(app - aqq) < tiny(1.0_dp)) then
      t = 1.0_dp
    else
      tau = (aqq - app) / (2.0_dp * beta_abs)
      ! Choose smaller root for numerical stability
      if (tau >= 0.0_dp) then
        t = 1.0_dp / (tau + sqrt(1.0_dp + tau**2))
      else
        t = -1.0_dp / (-tau + sqrt(1.0_dp + tau**2))
      end if
    end if

    c = 1.0_dp / sqrt(1.0_dp + t**2)
    s = t * c

    ! Update diagonal elements
    W(p,p) = cmplx(app - t * beta_abs, 0.0_dp, dp)
    W(q,q) = cmplx(aqq + t * beta_abs, 0.0_dp, dp)
    W(p,q) = CZERO
    W(q,p) = CZERO

    ! Update off-diagonal elements (rows/columns p and q)
    ! Combined transform G = D*R gives:
    !   W_new(k,p) = c * W(k,p) - s * conj(phase) * W(k,q)
    !   W_new(k,q) = s * W(k,p) + c * conj(phase) * W(k,q)
    do k = 1, n
      if (k == p .or. k == q) cycle

      tmp_p = W(k,p)
      tmp_q = W(k,q)

      W(k,p) = c * tmp_p - s * phase_conj * tmp_q
      W(k,q) = s * tmp_p + c * phase_conj * tmp_q

      ! Hermitian: W(p,k) = conj(W(k,p)), W(q,k) = conj(W(k,q))
      W(p,k) = conjg(W(k,p))
      W(q,k) = conjg(W(k,q))
    end do

    ! Update eigenvector matrix: V <- V * G = V * D * R
    do k = 1, n
      tmp_p = V(k,p)
      tmp_q = V(k,q)

      V(k,p) = c * tmp_p - s * phase_conj * tmp_q
      V(k,q) = s * tmp_p + c * phase_conj * tmp_q
    end do
  end subroutine jacobi_rotate

  subroutine compute_offdiag_norm(A, n, norm)
    complex(dp), intent(in) :: A(n,n)
    integer, intent(in) :: n
    real(dp), intent(out) :: norm
    integer :: i, j
    ! For Hermitian A, |A(i,j)|^2 = |A(j,i)|^2, so sum upper triangle and double
    norm = 0.0_dp
    do j = 2, n
      do i = 1, j - 1
        norm = norm + real(A(i,j) * conjg(A(i,j)), dp)
      end do
    end do
    norm = sqrt(2.0_dp * norm)
  end subroutine compute_offdiag_norm

  subroutine compute_diag_norm(A, n, norm)
    complex(dp), intent(in) :: A(n,n)
    integer, intent(in) :: n
    real(dp), intent(out) :: norm
    integer :: i
    norm = 0.0_dp
    do i = 1, n
      norm = norm + real(A(i,i), dp)**2
    end do
    norm = sqrt(norm)
  end subroutine compute_diag_norm

  !> Extract eigenvalues from diagonalized matrix and sort ascending.
  subroutine extract_and_sort_eigh(W, V, n, eigenvalues)
    complex(dp), intent(in) :: W(n,n)
    complex(dp), intent(inout) :: V(n,n)
    integer, intent(in) :: n
    real(dp), intent(out) :: eigenvalues(n)

    integer :: i, j, min_idx
    real(dp) :: tmp_val
    complex(dp) :: tmp_vec(n)

    do i = 1, n
      eigenvalues(i) = real(W(i,i), dp)
    end do

    ! Selection sort
    do i = 1, n - 1
      min_idx = i
      do j = i + 1, n
        if (eigenvalues(j) < eigenvalues(min_idx)) min_idx = j
      end do
      if (min_idx /= i) then
        tmp_val = eigenvalues(i)
        eigenvalues(i) = eigenvalues(min_idx)
        eigenvalues(min_idx) = tmp_val
        tmp_vec = V(:, i)
        V(:, i) = V(:, min_idx)
        V(:, min_idx) = tmp_vec
      end if
    end do
  end subroutine extract_and_sort_eigh

  ! ================================================================
  !  GENERAL EIGENDECOMPOSITION — QR algorithm
  !  1. Reduce to upper Hessenberg via Householder reflectors
  !  2. Single-shift QR iteration to upper triangular (Schur form)
  !  3. Extract eigenvectors from Schur form by back-substitution
  ! ================================================================

  subroutine eig(A, n, eigenvalues, eigenvectors)
    integer, intent(in) :: n
    complex(dp), intent(in) :: A(n,n)
    complex(dp), intent(out) :: eigenvalues(n)
    complex(dp), intent(out) :: eigenvectors(n,n)

    complex(dp) :: H(n,n), Q_hess(n,n), Q_schur(n,n), Q_total(n,n)
    complex(dp) :: Y(n,n)
    integer :: i

    if (n == 1) then
      eigenvalues(1) = A(1,1)
      eigenvectors(1,1) = CONE
      return
    end if

    H = A

    ! Step 1: Reduce to upper Hessenberg form
    call hessenberg_reduce(H, n, Q_hess)

    ! Step 2: QR iteration to Schur form (upper triangular)
    call qr_iteration_schur(H, n, Q_schur)

    ! Eigenvalues are on the diagonal of H (now in Schur form)
    do i = 1, n
      eigenvalues(i) = H(i,i)
    end do

    ! Step 3: Compute eigenvectors from upper triangular Schur form
    call triangular_eigenvectors(H, n, Y)

    ! Transform back: eigenvectors = Q_hess * Q_schur * Y
    Q_total = matmul(Q_hess, Q_schur)
    eigenvectors = matmul(Q_total, Y)

    ! Normalize eigenvectors
    call normalize_columns(eigenvectors, n)
  end subroutine eig

  !> Reduce A to upper Hessenberg form via Householder reflectors.
  !> On exit, H is upper Hessenberg and Q is the accumulated unitary transform.
  subroutine hessenberg_reduce(H, n, Q)
    complex(dp), intent(inout) :: H(n,n)
    integer, intent(in) :: n
    complex(dp), intent(out) :: Q(n,n)

    complex(dp) :: v(n), alpha, beta_val
    real(dp) :: x_norm
    integer :: k, i

    call eye_complex(n, Q)

    do k = 1, n - 2
      ! Compute Householder vector to zero H(k+2:n, k)
      x_norm = 0.0_dp
      do i = k + 1, n
        x_norm = x_norm + real(H(i,k) * conjg(H(i,k)), dp)
      end do
      x_norm = sqrt(x_norm)

      if (x_norm < tiny(1.0_dp)) cycle

      ! Choose alpha to avoid cancellation
      if (abs(H(k+1,k)) > 0.0_dp) then
        alpha = -H(k+1,k) / abs(H(k+1,k)) * x_norm
      else
        alpha = cmplx(x_norm, 0.0_dp, dp)
      end if

      ! Householder vector v = x - alpha * e1
      v = CZERO
      do i = k + 1, n
        v(i) = H(i, k)
      end do
      v(k+1) = v(k+1) - alpha

      ! Normalize v
      beta_val = CZERO
      do i = k + 1, n
        beta_val = beta_val + v(i) * conjg(v(i))
      end do
      if (abs(beta_val) < tiny(1.0_dp)) cycle
      v(k+1:n) = v(k+1:n) / sqrt(real(beta_val, dp))

      ! Apply from left:  H <- (I - 2*v*v^H) * H
      call apply_householder_left(v, n, k+1, n, H, n)

      ! Apply from right: H <- H * (I - 2*v*v^H)
      call apply_householder_right(v, n, 1, n, H, n)

      ! Accumulate Q: Q <- Q * (I - 2*v*v^H)
      call apply_householder_right(v, n, 1, n, Q, n)
    end do
  end subroutine hessenberg_reduce

  !> Apply Householder reflector from the left: A(r1:r2, :) <- (I - 2*v*v^H) * A(r1:r2, :)
  !> v is nonzero only in indices r1:r2
  subroutine apply_householder_left(v, nv, r1, r2, A, n)
    complex(dp), intent(in) :: v(nv)
    integer, intent(in) :: nv, r1, r2, n
    complex(dp), intent(inout) :: A(n, n)

    complex(dp) :: w(n)
    integer :: j, i

    ! w = v^H * A(r1:r2, :) — a row vector
    do j = 1, n
      w(j) = CZERO
      do i = r1, r2
        w(j) = w(j) + conjg(v(i)) * A(i, j)
      end do
    end do

    ! A(r1:r2, :) <- A(r1:r2, :) - 2 * v * w^T
    do j = 1, n
      do i = r1, r2
        A(i, j) = A(i, j) - 2.0_dp * v(i) * w(j)
      end do
    end do
  end subroutine apply_householder_left

  !> Apply Householder reflector from the right: A(:, r1:r2) <- A(:, r1:r2) * (I - 2*v*v^H)
  subroutine apply_householder_right(v, nv, r1, r2, A, n)
    complex(dp), intent(in) :: v(nv)
    integer, intent(in) :: nv, r1, r2, n
    complex(dp), intent(inout) :: A(n, n)

    complex(dp) :: w(n)
    integer :: i, j

    ! w = A(r1:r2, :) * v — a column vector
    do i = r1, r2
      w(i) = CZERO
      do j = 1, n
        w(i) = w(i) + A(i, j) * v(j)
      end do
    end do

    ! A(r1:r2, :) <- A(r1:r2, :) - 2 * w * v^H
    do j = 1, n
      do i = r1, r2
        A(i, j) = A(i, j) - 2.0_dp * w(i) * conjg(v(j))
      end do
    end do
  end subroutine apply_householder_right

  !> QR iteration with single shift on upper Hessenberg matrix.
  !> On exit, H is upper triangular (Schur form) and Q_schur accumulates the transforms.
  subroutine qr_iteration_schur(H, n, Q_schur)
    complex(dp), intent(inout) :: H(n,n)
    integer, intent(in) :: n
    complex(dp), intent(out) :: Q_schur(n,n)

    complex(dp) :: shift
    real(dp) :: tol, sub_norm
    integer :: iter, max_iter, m

    call eye_complex(n, Q_schur)
    max_iter = 200 * n
    tol = epsilon(1.0_dp)

    m = n  ! active submatrix size

    do iter = 1, max_iter
      if (m <= 1) exit

      ! Check for deflation: is H(m, m-1) small enough?
      sub_norm = abs(H(m, m-1))
      if (sub_norm <= tol * (abs(H(m-1, m-1)) + abs(H(m, m)))) then
        H(m, m-1) = CZERO
        m = m - 1
        cycle
      end if

      ! Also check for double deflation
      if (m >= 3) then
        sub_norm = abs(H(m-1, m-2))
        if (sub_norm <= tol * (abs(H(m-2, m-2)) + abs(H(m-1, m-1)))) then
          H(m-1, m-2) = CZERO
        end if
      end if

      ! Wilkinson shift: eigenvalue of bottom-right 2x2 closer to H(m,m)
      shift = wilkinson_shift(H(m-1,m-1), H(m-1,m), H(m,m-1), H(m,m))

      ! Apply shifted QR step via Givens rotations
      call qr_step_givens(H, Q_schur, n, m, shift)
    end do
  end subroutine qr_iteration_schur

  !> Wilkinson shift: eigenvalue of 2x2 [a,b;c,d] closer to d.
  function wilkinson_shift(a, b, c, d) result(shift)
    complex(dp), intent(in) :: a, b, c, d
    complex(dp) :: shift
    complex(dp) :: trace, det, disc, e1, e2

    trace = a + d
    det = a * d - b * c
    disc = sqrt(trace**2 - 4.0_dp * det)

    e1 = (trace + disc) / 2.0_dp
    e2 = (trace - disc) / 2.0_dp

    ! Pick the eigenvalue closer to d
    if (abs(e1 - d) < abs(e2 - d)) then
      shift = e1
    else
      shift = e2
    end if
  end function wilkinson_shift

  !> One QR step with shift on H(1:m, 1:m) using Givens rotations.
  !> Two-phase approach: first all left rotations (QR), then all right rotations (RQ).
  subroutine qr_step_givens(H, Q, n, m, shift)
    complex(dp), intent(inout) :: H(n,n), Q(n,n)
    integer, intent(in) :: n, m
    complex(dp), intent(in) :: shift

    complex(dp) :: tmp, x, y
    complex(dp) :: cs(m-1), ss(m-1)   ! saved Givens parameters
    integer :: i, j

    ! Shift
    do i = 1, m
      H(i,i) = H(i,i) - shift
    end do

    ! Phase 1: QR factorization — apply all G_i^H from the left
    do i = 1, m - 1
      x = H(i, i)
      y = H(i+1, i)
      call givens_rotation(x, y, cs(i), ss(i))

      ! Apply G_i^H from left to rows i, i+1
      do j = i, n
        tmp = conjg(cs(i)) * H(i,j) + conjg(ss(i)) * H(i+1,j)
        H(i+1,j) = -ss(i) * H(i,j) + cs(i) * H(i+1,j)
        H(i,j) = tmp
      end do
    end do

    ! Phase 2: RQ product — apply all G_i from the right
    do i = 1, m - 1
      ! Apply G_i from right to columns i, i+1
      do j = 1, m
        tmp = cs(i) * H(j,i) + ss(i) * H(j,i+1)
        H(j,i+1) = -conjg(ss(i)) * H(j,i) + conjg(cs(i)) * H(j,i+1)
        H(j,i) = tmp
      end do

      ! Accumulate in Q: Q <- Q * G_i
      do j = 1, n
        tmp = cs(i) * Q(j,i) + ss(i) * Q(j,i+1)
        Q(j,i+1) = -conjg(ss(i)) * Q(j,i) + conjg(cs(i)) * Q(j,i+1)
        Q(j,i) = tmp
      end do
    end do

    ! Un-shift
    do i = 1, m
      H(i,i) = H(i,i) + shift
    end do
  end subroutine qr_step_givens

  !> Compute Givens rotation to zero y: G^H * [x; y] = [r; 0]
  !> Returns c, s such that conjg(c)*x + conjg(s)*y = r, -s*x + c*y = 0
  subroutine givens_rotation(x, y, c_val, s_val)
    complex(dp), intent(in) :: x, y
    complex(dp), intent(out) :: c_val, s_val
    real(dp) :: norm_val

    if (abs(y) < tiny(1.0_dp)) then
      c_val = CONE
      s_val = CZERO
      return
    end if

    if (abs(x) < tiny(1.0_dp)) then
      c_val = CZERO
      s_val = y / abs(y)
      return
    end if

    norm_val = sqrt(real(x * conjg(x) + y * conjg(y), dp))
    c_val = x / norm_val
    s_val = y / norm_val
  end subroutine givens_rotation

  !> Compute eigenvectors from upper triangular matrix T.
  !> Solves (T - lambda_k * I) * y = 0 by back-substitution for each eigenvalue.
  subroutine triangular_eigenvectors(T, n, Y)
    complex(dp), intent(in) :: T(n,n)
    integer, intent(in) :: n
    complex(dp), intent(out) :: Y(n,n)

    complex(dp) :: lambda_k, diag_diff, s
    real(dp) :: safe_min
    integer :: k, i, j

    safe_min = epsilon(1.0_dp)
    Y = CZERO

    do k = 1, n
      lambda_k = T(k,k)
      Y(k,k) = CONE

      ! Back-substitute: for i = k-1 down to 1
      do i = k - 1, 1, -1
        s = CZERO
        do j = i + 1, k
          s = s + T(i,j) * Y(j,k)
        end do

        diag_diff = T(i,i) - lambda_k
        ! Guard against division by zero (repeated/clustered eigenvalues)
        if (abs(diag_diff) < safe_min) then
          diag_diff = cmplx(safe_min, 0.0_dp, dp)
        end if

        Y(i,k) = -s / diag_diff
      end do
    end do
  end subroutine triangular_eigenvectors

  !> Normalize columns of a matrix to unit length.
  subroutine normalize_columns(V, n)
    complex(dp), intent(inout) :: V(n,n)
    integer, intent(in) :: n
    real(dp) :: col_norm
    integer :: j, i

    do j = 1, n
      col_norm = 0.0_dp
      do i = 1, n
        col_norm = col_norm + real(V(i,j) * conjg(V(i,j)), dp)
      end do
      col_norm = sqrt(col_norm)
      if (col_norm > tiny(1.0_dp)) then
        V(:,j) = V(:,j) / col_norm
      end if
    end do
  end subroutine normalize_columns

  ! ================================================================
  !  MATRIX INVERSE — Gauss-Jordan elimination
  ! ================================================================

  !> Matrix inverse via Gauss-Jordan elimination with partial pivoting.
  subroutine matrix_inverse(A, n, A_inv)
    integer, intent(in) :: n
    complex(dp), intent(in) :: A(n,n)
    complex(dp), intent(out) :: A_inv(n,n)

    complex(dp) :: augmented(n, 2*n)
    complex(dp) :: pivot, factor, tmp
    real(dp) :: max_val
    integer :: i, j, k, max_row

    ! Build augmented matrix [A | I]
    augmented(:, 1:n) = A
    augmented(:, n+1:2*n) = CZERO
    do i = 1, n
      augmented(i, n + i) = CONE
    end do

    ! Forward elimination with partial pivoting
    do k = 1, n
      ! Find pivot (max magnitude in column k, rows k:n)
      max_val = 0.0_dp
      max_row = k
      do i = k, n
        if (abs(augmented(i, k)) > max_val) then
          max_val = abs(augmented(i, k))
          max_row = i
        end if
      end do

      if (max_val < tiny(1.0_dp)) then
        write(*,*) 'ERROR: Singular matrix in matrix_inverse'
        stop 1
      end if

      ! Swap rows
      if (max_row /= k) then
        do j = 1, 2 * n
          tmp = augmented(k, j)
          augmented(k, j) = augmented(max_row, j)
          augmented(max_row, j) = tmp
        end do
      end if

      ! Scale pivot row
      pivot = augmented(k, k)
      do j = 1, 2 * n
        augmented(k, j) = augmented(k, j) / pivot
      end do

      ! Eliminate column k in all other rows
      do i = 1, n
        if (i == k) cycle
        factor = augmented(i, k)
        do j = 1, 2 * n
          augmented(i, j) = augmented(i, j) - factor * augmented(k, j)
        end do
      end do
    end do

    ! Extract inverse from right half
    A_inv = augmented(:, n+1:2*n)
  end subroutine matrix_inverse

  ! ================================================================
  !  MATRIX EXPONENTIAL for Hermitian matrices
  ! ================================================================

  !> Matrix exponential of Hermitian matrix: result = exp(-i * alpha * H)
  !> Uses eigendecomposition: exp(-i*alpha*H) = V * diag(exp(-i*alpha*lambda)) * V^dag
  subroutine expm_hermitian(H, n, alpha, result)
    integer, intent(in) :: n
    complex(dp), intent(in) :: H(n,n)
    real(dp), intent(in) :: alpha
    complex(dp), intent(out) :: result(n,n)

    real(dp) :: eigenvalues(n)
    complex(dp) :: eigvecs(n,n), eigvecs_dag(n,n)
    complex(dp) :: scaled(n,n)
    integer :: i

    call eigh(H, n, eigenvalues, eigvecs)

    ! Scale columns of eigvecs by exp(-i * alpha * lambda_i)
    ! This replaces matmul(eigvecs, diag_exp) without forming the diagonal matrix
    do i = 1, n
      scaled(:, i) = eigvecs(:, i) * exp(-CI * alpha * eigenvalues(i))
    end do

    call adjoint(eigvecs, n, eigvecs_dag)
    result = matmul(scaled, eigvecs_dag)
  end subroutine expm_hermitian

  ! ================================================================
  !  UTILITY ROUTINES
  ! ================================================================

  !> Conjugate transpose (adjoint)
  subroutine adjoint(A, n, A_dag)
    integer, intent(in) :: n
    complex(dp), intent(in) :: A(n,n)
    complex(dp), intent(out) :: A_dag(n,n)

    A_dag = conjg(transpose(A))
  end subroutine adjoint

  !> Complex identity matrix
  subroutine eye_complex(n, I_mat)
    integer, intent(in) :: n
    complex(dp), intent(out) :: I_mat(n,n)
    integer :: i

    I_mat = CZERO
    do i = 1, n
      I_mat(i,i) = CONE
    end do
  end subroutine eye_complex

  !> Allocating matrix multiply helper (returns C = A * B)
  function matmul_alloc(A, B, n) result(C)
    integer, intent(in) :: n
    complex(dp), intent(in) :: A(n,n), B(n,n)
    complex(dp) :: C(n,n)

    C = matmul(A, B)
  end function matmul_alloc

end module lmg_linalg
