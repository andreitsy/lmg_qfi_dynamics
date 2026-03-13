module lmg_operators
  use lmg_constants, only: dp, CI, CZERO, CONE
  use lmg_linalg, only: expm_hermitian
  implicit none
  private
  public :: create_z_operator, create_spin_minus_operator, create_spin_plus_operator
  public :: create_spin_xyz_operators, create_hamiltonian_h0
  public :: create_kick_operator, ac_time, create_v_operator

contains

  !> Sz diagonal operator: diag(n/2, n/2-1, ..., -n/2)
  subroutine create_z_operator(n, Sz)
    integer, intent(in) :: n
    complex(dp), intent(out) :: Sz(n+1, n+1)
    real(dp) :: half_n
    integer :: i

    half_n = real(n, dp) / 2.0_dp
    Sz = CZERO
    do i = 1, n+1
      Sz(i,i) = cmplx(half_n - real(i-1, dp), 0.0_dp, dp)
    end do
  end subroutine create_z_operator

  !> Spin lowering operator S-
  subroutine create_spin_minus_operator(n, Sminus)
    integer, intent(in) :: n
    complex(dp), intent(out) :: Sminus(n+1, n+1)
    real(dp) :: s, m
    integer :: i

    s = real(n, dp) / 2.0_dp
    Sminus = CZERO
    do i = 1, n
      m = s - real(i-1, dp)  ! m values: s, s-1, ..., -s+1
      Sminus(i+1, i) = cmplx(sqrt(s*(s+1.0_dp) - m*(m-1.0_dp)), 0.0_dp, dp)
    end do
  end subroutine create_spin_minus_operator

  !> Spin raising operator S+ = (S-)^dagger
  subroutine create_spin_plus_operator(n, Splus)
    integer, intent(in) :: n
    complex(dp), intent(out) :: Splus(n+1, n+1)
    complex(dp) :: Sminus(n+1, n+1)

    call create_spin_minus_operator(n, Sminus)
    Splus = conjg(transpose(Sminus))
  end subroutine create_spin_plus_operator

  !> Create all three spin operators. Returns (Sz, Sx, Sy) -- Sz FIRST.
  subroutine create_spin_xyz_operators(n, Sz, Sx, Sy)
    integer, intent(in) :: n
    complex(dp), intent(out) :: Sz(n+1, n+1), Sx(n+1, n+1), Sy(n+1, n+1)
    complex(dp) :: Splus(n+1, n+1), Sminus(n+1, n+1)

    call create_z_operator(n, Sz)
    call create_spin_plus_operator(n, Splus)
    call create_spin_minus_operator(n, Sminus)
    Sx = (Splus + Sminus) / 2.0_dp
    Sy = -CI * (Splus - Sminus) / 2.0_dp
  end subroutine create_spin_xyz_operators

  !> Hamiltonian H0 = -(2J/N) Sz^2 - 2B Sx
  subroutine create_hamiltonian_h0(J, B, n, H0)
    real(dp), intent(in) :: J, B
    integer, intent(in) :: n
    complex(dp), intent(out) :: H0(n+1, n+1)
    complex(dp) :: Sz(n+1, n+1), Sx(n+1, n+1), Sy(n+1, n+1)
    complex(dp) :: Sz2(n+1, n+1)

    call create_spin_xyz_operators(n, Sz, Sx, Sy)
    Sz2 = matmul(Sz, Sz)
    H0 = -J * (2.0_dp / real(n, dp)) * Sz2 - 2.0_dp * B * Sx
  end subroutine create_hamiltonian_h0

  !> Kick operator: exp(-i * phi * Sx)
  subroutine create_kick_operator(phi, Sx, n, kick)
    real(dp), intent(in) :: phi
    complex(dp), intent(in) :: Sx(n+1, n+1)
    integer, intent(in) :: n
    complex(dp), intent(out) :: kick(n+1, n+1)
    complex(dp) :: H_real(n+1, n+1)

    ! Sx is Hermitian, so expm_hermitian computes exp(-i * phi * Sx)
    H_real = Sx
    call expm_hermitian(H_real, n+1, phi, kick)
  end subroutine create_kick_operator

  !> AC field: sin(omega*t + phi_0) * [sin(theta)*cos(varphi)*Sx + sin(theta)*sin(varphi)*Sy + cos(theta)*Sz]
  subroutine ac_time(Sx, Sy, Sz, n, omega, phi_0, t_k, theta, varphi, result)
    complex(dp), intent(in) :: Sx(n+1,n+1), Sy(n+1,n+1), Sz(n+1,n+1)
    integer, intent(in) :: n
    real(dp), intent(in) :: omega, phi_0, t_k, theta, varphi
    complex(dp), intent(out) :: result(n+1,n+1)
    real(dp) :: sinusoidal

    sinusoidal = sin(t_k * omega + phi_0)
    result = sinusoidal * ( &
      sin(theta) * cos(varphi) * Sx + &
      sin(theta) * sin(varphi) * Sy + &
      cos(theta) * Sz )
  end subroutine ac_time

  !> V(t) = H0 + h * S_alpha(t)
  subroutine create_v_operator(H0, Sx, Sy, Sz, n, omega, phi_0, h, t_k, theta, varphi, V)
    complex(dp), intent(in) :: H0(n+1,n+1), Sx(n+1,n+1), Sy(n+1,n+1), Sz(n+1,n+1)
    integer, intent(in) :: n
    real(dp), intent(in) :: omega, phi_0, h, t_k, theta, varphi
    complex(dp), intent(out) :: V(n+1,n+1)
    complex(dp) :: S_alpha(n+1,n+1)

    call ac_time(Sx, Sy, Sz, n, omega, phi_0, t_k, theta, varphi, S_alpha)
    V = H0 + h * S_alpha
  end subroutine create_v_operator

end module lmg_operators
