module lmg_constants
  implicit none
  ! Quad precision: ~33 significant digits
  integer, parameter :: dp = selected_real_kind(33, 4931)
  real(dp), parameter :: PI = 4.0_dp * atan(1.0_dp)
  complex(dp), parameter :: CI = (0.0_dp, 1.0_dp)
  complex(dp), parameter :: CZERO = (0.0_dp, 0.0_dp)
  complex(dp), parameter :: CONE = (1.0_dp, 0.0_dp)
  ! Finite-difference epsilon: with ~33 digits, use 1e-16
  real(dp), parameter :: EPSILON_FD = 1.0e-16_dp
end module lmg_constants
