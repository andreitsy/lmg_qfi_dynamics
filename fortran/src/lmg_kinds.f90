!> Numeric kinds and constants shared by the whole solver.
!>
!> Everything is computed in IEEE 754 binary128 ("extended" / quadruple
!> precision, 113-bit significand, about 34 significant decimal digits).
!> Changing `wp` here is the only edit needed to build the code in another
!> real kind (for example `real64` for quick experiments).
module lmg_kinds
   use, intrinsic :: iso_fortran_env, only: real64, real128
   implicit none
   private

   public :: wp, dp, ik
   public :: pi, twopi, czero, cone, ci

   !> Working precision of all matrices, states and observables.
   integer, parameter :: wp = real128
   !> Double precision, used only to mimic NumPy when building the time grid.
   integer, parameter :: dp = real64
   !> Integer kind of the stroboscopic time (number of kicks): 128 bit,
   !> large enough for every time below 1.7e38 periods.
   integer, parameter :: ik = selected_int_kind(38)

   real(wp), parameter :: pi = &
      3.141592653589793238462643383279502884197169399375105820974944592_wp
   real(wp), parameter :: twopi = 2.0_wp*pi

   complex(wp), parameter :: czero = (0.0_wp, 0.0_wp)
   complex(wp), parameter :: cone = (1.0_wp, 0.0_wp)
   complex(wp), parameter :: ci = (0.0_wp, 1.0_wp)

end module lmg_kinds
