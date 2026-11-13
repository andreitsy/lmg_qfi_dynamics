!> Physical model: parameters, collective spin operators and the LMG
!> Hamiltonian in the symmetric (Dicke) sector.
!>
!> Basis convention (identical to the Python package): index i = 1..N+1 is
!> the Dicke state |S, m> with S = N/2 and m = S - (i - 1), i.e. the first
!> basis vector is the fully polarised state |S, S>.
module lmg_operators
   use lmg_kinds, only: wp, pi, czero
   implicit none
   private

   public :: lmg_model
   public :: ladder_coefficients, spin_operators, hamiltonian_h0, probe_operator
   public :: magnetization, apply_mom_observable
   public :: obs_sz, obs_x, obs_y, n_mom_observables

   !> Observables of the method of moments (see apply_mom_observable).
   integer, parameter :: obs_sz = 1, obs_x = 2, obs_y = 3, n_mom_observables = 3

   !> Parameters of the kicked LMG model with an AC probe field,
   !>
   !>   H(t) = -(2J/N) Sz^2 - 2B Sx - phi * sum_m delta(t - mT) Sx
   !>          + h sin(omega t + phi0) S_alpha,     omega = 2 pi / (nu T),
   !>
   !> with S_alpha = sin(theta) cos(varphi) Sx + sin(theta) sin(varphi) Sy
   !> + cos(theta) Sz. Each period T is integrated with `steps` Trotter steps.
   type :: lmg_model
      integer  :: nspins = 20          !< N
      real(wp) :: j = 1.0_wp           !< J, Sz^2 coupling
      real(wp) :: b = 0.4_wp           !< B, transverse field
      real(wp) :: t = 1.0_wp           !< T, kick period
      real(wp) :: phi_kick = pi        !< phi, kick angle
      real(wp) :: h = 0.0_wp           !< probe amplitude at which the QFI is evaluated
      integer  :: nu = 2               !< probe period in units of T
      real(wp) :: phi0 = 0.0_wp        !< initial phase of the probe
      real(wp) :: theta = 0.0_wp       !< polar angle of the probe direction
      real(wp) :: varphi = 0.0_wp      !< azimuthal angle of the probe direction
      integer  :: steps = 10           !< Trotter steps per period T
   end type lmg_model

contains

   !> Matrix elements of S-: c(i) = <S, m_i - 1| S- |S, m_i> = sqrt(i (N - i + 1)),
   !> i = 1..N.
   pure subroutine ladder_coefficients(nspins, c)
      integer, intent(in)   :: nspins
      real(wp), intent(out) :: c(:)
      integer :: i

      do i = 1, nspins
         c(i) = sqrt(real(i, wp)*real(nspins - i + 1, wp))
      end do
   end subroutine ladder_coefficients

   !> Dense collective spin operators Sx, Sy, Sz of size (N+1) x (N+1).
   pure subroutine spin_operators(nspins, sx, sy, sz)
      integer, intent(in)      :: nspins
      complex(wp), intent(out) :: sx(:, :), sy(:, :), sz(:, :)
      real(wp) :: c(nspins)
      integer :: i

      call ladder_coefficients(nspins, c)
      sx = czero; sy = czero; sz = czero
      do i = 1, nspins + 1
         sz(i, i) = cmplx(0.5_wp*real(nspins, wp) - real(i - 1, wp), 0.0_wp, wp)
      end do
      do i = 1, nspins
         sx(i, i + 1) = cmplx(0.5_wp*c(i), 0.0_wp, wp)
         sx(i + 1, i) = cmplx(0.5_wp*c(i), 0.0_wp, wp)
         sy(i, i + 1) = cmplx(0.0_wp, -0.5_wp*c(i), wp)
         sy(i + 1, i) = cmplx(0.0_wp, 0.5_wp*c(i), wp)
      end do
   end subroutine spin_operators

   !> H0 = -(2J/N) Sz^2 - 2B Sx.
   pure subroutine hamiltonian_h0(nspins, j, b, h0)
      integer, intent(in)      :: nspins
      real(wp), intent(in)     :: j, b
      complex(wp), intent(out) :: h0(:, :)
      real(wp) :: c(nspins), m
      integer :: i

      call ladder_coefficients(nspins, c)
      h0 = czero
      do i = 1, nspins + 1
         m = 0.5_wp*real(nspins, wp) - real(i - 1, wp)
         h0(i, i) = cmplx(-(2.0_wp*j/real(nspins, wp))*m*m, 0.0_wp, wp)
      end do
      do i = 1, nspins
         h0(i, i + 1) = cmplx(-b*c(i), 0.0_wp, wp)
         h0(i + 1, i) = cmplx(-b*c(i), 0.0_wp, wp)
      end do
   end subroutine hamiltonian_h0

   !> Probe operator S_alpha for the direction (theta, varphi).
   pure subroutine probe_operator(nspins, theta, varphi, sa)
      integer, intent(in)      :: nspins
      real(wp), intent(in)     :: theta, varphi
      complex(wp), intent(out) :: sa(:, :)
      real(wp) :: c(nspins)
      complex(wp) :: up
      integer :: i

      ! sin(theta) (cos(varphi) Sx + sin(varphi) Sy) has the elements
      ! sin(theta) c_i/2 exp(-/+ i varphi) above/below the diagonal.
      call ladder_coefficients(nspins, c)
      up = sin(theta)*cmplx(cos(varphi), -sin(varphi), wp)
      sa = czero
      do i = 1, nspins + 1
         sa(i, i) = cmplx(cos(theta)*(0.5_wp*real(nspins, wp) - real(i - 1, wp)), 0.0_wp, wp)
      end do
      do i = 1, nspins
         sa(i, i + 1) = 0.5_wp*c(i)*up
         sa(i + 1, i) = 0.5_wp*c(i)*conjg(up)
      end do
   end subroutine probe_operator

   !> Magnetisation per spin, (<Sx>, <Sy>, <Sz>)/N, of a state vector.
   pure subroutine magnetization(nspins, c, psi, mx, my, mz)
      integer, intent(in)     :: nspins
      real(wp), intent(in)    :: c(:)
      complex(wp), intent(in) :: psi(:)
      real(wp), intent(out)   :: mx, my, mz
      complex(wp) :: sminus
      real(wp) :: sz
      integer :: i

      ! <S-> = sum_i c_i conj(psi_{i+1}) psi_i ;  <Sx> = Re <S->,  <Sy> = -Im <S->
      sminus = czero
      do i = 1, nspins
         sminus = sminus + c(i)*conjg(psi(i + 1))*psi(i)
      end do
      sz = 0.0_wp
      do i = 1, nspins + 1
         sz = sz + (0.5_wp*real(nspins, wp) - real(i - 1, wp)) &
              *(real(psi(i), wp)**2 + aimag(psi(i))**2)
      end do
      mx = real(sminus, wp)/real(nspins, wp)
      my = -aimag(sminus)/real(nspins, wp)
      mz = sz/real(nspins, wp)
   end subroutine magnetization

   !> w = O v for the method-of-moments observables
   !>   obs_sz:  Sz
   !>   obs_x:   spin-flip parity X = (sigma^x)^(tensor N), |S, m> -> |S, -m>
   !>            (= exp(-i pi Sx) up to the global phase i^N)
   !>   obs_y:   dressed parity-magnetisation Y = i Sz X (Hermitian, {Sz, X} = 0)
   pure subroutine apply_mom_observable(nspins, obs, v, w)
      integer, intent(in)      :: nspins, obs
      complex(wp), intent(in)  :: v(:)
      complex(wp), intent(out) :: w(:)
      integer :: i, n
      real(wp) :: m

      n = nspins + 1
      do i = 1, n
         m = 0.5_wp*real(nspins, wp) - real(i - 1, wp)
         select case (obs)
         case (obs_sz)
            w(i) = m*v(i)
         case (obs_x)
            w(i) = v(n + 1 - i)
         case default
            w(i) = cmplx(0.0_wp, m, wp)*v(n + 1 - i)
         end select
      end do
   end subroutine apply_mom_observable

end module lmg_operators
