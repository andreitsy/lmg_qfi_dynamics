!> Initial states of the simulation.
!>
!> | name    | Python enum | construction                                   |
!> |---------|-------------|------------------------------------------------|
!> | Phys    | PHYS        | fully polarised Dicke state |S, S>              |
!> | GS_phys | GS_PHYS     | (ground + first excited state of H0)/sqrt(2)    |
!> | GS_cat  | GS_CAT      | ground state of H0                              |
!> | CatSum  | CAT_SUM     | (|S, S> + |S, -S>)/sqrt(2)                      |
module lmg_states
   use lmg_kinds, only: wp, czero, cone
   use lmg_linalg, only: herm_eig
   use lmg_operators, only: hamiltonian_h0, ladder_coefficients, magnetization
   implicit none
   private

   public :: n_state_kinds, st_phys, st_gs_phys, st_gs_cat, st_cat_sum
   public :: state_cli_name, state_enum_name, state_from_name
   public :: ground_doublet, initial_state

   integer, parameter :: n_state_kinds = 4
   !> Same order as the default list of the Python driver.
   integer, parameter :: st_phys = 1, st_gs_phys = 2, st_gs_cat = 3, st_cat_sum = 4

   character(len=*), parameter :: cli_names(n_state_kinds) = &
      [character(len=7) :: 'Phys', 'GS_phys', 'GS_cat', 'CatSum']
   character(len=*), parameter :: enum_names(n_state_kinds) = &
      [character(len=7) :: 'PHYS', 'GS_PHYS', 'GS_CAT', 'CAT_SUM']

contains

   !> Name used on the command line (value of the Python enum), e.g. GS_phys.
   pure function state_cli_name(kind) result(name)
      integer, intent(in) :: kind
      character(len=:), allocatable :: name

      name = trim(cli_names(kind))
   end function state_cli_name

   !> Name of the Python enum member, used in output file names, e.g. GS_PHYS.
   pure function state_enum_name(kind) result(name)
      integer, intent(in) :: kind
      character(len=:), allocatable :: name

      name = trim(enum_names(kind))
   end function state_enum_name

   !> Map a (case-insensitive) state name to its kind; 0 if unknown.
   pure function state_from_name(name) result(kind)
      character(len=*), intent(in) :: name
      integer :: kind, i

      kind = 0
      do i = 1, n_state_kinds
         if (lower(name) == lower(trim(cli_names(i))) .or. lower(name) == lower(trim(enum_names(i)))) then
            kind = i
            return
         end if
      end do
   end function state_from_name

   pure function lower(s) result(t)
      character(len=*), intent(in) :: s
      character(len=len_trim(s)) :: t
      integer :: i, c

      do i = 1, len_trim(s)
         c = iachar(s(i:i))
         if (c >= iachar('A') .and. c <= iachar('Z')) c = c + 32
         t(i:i) = achar(c)
      end do
   end function lower

   !> Two lowest eigenpairs (e0 <= e1) of H0 = -(2J/N) Sz^2 - 2B Sx.
   !>
   !> H0 commutes with the reflection m -> -m, so it is diagonalised
   !> separately in the even and odd sectors. In the symmetry-broken phase
   !> the ground and first excited states form a doublet with an
   !> exponentially small splitting; they live in different sectors and are
   !> therefore obtained to full working precision, where a diagonalisation
   !> of the whole matrix would mix them by (rounding error)/(splitting).
   !> The eigenvectors are real, normalised, and their largest component is
   !> positive.
   subroutine ground_doublet(nspins, j, b, e0, e1, v0, v1)
      integer, intent(in)      :: nspins
      real(wp), intent(in)     :: j, b
      real(wp), intent(out)    :: e0, e1
      complex(wp), intent(out) :: v0(:), v1(:)

      complex(wp), allocatable :: h0(:, :), he(:, :), ho(:, :), ve(:, :), vo(:, :)
      real(wp), allocatable :: we(:), wo(:), eall(:)
      integer, allocatable :: parity(:), idx(:)
      real(wp) :: rt2
      integer :: n, no, ne, i, k, i0, i1

      n = nspins + 1
      no = n/2
      ne = n - no
      rt2 = sqrt(2.0_wp)

      allocate (h0(n, n), he(ne, ne), ho(no, no), ve(ne, ne), vo(no, no), we(ne), wo(no))
      call hamiltonian_h0(nspins, j, b, h0)

      ! Even/odd blocks in the basis (e_i +/- e_{n+1-i})/sqrt(2), i <= n/2,
      ! plus the unpaired middle state (even) when n is odd.
      do k = 1, no
         do i = 1, no
            he(i, k) = h0(i, k) + h0(i, n + 1 - k)
            ho(i, k) = h0(i, k) - h0(i, n + 1 - k)
         end do
      end do
      if (ne > no) then
         do i = 1, no
            he(i, ne) = rt2*h0(i, ne)
            he(ne, i) = conjg(he(i, ne))
         end do
         he(ne, ne) = h0(ne, ne)
      end if
      call herm_eig(he, we, ve)
      call herm_eig(ho, wo, vo)

      ! Merge both spectra and pick the two lowest levels.
      allocate (eall(n), parity(n), idx(n))
      do i = 1, ne
         eall(i) = we(i); parity(i) = 0; idx(i) = i
      end do
      do i = 1, no
         eall(ne + i) = wo(i); parity(ne + i) = 1; idx(ne + i) = i
      end do
      i0 = 1
      do i = 2, n
         if (eall(i) < eall(i0)) i0 = i
      end do
      i1 = 0
      do i = 1, n
         if (i == i0) cycle
         if (i1 == 0) then
            i1 = i
         else if (eall(i) < eall(i1)) then
            i1 = i
         end if
      end do

      e0 = eall(i0)
      e1 = eall(i1)
      call expand(i0, v0)
      call expand(i1, v1)

   contains

      !> Eigenvector `which` of the merged list in the full Dicke basis.
      subroutine expand(which, v)
         integer, intent(in)      :: which
         complex(wp), intent(out) :: v(:)
         complex(wp) :: phase
         integer :: l, col, lmax

         col = idx(which)
         v = czero
         if (parity(which) == 0) then
            do l = 1, no
               v(l) = ve(l, col)/rt2
               v(n + 1 - l) = ve(l, col)/rt2
            end do
            if (ne > no) v(ne) = ve(ne, col)
         else
            do l = 1, no
               v(l) = vo(l, col)/rt2
               v(n + 1 - l) = -vo(l, col)/rt2
            end do
         end if
         lmax = maxloc(abs(v), dim=1)
         phase = conjg(v(lmax))/abs(v(lmax))
         v = v*phase
         v = v/sqrt(sum(real(v, wp)**2 + aimag(v)**2))
      end subroutine expand

   end subroutine ground_doublet

   !> Normalised initial state of the given kind.
   !>
   !> For GS_phys the sign of the superposition is fixed by the magnetisation:
   !> `gs_up = .true.` selects the symmetry-broken state with <Sz> > 0 (the
   !> one adiabatically connected to |S, S>), `.false.` the one with <Sz> < 0.
   !> The two are mapped onto each other by the Z2 symmetry of the model
   !> (m_y, m_z change sign, m_x and the QFI at h = 0 are identical).
   subroutine initial_state(kind, nspins, v0, v1, gs_up, psi)
      integer, intent(in)      :: kind, nspins
      complex(wp), intent(in)  :: v0(:), v1(:)
      logical, intent(in)      :: gs_up
      complex(wp), intent(out) :: psi(:)

      real(wp) :: c(nspins), mx, my, mz
      integer :: n

      n = nspins + 1
      psi = czero
      select case (kind)
      case (st_phys)
         psi(1) = cone
      case (st_cat_sum)
         psi(1) = cone
         psi(n) = cone
      case (st_gs_cat)
         psi = v0
      case (st_gs_phys)
         psi = v0 + v1
         call ladder_coefficients(nspins, c)
         call magnetization(nspins, c, psi, mx, my, mz)
         if ((mz < 0.0_wp) .eqv. gs_up) psi = v0 - v1
      case default
         error stop 'initial_state: unknown state kind'
      end select
      psi = psi/sqrt(sum(real(psi, wp)**2 + aimag(psi)**2))
   end subroutine initial_state

end module lmg_states
