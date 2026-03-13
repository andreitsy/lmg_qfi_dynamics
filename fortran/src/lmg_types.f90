module lmg_types
  use lmg_constants, only: dp
  implicit none

  ! Initial state identifiers
  integer, parameter :: STATE_GS_PHYS = 1
  integer, parameter :: STATE_GS_CAT  = 2
  integer, parameter :: STATE_PHYS    = 3
  integer, parameter :: STATE_CAT_SUM = 4

  character(len=10), parameter :: STATE_LABELS(4) = &
    [character(len=10) :: 'GS_PHYS', 'GS_CAT', 'PHYS', 'CAT_SUM']

  ! Floquet unitary eigendecomposition
  type :: UF_type
    integer :: dim
    complex(dp), allocatable :: eigenvalues(:)
    complex(dp), allocatable :: U(:,:)
    complex(dp), allocatable :: U_inv(:,:)
  end type UF_type

  ! QFI results for a single time point
  type :: QFIInformation_type
    real(dp) :: m_x, m_y, m_z
    real(dp) :: qfi          ! normalized: QFI / (N^2 * t^2)
    integer  :: time
    real(dp) :: qfi_raw      ! unnormalized QFI
  end type QFIInformation_type

  ! SLD results for a single time point
  type :: SLDInformation_type
    integer  :: time
    real(dp) :: L_expectation
    real(dp) :: L_squared_expectation
    real(dp) :: qfi_from_sld
  end type SLDInformation_type

  ! Simulation parameters
  type :: SimulationParams_type
    integer  :: N                       ! number of spins
    real(dp) :: J                       ! ZZ coupling
    real(dp) :: B                       ! X coupling
    real(dp) :: T                       ! period
    real(dp) :: phi_kick_phase          ! kick phase
    real(dp) :: h                       ! AC drive amplitude
    real(dp) :: varphi                  ! field direction angle
    real(dp) :: theta                   ! field direction polar angle
    real(dp) :: phi_0                   ! initial phase
    integer  :: freq                    ! frequency (nu)
    integer  :: steps_floquet_unitary   ! Trotter steps per period
    integer  :: num_points              ! points per decade in logspace
    integer  :: dps                     ! precision (unused in Fortran, kept for config compat)
    character(len=256) :: output_dir
  end type SimulationParams_type

end module lmg_types
