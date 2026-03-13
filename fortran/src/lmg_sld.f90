module lmg_sld
  use lmg_constants, only: dp, CZERO
  use lmg_types, only: SLDInformation_type
  implicit none
  private
  public :: compute_sld_matrix, sld_expectation, sld_squared_expectation, compute_sld_info

contains

  !> SLD matrix: L = 2(|dket><ket| + |ket><dket|)
  subroutine compute_sld_matrix(dket, ket, dim, L)
    complex(dp), intent(in) :: dket(dim), ket(dim)
    integer, intent(in) :: dim
    complex(dp), intent(out) :: L(dim, dim)
    integer :: i, j

    do j = 1, dim
      do i = 1, dim
        L(i,j) = 2.0_dp * (dket(i) * conjg(ket(j)) + ket(i) * conjg(dket(j)))
      end do
    end do
  end subroutine compute_sld_matrix

  !> <L> = 4 * Re(<ket|dket>)
  function sld_expectation(dket, ket, dim) result(L_exp)
    complex(dp), intent(in) :: dket(dim), ket(dim)
    integer, intent(in) :: dim
    real(dp) :: L_exp
    complex(dp) :: alpha

    alpha = dot_product(ket, dket)  ! Fortran conjugates first arg: <ket|dket>
    L_exp = 4.0_dp * real(alpha, dp)
  end function sld_expectation

  !> <L^2> = ||L|ket>||^2, where L|ket> = 2(|dket> + <dket|ket>|ket>)
  function sld_squared_expectation(dket, ket, dim) result(L2_exp)
    complex(dp), intent(in) :: dket(dim), ket(dim)
    integer, intent(in) :: dim
    real(dp) :: L2_exp
    complex(dp) :: alpha_conj, L_ket(dim)

    ! <dket|ket> = conjg(<ket|dket>)
    alpha_conj = dot_product(dket, ket)
    L_ket = 2.0_dp * (dket + alpha_conj * ket)
    L2_exp = real(dot_product(L_ket, L_ket), dp)
  end function sld_squared_expectation

  !> Compute all SLD observables at a single time point
  subroutine compute_sld_info(dket, ket, dim, N, time, info)
    complex(dp), intent(in) :: dket(dim), ket(dim)
    integer, intent(in) :: dim, N, time
    type(SLDInformation_type), intent(out) :: info

    info%time = time
    info%L_expectation = sld_expectation(dket, ket, dim)
    info%L_squared_expectation = sld_squared_expectation(dket, ket, dim)
    info%qfi_from_sld = info%L_squared_expectation / (real(N, dp)**2 * real(time, dp)**2)
  end subroutine compute_sld_info

end module lmg_sld
