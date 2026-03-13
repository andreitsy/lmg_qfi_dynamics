module lmg_io
  use lmg_constants, only: dp
  use lmg_types, only: QFIInformation_type
  implicit none
  private
  public :: save_to_file_qfi_dynamics, build_output_filename

contains

  !> Save QFI dynamics results to CSV file.
  !> Format: time,m_x,m_y,m_z,qfi,qfi_raw
  subroutine save_to_file_qfi_dynamics(results, num_results, output_file)
    type(QFIInformation_type), intent(in) :: results(:)
    integer, intent(in) :: num_results
    character(len=*), intent(in) :: output_file

    integer :: unit_num, i, ios

    open(newunit=unit_num, file=output_file, status='replace', action='write', iostat=ios)
    if (ios /= 0) then
      write(*,*) 'ERROR: Cannot open file for writing: ', trim(output_file)
      return
    end if

    ! Header
    write(unit_num, '(A)') 'time,m_x,m_y,m_z,qfi,qfi_raw'

    ! Data rows
    do i = 1, num_results
      write(unit_num, '(I0,",",ES23.15E3,",",ES23.15E3,",",ES23.15E3,",",ES23.15E3,",",ES23.15E3)') &
        results(i)%time, results(i)%m_x, results(i)%m_y, results(i)%m_z, &
        results(i)%qfi, results(i)%qfi_raw
    end do

    close(unit_num)
  end subroutine save_to_file_qfi_dynamics

  !> Build output filename: InitialState.<STATE>_N=<N>_B=<B>.csv
  subroutine build_output_filename(state_label, N, B, output_dir, filename)
    character(len=*), intent(in) :: state_label, output_dir
    integer, intent(in) :: N
    real(dp), intent(in) :: B
    character(len=256), intent(out) :: filename

    write(filename, '(A,"/InitialState.",A,"_N=",I0,"_B=",F4.2,".csv")') &
      trim(output_dir), trim(state_label), N, B
  end subroutine build_output_filename

end module lmg_io
