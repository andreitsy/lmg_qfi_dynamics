program test_io
  use lmg_constants, only: dp
  use lmg_types, only: QFIInformation_type
  use lmg_io, only: save_to_file_qfi_dynamics, build_output_filename
  use test_utils
  implicit none

  call test_suite_start('lmg_io')

  call test_csv_header_and_values()
  call test_build_filename()
  call test_negative_values()

  call test_suite_end()
  if (tests_failed > 0) stop 1

contains

  subroutine test_csv_header_and_values()
    type(QFIInformation_type) :: results(2)
    character(len=256) :: filename, line
    integer :: unit_num, ios

    call test_start('csv_header_and_values')

    filename = 'test_output_io.csv'

    results(1)%time = 1
    results(1)%m_x = 0.1_dp
    results(1)%m_y = 0.2_dp
    results(1)%m_z = 0.3_dp
    results(1)%qfi = 0.5_dp
    results(1)%qfi_raw = 2.0_dp

    results(2)%time = 10
    results(2)%m_x = -0.1_dp
    results(2)%m_y = -0.2_dp
    results(2)%m_z = 0.4_dp
    results(2)%qfi = 1.5_dp
    results(2)%qfi_raw = 150.0_dp

    call save_to_file_qfi_dynamics(results, 2, filename)

    ! Read back and check header
    open(newunit=unit_num, file=filename, status='old', action='read', iostat=ios)
    call assert_true(ios == 0, 'file opened')

    read(unit_num, '(A)') line
    call assert_true(trim(line) == 'time,m_x,m_y,m_z,qfi,qfi_raw', 'header correct')

    ! Read first data line
    read(unit_num, '(A)') line
    call assert_true(len_trim(line) > 0, 'data line 1 not empty')

    ! Read second data line
    read(unit_num, '(A)') line
    call assert_true(len_trim(line) > 0, 'data line 2 not empty')

    close(unit_num)

    ! Clean up
    open(newunit=unit_num, file=filename, status='old')
    close(unit_num, status='delete')

    call test_pass()
  end subroutine

  subroutine test_build_filename()
    character(len=256) :: filename

    call test_start('build_filename')

    call build_output_filename('GS_PHYS', 5, 0.20_dp, 'results', filename)
    call assert_true(index(filename, 'InitialState.GS_PHYS') > 0, 'contains state label')
    call assert_true(index(filename, 'N=5') > 0, 'contains N=5')
    call assert_true(index(filename, '.csv') > 0, 'ends with .csv')
    call test_pass()
  end subroutine

  subroutine test_negative_values()
    type(QFIInformation_type) :: results(1)
    character(len=256) :: filename, line
    integer :: unit_num, ios

    call test_start('negative_values_in_csv')

    filename = 'test_output_neg.csv'

    results(1)%time = 5
    results(1)%m_x = -0.3_dp
    results(1)%m_y = -0.4_dp
    results(1)%m_z = -0.5_dp
    results(1)%qfi = 0.01_dp
    results(1)%qfi_raw = 0.25_dp

    call save_to_file_qfi_dynamics(results, 1, filename)

    open(newunit=unit_num, file=filename, status='old', action='read', iostat=ios)
    read(unit_num, '(A)') line  ! header
    read(unit_num, '(A)') line  ! data
    call assert_true(index(line, '-') > 0, 'contains negative sign')
    close(unit_num)

    open(newunit=unit_num, file=filename, status='old')
    close(unit_num, status='delete')

    call test_pass()
  end subroutine

end program test_io
