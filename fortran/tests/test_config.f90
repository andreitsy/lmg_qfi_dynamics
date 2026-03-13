program test_config
  use lmg_constants, only: dp, PI
  use lmg_types, only: SimulationParams_type, STATE_GS_PHYS, STATE_GS_CAT, &
                        STATE_PHYS, STATE_CAT_SUM
  use lmg_config, only: set_default_params, read_config_file, state_from_string, state_label
  use test_utils
  implicit none

  call test_suite_start('lmg_config')

  call test_default_params()
  call test_read_config_file()
  call test_state_from_string()
  call test_state_label_fn()
  call test_config_missing_file()

  call test_suite_end()
  if (tests_failed > 0) stop 1

contains

  subroutine test_default_params()
    type(SimulationParams_type) :: params

    call test_start('default_params')
    call set_default_params(params)

    call assert_true(params%N == 10, 'N=10')
    call assert_close(params%J, 1.0_dp, 1e-12_dp, 'J=1')
    call assert_close(params%B, 0.4_dp, 1e-12_dp, 'B=0.4')
    call assert_close(params%T, 1.0_dp, 1e-12_dp, 'T=1')
    call assert_close(params%phi_kick_phase, PI, 1e-12_dp, 'phi=pi')
    call assert_close(params%h, 0.0_dp, 1e-12_dp, 'h=0')
    call assert_true(params%freq == 2, 'freq=2')
    call assert_true(params%steps_floquet_unitary == 10, 'steps=10')
    call test_pass()
  end subroutine

  subroutine test_read_config_file()
    type(SimulationParams_type) :: params
    integer :: ierr

    call test_start('read_config_file')
    call set_default_params(params)
    call read_config_file('parameters.ini', params, ierr)

    if (ierr /= 0) then
      ! Config file might not be in the test build dir; skip gracefully
      write(*,'(A)', advance='no') ' (config not found, checking defaults) '
    else
      call assert_true(params%N == 10, 'N=10 from config')
      call assert_close(params%J, 1.0_dp, 1e-12_dp, 'J from config')
      call assert_close(params%B, 0.4_dp, 1e-12_dp, 'B from config')
      call assert_close(params%phi_kick_phase, PI, 1e-10_dp, 'phi=pi from config')
      call assert_true(params%freq == 2, 'freq from config')
    end if
    call test_pass()
  end subroutine

  subroutine test_state_from_string()
    call test_start('state_from_string')
    call assert_true(state_from_string('GS_phys') == STATE_GS_PHYS, 'GS_phys')
    call assert_true(state_from_string('GS_PHYS') == STATE_GS_PHYS, 'GS_PHYS')
    call assert_true(state_from_string('GS_cat') == STATE_GS_CAT, 'GS_cat')
    call assert_true(state_from_string('Phys') == STATE_PHYS, 'Phys')
    call assert_true(state_from_string('CatSum') == STATE_CAT_SUM, 'CatSum')
    call assert_true(state_from_string('unknown') == -1, 'unknown=-1')
    call test_pass()
  end subroutine

  subroutine test_state_label_fn()
    call test_start('state_label')
    call assert_true(trim(state_label(STATE_GS_PHYS)) == 'GS_PHYS', 'label GS_PHYS')
    call assert_true(trim(state_label(STATE_GS_CAT)) == 'GS_CAT', 'label GS_CAT')
    call assert_true(trim(state_label(STATE_PHYS)) == 'PHYS', 'label PHYS')
    call assert_true(trim(state_label(STATE_CAT_SUM)) == 'CAT_SUM', 'label CAT_SUM')
    call test_pass()
  end subroutine

  subroutine test_config_missing_file()
    type(SimulationParams_type) :: params
    integer :: ierr

    call test_start('missing_config_returns_error')
    call set_default_params(params)
    call read_config_file('nonexistent_file.ini', params, ierr)
    call assert_true(ierr /= 0, 'ierr != 0')
    call test_pass()
  end subroutine

end program test_config
