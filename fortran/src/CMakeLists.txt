# Library containing all LMG QFI modules
add_library(lmg_qfi
  lmg_constants.f90
  lmg_types.f90
  lmg_linalg.f90
  lmg_operators.f90
  lmg_evolution.f90
  lmg_qfi.f90
  lmg_sld.f90
  lmg_config.f90
  lmg_io.f90
  lmg_simulation.f90
)

# Expose module directory for consumers
target_include_directories(lmg_qfi PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

if(ENABLE_OPENMP AND OpenMP_Fortran_FOUND)
  target_link_libraries(lmg_qfi PUBLIC OpenMP::OpenMP_Fortran)
  target_compile_definitions(lmg_qfi PUBLIC USE_OPENMP)
endif()

# Main executable
add_executable(lmg_qfi_sim main.f90)
target_link_libraries(lmg_qfi_sim PRIVATE lmg_qfi)
