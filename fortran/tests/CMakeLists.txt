# Build test_utils as a library to avoid parallel .mod file race
add_library(test_utils_lib STATIC test_utils.f90)
target_link_libraries(test_utils_lib PUBLIC lmg_qfi)

# Macro to add a test executable
macro(add_fortran_test test_name)
  add_executable(${test_name} ${test_name}.f90)
  target_link_libraries(${test_name} PRIVATE test_utils_lib)
  target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  add_test(NAME ${test_name} COMMAND ${test_name})
endmacro()

add_fortran_test(test_linalg)
add_fortran_test(test_operators)
add_fortran_test(test_evolution)
add_fortran_test(test_qfi)
add_fortran_test(test_sld)
add_fortran_test(test_config)
add_fortran_test(test_io)
add_fortran_test(test_simulation)
