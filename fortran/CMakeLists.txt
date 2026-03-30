cmake_minimum_required(VERSION 3.14)
project(lmg_qfi_dynamics Fortran)

set(CMAKE_Fortran_STANDARD 2008)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler flags
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")

  set(CMAKE_Fortran_FLAGS_DEBUG
      "-g -O0 -Wall -Wextra -fcheck=all -fbacktrace")

  set(CMAKE_Fortran_FLAGS_RELEASE
      "-O3 -march=native -mtune=native -funroll-loops -flto")

elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")

  set(CMAKE_Fortran_FLAGS_DEBUG
      "-g -O0 -warn all -check all -traceback")

  set(CMAKE_Fortran_FLAGS_RELEASE
      "-O3 -xHost -ipo -qopenmp")

endif()

# OpenMP
option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)

if(ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# CTest
enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
