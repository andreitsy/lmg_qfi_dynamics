# Build of the quadruple-precision OpenMP solver.
#
#   make            build $(BUILD)/lmg_qfi
#   make test       build and run the Fortran test suite
#   make clean
#
# Variables: FC (default gfortran), FFLAGS, OPENMP=0 to build without OpenMP,
# BUILD (default ./build). On a cluster, build on the login node if the
# compute nodes have no complete toolchain.

ifeq ($(origin FC),default)
  FC := gfortran
endif
BUILD  ?= build
OPENMP ?= 1

# -ffp-contract=off: no FMA contraction, so the NumPy-compatible time grid is
# bit-for-bit reproducible. The hot loops are software binary128 arithmetic
# and gain nothing from -march=native, so the binary stays portable across
# the cluster nodes.
FFLAGS ?= -O2 -g
FFLAGS += -std=f2008 -fimplicit-none -ffp-contract=off -Wall -Wextra -Wno-compare-reals
ifeq ($(OPENMP),1)
  FFLAGS += -fopenmp
endif

LDFLAGS ?=
ifeq ($(shell uname -s),Linux)
  # Find libgfortran/libquadmath/libgomp at run time without loading a
  # compiler module on the compute nodes.
  LIBGFORTRAN := $(shell $(FC) -print-file-name=libgfortran.so)
  ifneq ($(filter /%,$(LIBGFORTRAN)),)
    LDFLAGS += -Wl,-rpath,$(dir $(realpath $(LIBGFORTRAN)))
  endif
endif

MODULES := lmg_kinds lmg_linalg lmg_operators lmg_floquet lmg_states \
           lmg_timegrid lmg_dynamics lmg_config lmg_io
OBJS    := $(MODULES:%=$(BUILD)/%.o)

.PHONY: all test clean
all: $(BUILD)/lmg_qfi

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.f90 | $(BUILD)
	$(FC) $(FFLAGS) -J$(BUILD) -c $< -o $@

$(BUILD)/%.o: tests/%.f90 | $(BUILD)
	$(FC) $(FFLAGS) -J$(BUILD) -c $< -o $@

$(BUILD)/lmg_qfi: $(OBJS) $(BUILD)/main.o
	$(FC) $(FFLAGS) $^ $(LDFLAGS) -o $@

$(BUILD)/test_lmg: $(OBJS) $(BUILD)/test_lmg.o
	$(FC) $(FFLAGS) $^ $(LDFLAGS) -o $@

test: $(BUILD)/test_lmg
	$(BUILD)/test_lmg

clean:
	rm -rf $(BUILD)

# Module dependencies
$(BUILD)/lmg_linalg.o:    $(BUILD)/lmg_kinds.o
$(BUILD)/lmg_operators.o: $(BUILD)/lmg_kinds.o
$(BUILD)/lmg_floquet.o:   $(BUILD)/lmg_kinds.o $(BUILD)/lmg_linalg.o $(BUILD)/lmg_operators.o
$(BUILD)/lmg_states.o:    $(BUILD)/lmg_kinds.o $(BUILD)/lmg_linalg.o $(BUILD)/lmg_operators.o
$(BUILD)/lmg_timegrid.o:  $(BUILD)/lmg_kinds.o
$(BUILD)/lmg_dynamics.o:  $(BUILD)/lmg_kinds.o $(BUILD)/lmg_linalg.o $(BUILD)/lmg_operators.o $(BUILD)/lmg_floquet.o
$(BUILD)/lmg_config.o:    $(BUILD)/lmg_kinds.o $(BUILD)/lmg_operators.o $(BUILD)/lmg_states.o
$(BUILD)/lmg_io.o:        $(BUILD)/lmg_kinds.o $(BUILD)/lmg_dynamics.o
$(BUILD)/main.o:          $(OBJS)
$(BUILD)/test_lmg.o:      $(OBJS)
