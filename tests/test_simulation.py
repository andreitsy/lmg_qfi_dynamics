"""Tests for the simulation module."""

import pytest
import numpy as np
import mpmath as mp

from lmg_qfi import (
    generate_time_interval,
    InitialState,
    SimulationParams,
)


class TestGenerateTimeInterval:
    """Tests for time interval generation."""

    def test_time_interval_basic(self):
        """Test basic time interval generation."""
        interval = generate_time_interval(10, 5)
        
        assert len(interval) > 0
        assert interval[0] == 1
        assert interval[-1] > interval[0]

    def test_time_interval_monotonically_increasing(self):
        """Test that time interval is monotonically increasing."""
        interval = generate_time_interval(20, 6)
        
        for i in range(1, len(interval)):
            assert interval[i] > interval[i - 1], f"Not increasing at index {i}"

    def test_time_interval_starts_at_one(self):
        """Test that time interval starts at 1."""
        interval = generate_time_interval(10, 4)
        assert interval[0] == 1

    def test_time_interval_contains_linear_part(self):
        """Test that time interval contains linear part from 1 to 99."""
        interval = generate_time_interval(10, 5)
        
        # Check that 1 through 99 are in the interval
        for t in range(1, 100):
            assert t in interval, f"Missing time point {t}"

    @pytest.mark.parametrize("max_degree", [3, 4, 5, 6])
    def test_time_interval_max_degree(self, max_degree):
        """Test that max time is around 10^max_degree."""
        interval = generate_time_interval(10, max_degree)
        
        max_time = interval[-1]
        assert max_time >= 10 ** (max_degree - 1)
        assert max_time <= 10 ** (max_degree + 1)

    def test_time_interval_invalid_max_degree(self):
        """Test that max_degree <= 1 raises ValueError."""
        with pytest.raises(ValueError):
            generate_time_interval(10, 1)
        
        with pytest.raises(ValueError):
            generate_time_interval(10, 0)

    @pytest.mark.parametrize("num_points", [5, 10, 50])
    def test_time_interval_num_points_affects_density(self, num_points):
        """Test that num_points affects the density of points."""
        interval_sparse = generate_time_interval(5, 5)
        interval_dense = generate_time_interval(50, 5)
        
        # More num_points should give more total points
        assert len(interval_dense) > len(interval_sparse)

    def test_time_interval_all_integers(self):
        """Test that all time points are integers."""
        interval = generate_time_interval(10, 5)
        
        for t in interval:
            assert isinstance(t, int), f"Time point {t} is not an integer"

    def test_time_interval_no_duplicates(self):
        """Test that time interval has no duplicate values."""
        interval = generate_time_interval(20, 5)
        
        # Convert to set and compare length
        assert len(interval) == len(set(interval))


class TestSimulationParamsIntegration:
    """Integration tests for simulation parameters."""

    def test_simulation_params_with_all_initial_states(self):
        """Test that all initial states are valid."""
        params = SimulationParams(
            run_arguments={"dps": 50, "steps_floquet_unitary": 10, "num_points": 10},
            N=10,
            J=mp.mpf(1.0),
            B=mp.mpf(0.4),
        )
        
        all_states = [
            InitialState.GS_PHYS,
            InitialState.GS_CAT,
            InitialState.CAT_SUM,
            InitialState.PHYS,
        ]
        
        for state in all_states:
            # Just verify we can access the state
            assert state.value is not None

    def test_simulation_params_typical_values(self):
        """Test simulation params with typical physical values."""
        params = SimulationParams(
            run_arguments={
                "dps": 50,
                "steps_floquet_unitary": 100,
                "num_points": 100,
                "output_dir": "results",
            },
            N=20,
            J=mp.mpf(1.0),
            B=mp.mpf(0.4),
            T=mp.mpf(1.0),
            phi_kick_phase=mp.pi,
            h=mp.mpf(0.0),
            freq=2,
        )
        
        # Verify all parameters are set correctly
        assert params.N == 20
        assert 0 < float(params.B) < float(params.J)  # B < J for ferromagnetic phase
        assert float(params.T) > 0
        assert params.freq > 0

    def test_simulation_params_critical_point(self):
        """Test simulation params near critical point B/J = 1."""
        params = SimulationParams(
            run_arguments={"dps": 50},
            N=20,
            J=mp.mpf(1.0),
            B=mp.mpf(1.0),  # Critical point
        )
        
        assert float(params.B / params.J) == 1.0

    def test_simulation_params_strong_field(self):
        """Test simulation params in paramagnetic phase B > J."""
        params = SimulationParams(
            run_arguments={"dps": 50},
            N=20,
            J=mp.mpf(1.0),
            B=mp.mpf(2.0),  # Paramagnetic phase
        )
        
        assert float(params.B) > float(params.J)


class TestInitialStateProperties:
    """Tests for initial state physical properties."""

    def test_initial_states_are_distinct(self):
        """Test that all initial states have distinct values."""
        values = [state.value for state in InitialState]
        assert len(values) == len(set(values))

    def test_gs_phys_naming(self):
        """Test GS_PHYS state naming convention."""
        assert "GS" in InitialState.GS_PHYS.value
        assert "phys" in InitialState.GS_PHYS.value.lower()

    def test_gs_cat_naming(self):
        """Test GS_CAT state naming convention."""
        assert "GS" in InitialState.GS_CAT.value
        assert "cat" in InitialState.GS_CAT.value.lower()

    def test_cat_sum_naming(self):
        """Test CAT_SUM state naming convention."""
        assert "Cat" in InitialState.CAT_SUM.value

    def test_phys_naming(self):
        """Test PHYS state naming convention."""
        assert InitialState.PHYS.value == "Phys"
