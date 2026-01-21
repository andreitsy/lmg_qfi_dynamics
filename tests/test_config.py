"""Tests for the config module."""

import pytest
import tempfile
import os
import mpmath as mp

from lmg_qfi import (
    InitialState,
    SimulationParams,
    QFIInformation,
    UF,
    read_defaults_args_from_config,
)


class TestInitialState:
    """Tests for InitialState enum."""

    def test_initial_state_values(self):
        """Test that InitialState enum has correct values."""
        assert InitialState.GS_PHYS.value == "GS_phys"
        assert InitialState.GS_CAT.value == "GS_cat"
        assert InitialState.CAT_SUM.value == "CatSum"
        assert InitialState.PHYS.value == "Phys"

    def test_initial_state_from_string(self):
        """Test creating InitialState from string value."""
        assert InitialState("GS_phys") == InitialState.GS_PHYS
        assert InitialState("GS_cat") == InitialState.GS_CAT
        assert InitialState("CatSum") == InitialState.CAT_SUM
        assert InitialState("Phys") == InitialState.PHYS

    def test_initial_state_invalid_value(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError):
            InitialState("invalid_state")

    def test_initial_state_iteration(self):
        """Test iterating over all initial states."""
        states = list(InitialState)
        assert len(states) == 4
        assert InitialState.GS_PHYS in states
        assert InitialState.GS_CAT in states
        assert InitialState.CAT_SUM in states
        assert InitialState.PHYS in states


class TestSimulationParams:
    """Tests for SimulationParams dataclass."""

    def test_simulation_params_creation(self):
        """Test creating SimulationParams with required fields."""
        params = SimulationParams(
            run_arguments={"dps": 50},
            N=20,
            J=mp.mpf(1.0),
            B=mp.mpf(0.4),
        )
        
        assert params.N == 20
        assert float(params.J) == 1.0
        assert float(params.B) == 0.4

    def test_simulation_params_defaults(self):
        """Test that SimulationParams has correct defaults."""
        params = SimulationParams(
            run_arguments={},
            N=10,
            J=mp.mpf(1.0),
            B=mp.mpf(0.5),
        )
        
        assert float(params.T) == 1.0
        assert float(params.h) == 0.0
        assert float(params.varphi) == 0.0
        assert float(params.theta) == 0.0
        assert params.freq == 2
        assert float(params.phi_0) == 0.0

    def test_simulation_params_str(self):
        """Test string representation of SimulationParams."""
        params = SimulationParams(
            run_arguments={"dps": 50},
            N=20,
            J=mp.mpf(1.0),
            B=mp.mpf(0.4),
        )
        
        str_repr = str(params)
        assert "N=20" in str_repr
        assert "B=0.4" in str_repr
        assert "J=1.0" in str_repr

    def test_simulation_params_all_fields(self):
        """Test SimulationParams with all fields specified."""
        params = SimulationParams(
            run_arguments={"dps": 100, "output_dir": "results"},
            N=40,
            J=mp.mpf(2.0),
            B=mp.mpf(0.8),
            T=mp.mpf(2.0),
            phi_kick_phase=mp.pi / 2,
            h=mp.mpf(0.1),
            varphi=mp.mpf(0.2),
            theta=mp.mpf(0.3),
            freq=4,
            phi_0=mp.mpf(0.5),
        )
        
        assert params.N == 40
        assert float(params.J) == 2.0
        assert float(params.B) == 0.8
        assert float(params.T) == 2.0
        assert params.freq == 4


class TestQFIInformation:
    """Tests for QFIInformation dataclass."""

    def test_qfi_information_creation(self):
        """Test creating QFIInformation."""
        info = QFIInformation(
            m_x=0.1,
            m_y=0.2,
            m_z=0.3,
            qfi=0.5,
            time=100,
            qfi_raw_value="0.5000000000",
        )
        
        assert info.m_x == 0.1
        assert info.m_y == 0.2
        assert info.m_z == 0.3
        assert info.qfi == 0.5
        assert info.time == 100
        assert info.qfi_raw_value == "0.5000000000"

    def test_qfi_information_fields(self):
        """Test that QFIInformation has all expected fields."""
        info = QFIInformation(
            m_x=0.0,
            m_y=0.0,
            m_z=1.0,
            qfi=0.0,
            time=1,
            qfi_raw_value="0",
        )
        
        # Check field access
        assert hasattr(info, 'm_x')
        assert hasattr(info, 'm_y')
        assert hasattr(info, 'm_z')
        assert hasattr(info, 'qfi')
        assert hasattr(info, 'time')
        assert hasattr(info, 'qfi_raw_value')


class TestUF:
    """Tests for UF (Floquet Unitary) dataclass."""

    def test_uf_creation(self):
        """Test creating UF dataclass."""
        eigenvalues = mp.matrix([[1], [mp.exp(mp.j * 0.5)]])
        U = mp.eye(2)
        U_inv = mp.eye(2)
        
        uf = UF(eigenvalues=eigenvalues, U=U, U_inv=U_inv)
        
        assert uf.eigenvalues is not None
        assert uf.U is not None
        assert uf.U_inv is not None


class TestReadDefaultsArgsFromConfig:
    """Tests for config file reading."""

    def test_read_config_file_not_exists(self):
        """Test reading from non-existent config file returns defaults."""
        params = read_defaults_args_from_config("/nonexistent/path/config.ini")
        
        # Should return default values
        assert params.N == 20
        assert float(params.J) == 1.0
        assert float(params.B) == 0.4

    def test_read_valid_config_file(self):
        """Test reading from valid config file."""
        config_content = """
[Simulation]
J = 2.0
N = 30
B = 0.5
T = 1.5
phi-kick-phase = pi
h = 0.1
frequency = 3
phi-0 = 0.2
varphi = 0.0
theta = 0.0
num-points = 50
steps-floquet-unitary = 50
dps = 30

[Files]
output-dir = test_results
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(config_content)
            config_file = f.name
        
        try:
            params = read_defaults_args_from_config(config_file)
            
            assert params.N == 30
            assert float(params.J) == 2.0
            assert float(params.B) == 0.5
            assert float(params.T) == 1.5
            assert float(params.h) == 0.1
            assert params.freq == 3
            assert params.run_arguments["dps"] == 30
            assert params.run_arguments["output_dir"] == "test_results"
        finally:
            os.unlink(config_file)

    def test_read_config_with_pi_value(self):
        """Test reading config file with 'pi' as value."""
        config_content = """
[Simulation]
J = 1.0
N = 10
B = 0.4
T = 1.0
phi-kick-phase = pi
h = 0.0
frequency = 2
phi-0 = 0.0
varphi = 0.0
theta = 0.0
num-points = 100
steps-floquet-unitary = 100
dps = 50

[Files]
output-dir = results
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(config_content)
            config_file = f.name
        
        try:
            params = read_defaults_args_from_config(config_file)
            
            # phi_kick_phase should be pi
            assert abs(float(params.phi_kick_phase) - float(mp.pi)) < 1e-10
        finally:
            os.unlink(config_file)
