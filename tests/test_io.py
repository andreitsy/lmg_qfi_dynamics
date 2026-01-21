"""Tests for the IO module."""

import pytest
import tempfile
import os
import pandas as pd
from pathlib import Path

from lmg_qfi import (
    QFIInformation,
    save_to_file_qfi_dynamics,
)


class TestSaveToFileQFIDynamics:
    """Tests for saving QFI dynamics to file."""

    def test_save_single_result(self):
        """Test saving a single QFI result."""
        results = [
            QFIInformation(
                m_x=0.1,
                m_y=0.2,
                m_z=0.3,
                qfi=0.5,
                time=100,
                qfi_raw_value="0.5000000000",
            )
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = Path(f.name)
        
        try:
            save_to_file_qfi_dynamics(results, output_file)
            
            # Read and verify
            df = pd.read_csv(output_file)
            assert len(df) == 1
            assert df["time"].iloc[0] == 100
            assert df["m_x"].iloc[0] == 0.1
            assert df["m_y"].iloc[0] == 0.2
            assert df["m_z"].iloc[0] == 0.3
            assert df["qfi"].iloc[0] == 0.5
            # qfi_raw may be read as float by pandas, so compare the numeric value
            assert abs(float(df["qfi_raw"].iloc[0]) - 0.5) < 1e-10
        finally:
            os.unlink(output_file)

    def test_save_multiple_results(self):
        """Test saving multiple QFI results."""
        results = [
            QFIInformation(m_x=0.1, m_y=0.1, m_z=0.9, qfi=0.1, time=1, qfi_raw_value="0.1"),
            QFIInformation(m_x=0.2, m_y=0.2, m_z=0.8, qfi=0.2, time=10, qfi_raw_value="0.2"),
            QFIInformation(m_x=0.3, m_y=0.3, m_z=0.7, qfi=0.3, time=100, qfi_raw_value="0.3"),
            QFIInformation(m_x=0.4, m_y=0.4, m_z=0.6, qfi=0.4, time=1000, qfi_raw_value="0.4"),
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = Path(f.name)
        
        try:
            save_to_file_qfi_dynamics(results, output_file)
            
            df = pd.read_csv(output_file)
            assert len(df) == 4
            
            # Check all times are correct
            assert list(df["time"]) == [1, 10, 100, 1000]
            
            # Check QFI values
            assert list(df["qfi"]) == [0.1, 0.2, 0.3, 0.4]
        finally:
            os.unlink(output_file)

    def test_save_empty_results(self):
        """Test saving empty results list."""
        results = []
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = Path(f.name)
        
        try:
            save_to_file_qfi_dynamics(results, output_file)
            
            df = pd.read_csv(output_file)
            assert len(df) == 0
            # Check columns exist
            assert "time" in df.columns
            assert "m_x" in df.columns
            assert "m_y" in df.columns
            assert "m_z" in df.columns
            assert "qfi" in df.columns
            assert "qfi_raw" in df.columns
        finally:
            os.unlink(output_file)

    def test_save_with_negative_values(self):
        """Test saving results with negative values."""
        results = [
            QFIInformation(
                m_x=-0.5,
                m_y=-0.3,
                m_z=-0.2,
                qfi=0.8,
                time=50,
                qfi_raw_value="0.8",
            )
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = Path(f.name)
        
        try:
            save_to_file_qfi_dynamics(results, output_file)
            
            df = pd.read_csv(output_file)
            assert df["m_x"].iloc[0] == -0.5
            assert df["m_y"].iloc[0] == -0.3
            assert df["m_z"].iloc[0] == -0.2
        finally:
            os.unlink(output_file)

    def test_save_with_high_precision_raw_value(self):
        """Test saving results with high precision QFI raw value."""
        high_precision_value = "0.123456789012345678901234567890"
        results = [
            QFIInformation(
                m_x=0.0,
                m_y=0.0,
                m_z=1.0,
                qfi=0.123456789,
                time=1,
                qfi_raw_value=high_precision_value,
            )
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = Path(f.name)
        
        try:
            save_to_file_qfi_dynamics(results, output_file)
            
            # Read with dtype to preserve string
            df = pd.read_csv(output_file, dtype={'qfi_raw': str})
            assert df["qfi_raw"].iloc[0] == high_precision_value
        finally:
            os.unlink(output_file)

    def test_csv_columns_order(self):
        """Test that CSV has correct column order."""
        results = [
            QFIInformation(m_x=0.1, m_y=0.2, m_z=0.3, qfi=0.4, time=5, qfi_raw_value="0.4")
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = Path(f.name)
        
        try:
            save_to_file_qfi_dynamics(results, output_file)
            
            df = pd.read_csv(output_file)
            expected_columns = ["time", "m_x", "m_y", "m_z", "qfi", "qfi_raw"]
            assert list(df.columns) == expected_columns
        finally:
            os.unlink(output_file)

    def test_save_creates_file(self):
        """Test that save function creates the file."""
        results = [
            QFIInformation(m_x=0.0, m_y=0.0, m_z=0.0, qfi=0.0, time=1, qfi_raw_value="0")
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.csv"
            
            assert not output_file.exists()
            
            save_to_file_qfi_dynamics(results, output_file)
            
            assert output_file.exists()
