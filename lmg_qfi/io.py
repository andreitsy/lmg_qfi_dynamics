"""File I/O functions for saving simulation results."""

import pandas as pd

from pathlib import Path
from typing import List

from .config import QFIInformation


def save_to_file_qfi_dynamics(
        results: List[QFIInformation],
        output_file: Path,
):
    """
    Save QFI dynamics results to a CSV file.
    
    Parameters
    ----------
    results : list
        List of QFIInformation objects.
    output_file : Path
        Path to the output CSV file.
    """
    csv_data = {
        "time": [],
        "m_x": [],
        "m_y": [],
        "m_z": [],
        "qfi": [],
        "qfi_raw": [],
    }
    for result in results:
        csv_data["time"].append(result.time)
        csv_data["m_x"].append(result.m_x)
        csv_data["m_y"].append(result.m_y)
        csv_data["m_z"].append(result.m_z)
        csv_data["qfi"].append(result.qfi)
        csv_data["qfi_raw"].append(result.qfi_raw_value)

    df = pd.DataFrame(csv_data)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
