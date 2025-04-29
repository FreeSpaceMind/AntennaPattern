"""
File input/output functions for antenna radiation patterns.
"""

import logging
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Union, Any, Tuple

from .pattern import AntennaPattern
from .polarization import polarization_rl2tp, polarization_xy2tp

# Configure logging
logger = logging.getLogger(__name__)


def save_pattern_npz(pattern, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save an antenna pattern to NPZ format for efficient loading.
    
    Args:
        pattern: AntennaPattern object to save
        file_path: Path to save the file to
        metadata: Optional metadata to include
        
    Raises:
        OSError: If file cannot be written
    """
    file_path = Path(file_path)
    
    # Ensure .npz extension
    if file_path.suffix.lower() != '.npz':
        file_path = file_path.with_suffix('.npz')
    
    # Extract data from pattern
    theta = pattern.theta_angles
    phi = pattern.phi_angles
    frequency = pattern.frequencies
    e_theta = pattern.data.e_theta.values
    e_phi = pattern.data.e_phi.values
    
    # Create metadata dictionary
    meta_dict = {
        'polarization': pattern.polarization,
        'version': '1.0',
        'format': 'AntPy Pattern NPZ'
    }
    
    # Add additional metadata if provided
    if metadata:
        meta_dict.update(metadata)
    
    # Convert metadata to JSON string
    meta_json = json.dumps(meta_dict)
    
    # Save data to NPZ file
    np.savez_compressed(
        file_path,
        theta=theta,
        phi=phi,
        frequency=frequency,
        e_theta=e_theta,
        e_phi=e_phi,
        metadata=meta_json
    )
    logger.info(f"Pattern saved to {file_path}")


def load_pattern_npz(file_path: Union[str, Path]) -> Tuple:
    """
    Load an antenna pattern from NPZ format.
    
    Args:
        file_path: Path to the NPZ file
        
    Returns:
        Tuple containing (pattern, metadata)
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is invalid
    """
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Pattern file not found: {file_path}")
    
    # Load data from NPZ file
    with np.load(file_path, allow_pickle=False) as data:
        theta = data['theta']
        phi = data['phi']
        frequency = data['frequency']
        e_theta = data['e_theta']
        e_phi = data['e_phi']
        
        # Load metadata
        metadata_json = str(data['metadata'])
        metadata = json.loads(metadata_json)
        
        polarization = metadata.get('polarization')
        
        # Create AntennaPattern object
        pattern = AntennaPattern(
            theta=theta,
            phi=phi,
            frequency=frequency,
            e_theta=e_theta,
            e_phi=e_phi,
            polarization=polarization
        )
        
        logger.info(f"Pattern loaded from {file_path}")
        return pattern, metadata


def read_cut(file_path: Union[str, Path], frequency_start: float, frequency_end: float):
    """
    Read an antenna CUT file and store it in an AntennaPattern.
    
    Optimized version with faster file reading and data processing.
    
    Args:
        file_path: Path to the CUT file
        frequency_start: Frequency of first pattern in Hz
        frequency_end: Frequency of last pattern in Hz
        
    Returns:
        AntennaPattern: The imported antenna pattern
    """
    
    # Validate inputs
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CUT file not found: {file_path}")
    
    if frequency_start <= 0 or frequency_end <= 0:
        raise ValueError("Frequencies must be positive")
    if frequency_start > frequency_end:
        raise ValueError("frequency_start must be less than or equal to frequency_end")
    
    # Read entire file at once for faster processing
    with open(file_path, "r") as reader:
        lines = reader.readlines()
    
    total_lines = len(lines)
    line_index = 0
    
    # Use lists with pre-estimated size for better performance
    estimated_phi_count = 36  # Typical number of phi cuts
    estimated_freq_count = 5  # Typical number of frequencies
    estimated_theta_count = 181  # Typical number of theta points
    
    phi = []
    ya_data = []
    yb_data = []
    
    # Use NumPy arrays to store theta array once
    theta = None
    icomp = None
    
    # Preallocate header info
    theta_start = 0
    theta_increment = 0
    theta_length = 0
    
    # Scan file structure to determine pattern dimensions
    # First pass to determine dimensions
    first_pass = True
    phi_values = set()
    header_count = 0
    
    if first_pass:
        while line_index < min(1000, total_lines):  # Just scan the first part of the file
            if "MHz" in lines[line_index]:
                line_index += 1
                if line_index >= total_lines:
                    break
                    
                # This should be a header line
                header_parts = lines[line_index].strip().split()
                if len(header_parts) >= 5:
                    header_count += 1
                    phi_values.add(float(header_parts[3]))
                
            line_index += 1
        
        # Reset line index for main parsing
        line_index = 0
        first_pass = False
        
        # Estimate dimensions more accurately
        estimated_phi_count = len(phi_values)
        
        if header_count > 0:
            estimated_freq_count = header_count // estimated_phi_count
    
    # Main parsing - optimized for speed
    title_flag = False
    header_flag = True
    first_flag = True
    data_counter = 0
    line_data_a = []
    line_data_b = []
    
    while line_index < total_lines:
        # Skip initial line
        if line_index == 0:
            line_index += 1
            continue
            
        data_str = lines[line_index]
        line_index += 1
        
        if "MHz" in data_str:
            # New frequency section begins
            header_flag = True
            continue
            
        if header_flag:
            # Parse header efficiently
            header_parts = data_str.strip().split()
            if len(header_parts) < 5:
                continue
                
            theta_length = int(header_parts[2])
            phi.append(float(header_parts[3]))
            
            if first_flag:
                theta_start = float(header_parts[0])
                theta_increment = float(header_parts[1])
                theta = np.linspace(
                    theta_start,
                    theta_start + (theta_length - 1) * theta_increment,
                    theta_length
                )
                icomp = int(header_parts[4])
                if icomp not in [1, 2, 3]:
                    raise ValueError(f"Invalid polarization format (ICOMP): {icomp}")
                first_flag = False
            
            # Preallocate data arrays for this section
            line_data_a = []
            line_data_b = []
            
            data_counter = theta_length
            header_flag = False
        else:
            parts = np.fromstring(data_str, dtype=float, sep=" ")
            if len(parts) >= 4:
                line_data_a.append(complex(parts[0], parts[1]))
                line_data_b.append(complex(parts[2], parts[3]))
                data_counter -= 1
                
                if data_counter == 0:
                    header_flag = True
                    ya_data.append(line_data_a)
                    yb_data.append(line_data_b)

    # Consistency checks
    if len(ya_data) == 0 or len(yb_data) == 0:
        raise ValueError("No valid data found in CUT file")
    
    # Make frequency vector more accurately
    phi_array = np.array(phi)
    unique_phi = np.sort(np.unique(phi_array))
    freq_num = len(ya_data) // len(unique_phi)
    
    if freq_num <= 0:
        raise ValueError(f"Invalid frequency count: {freq_num}")
    
    frequency = np.linspace(frequency_start, frequency_end, freq_num)
    
    # Convert to numpy arrays efficiently - specify dtype for better performance
    ya_np = np.array(ya_data, dtype=complex)
    yb_np = np.array(yb_data, dtype=complex)
    
    # Determine the correct shape
    num_theta = len(theta)
    num_phi = len(unique_phi)
    
    # Use optimized reshape approach with direct indexing
    e_theta = np.zeros((freq_num, num_theta, num_phi), dtype=complex)
    e_phi = np.zeros((freq_num, num_theta, num_phi), dtype=complex)
    
    # Custom reshape logic for the specific data structure
    for i in range(len(ya_data)):
        freq_idx = i // num_phi
        phi_idx = i % num_phi
        
        if freq_idx < freq_num and phi_idx < num_phi:
            e_theta[freq_idx, :, phi_idx] = ya_data[i]
            e_phi[freq_idx, :, phi_idx] = yb_data[i]
    
    # Convert polarizations based on icomp
    if icomp == 1:
        # Polarization is theta, phi - already in right form
        pass
    elif icomp == 2:
        # Polarization is right and left - vectorized conversion
        for phi_idx, phi_val in enumerate(unique_phi):
            theta_slice, phi_slice = polarization_rl2tp(
                phi_val, 
                e_theta[:, :, phi_idx], 
                e_phi[:, :, phi_idx]
            )
            e_theta[:, :, phi_idx] = theta_slice
            e_phi[:, :, phi_idx] = phi_slice
    elif icomp == 3:
        # Polarization is linear co and cross (x and y) - vectorized conversion
        for phi_idx, phi_val in enumerate(unique_phi):
            theta_slice, phi_slice = polarization_xy2tp(
                phi_val, 
                e_theta[:, :, phi_idx], 
                e_phi[:, :, phi_idx]
            )
            e_theta[:, :, phi_idx] = theta_slice
            e_phi[:, :, phi_idx] = phi_slice
            
    # Create AntennaPattern with results
    return AntennaPattern(
        theta=theta,
        phi=unique_phi,
        frequency=frequency,
        e_theta=e_theta,
        e_phi=e_phi
    )


def read_ffd(file_path: Union[str, Path]):
    """
    Read a far field data file from HFSS.
    
    Args:
        file_path: Path to the FFD file
        
    Returns:
        AntennaPattern: The imported antenna pattern
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is not a valid FFD file
    """
    
    # Validate input
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"FFD file not found: {file_path}")
    
    # Read a far field data file, format ffd
    with open(file_path, "r") as file_handle:
        lines = file_handle.readlines()

    # Read theta, phi, and frequency information
    if len(lines) < 3:
        raise ValueError("FFD file is too short")
        
    theta_info = lines[0].strip().split()
    phi_info = lines[1].strip().split()
    freq_info = lines[2].strip().split()

    if len(theta_info) < 3 or len(phi_info) < 3 or len(freq_info) < 2:
        raise ValueError("Invalid FFD file header format")

    theta_start, theta_stop, theta_points = map(int, theta_info[:3])
    phi_start, phi_stop, phi_points = map(int, phi_info[:3])
    num_frequencies = int(freq_info[1])

    theta = np.linspace(theta_start, theta_stop, theta_points)
    phi = np.linspace(phi_start, phi_stop, phi_points)

    # Initialize storage lists
    frequency_list = []
    e_theta_list = []
    e_phi_list = []

    # Read file
    index = 3
    for freq_idx in range(num_frequencies):
        if index >= len(lines):
            raise ValueError(f"Unexpected end of file at frequency {freq_idx+1}")
            
        freq_line = lines[index].strip().split()
        if len(freq_line) < 2:
            raise ValueError(f"Invalid frequency line: {lines[index].strip()}")
            
        frequency = float(freq_line[1])
        e_theta = []
        e_phi = []

        index += 1
        for _ in range(int(theta_points) * int(phi_points)):
            if index >= len(lines):
                raise ValueError(f"Unexpected end of file at frequency {freq_idx+1}")
                
            radiation_line = list(map(float, lines[index].strip().split()))
            if len(radiation_line) < 4:
                raise ValueError(f"Invalid radiation line: {lines[index].strip()}")
                
            e_th = radiation_line[0] + 1j * radiation_line[1]
            e_ph = radiation_line[2] + 1j * radiation_line[3]
            
            # Convert from HFSS units to standard field units
            e_theta.append(e_th / np.sqrt(60))
            e_phi.append(e_ph / np.sqrt(60))
            index += 1

        # Append into storage lists
        frequency_list.append(frequency)
        e_theta_list.append(e_theta)
        e_phi_list.append(e_phi)

    # Consistency checks
    if len(frequency_list) == 0:
        raise ValueError("No frequency data found in FFD file")
        
    # Convert to numpy
    frequency_np = np.array(frequency_list)
    e_theta_np = np.array(e_theta_list)
    e_phi_np = np.array(e_phi_list)

    # Reshape into 3D (freq, theta, phi) format
    e_theta_final = np.zeros((len(frequency_np), len(theta), len(phi)), dtype=complex)
    e_phi_final = np.zeros((len(frequency_np), len(theta), len(phi)), dtype=complex)
    
    # FFD file format may store data in a different order
    pattern_points = len(theta) * len(phi)
    
    # For each frequency
    for freq_idx in range(len(frequency_np)):
        # Process each theta/phi combination for this frequency
        for phi_idx in range(len(phi)):
            for theta_idx in range(len(theta)):
                data_idx = phi_idx * len(theta) + theta_idx
                
                if data_idx < len(e_theta_np[freq_idx]):
                    e_theta_final[freq_idx, theta_idx, phi_idx] = e_theta_np[freq_idx][data_idx]
                    e_phi_final[freq_idx, theta_idx, phi_idx] = e_phi_np[freq_idx][data_idx]

    # Create AntennaPattern - polarization will be auto-detected
    return AntennaPattern(
        theta=theta,
        phi=phi,
        frequency=frequency_np,
        e_theta=e_theta_final,
        e_phi=e_phi_final
    )