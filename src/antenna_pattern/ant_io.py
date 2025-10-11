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
    
    # Prepare save dictionary
    save_dict = {
        'theta': theta,
        'phi': phi,
        'frequency': frequency,
        'e_theta': e_theta,
        'e_phi': e_phi,
        'metadata': meta_json
    }
    
    # Check if pattern has spherical wave expansion data
    if hasattr(pattern, 'swe') and pattern.swe:
        # Save each frequency's SWE data
        swe_frequencies = list(pattern.swe.keys())
        save_dict['swe_frequencies'] = np.array(swe_frequencies)
        
        for i, freq in enumerate(swe_frequencies):
            swe_data = pattern.swe[freq]
            prefix = f'swe_{i}_'
            
            # Save arrays
            save_dict[f'{prefix}Q_coefficients'] = swe_data['Q_coefficients']
            save_dict[f'{prefix}power_per_n'] = swe_data['power_per_n']
            
            # Save scalar/simple values - convert numpy types to Python types
            swe_meta = {
                'M': int(swe_data['M']),
                'N': int(swe_data['N']),
                'frequency': float(swe_data['frequency']),
                'wavelength': float(swe_data['wavelength']),
                'radius': float(swe_data['radius']),
                'mode_power': float(swe_data['mode_power']),
                'k': float(swe_data['k'])
            }
            
            # Add optional adaptive parameters if present
            for key in ['N_full', 'M_full', 'converged', 'iterations', 'power_retained_fraction']:
                if key in swe_data:
                    value = swe_data[key]
                    # Convert numpy types to native Python types
                    if isinstance(value, (np.integer, np.int64, np.int32)):
                        swe_meta[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64, np.float32)):
                        swe_meta[key] = float(value)
                    elif isinstance(value, np.bool_):
                        swe_meta[key] = bool(value)
                    else:
                        swe_meta[key] = value
            
            save_dict[f'{prefix}metadata'] = json.dumps(swe_meta)
        
        logger.info(f"Saving SWE data for {len(swe_frequencies)} frequency(ies)")
    
    # Save data to NPZ file
    np.savez_compressed(file_path, **save_dict)
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
        
        # Check if file contains SWE data
        if 'swe_frequencies' in data:
            swe_frequencies = data['swe_frequencies']
            pattern.swe = {}
            
            for i, freq in enumerate(swe_frequencies):
                prefix = f'swe_{i}_'
                
                # Load arrays
                Q_coefficients = data[f'{prefix}Q_coefficients']
                power_per_n = data[f'{prefix}power_per_n']
                
                # Load metadata
                swe_meta_json = str(data[f'{prefix}metadata'])
                swe_meta = json.loads(swe_meta_json)
                
                # Reconstruct SWE dictionary
                swe_dict = {
                    'Q_coefficients': Q_coefficients,
                    'power_per_n': power_per_n,
                    'M': swe_meta['M'],
                    'N': swe_meta['N'],
                    'frequency': swe_meta['frequency'],
                    'wavelength': swe_meta['wavelength'],
                    'radius': swe_meta['radius'],
                    'mode_power': swe_meta['mode_power'],
                    'k': swe_meta['k']
                }
                
                # Add optional adaptive parameters if present
                for key in ['N_full', 'M_full', 'converged', 'iterations', 'power_retained_fraction']:
                    if key in swe_meta:
                        swe_dict[key] = swe_meta[key]
                
                pattern.swe[freq] = swe_dict
            
            logger.info(f"Loaded SWE data for {len(swe_frequencies)} frequency(ies)")
        
        logger.info(f"Pattern loaded from {file_path}")
        return pattern, metadata
    
def write_cut(pattern, file_path: Union[str, Path], polarization_format: int = 1) -> None:
    """
    Write an antenna pattern to GRASP CUT format.
    
    Args:
        pattern: AntennaPattern object to save
        file_path: Path to save the file to
        polarization_format: Output polarization format:
            1 = theta/phi (spherical)
            2 = RHCP/LHCP (circular)
            3 = X/Y (Ludwig-3 linear)
            
    Raises:
        OSError: If file cannot be written
        ValueError: If polarization_format is invalid
    """
    file_path = Path(file_path)
    
    # Ensure .cut extension
    if file_path.suffix.lower() != '.cut':
        file_path = file_path.with_suffix('.cut')
    
    if polarization_format not in [1, 2, 3]:
        raise ValueError("polarization_format must be 1 (theta/phi), 2 (RHCP/LHCP), or 3 (X/Y)")
    
    # Get pattern data
    theta = pattern.theta_angles
    phi = pattern.phi_angles
    frequencies = pattern.frequencies
    e_theta = pattern.data.e_theta.values
    e_phi = pattern.data.e_phi.values
    
    with open(file_path, 'w') as f:
        # Write data for each frequency
        for freq_idx, freq in enumerate(frequencies):
            # Write data for each phi cut
            for phi_idx, phi_val in enumerate(phi):
                # Write text description line for this cut
                f.write(f"{freq/1e6:.3f} MHz, Phi = {phi_val:.1f} deg\n")
                
                # Write cut header: theta_start, theta_step, num_theta, phi, icomp, icut, ncomp
                theta_start = theta[0]
                theta_step = theta[1] - theta[0] if len(theta) > 1 else 1.0
                num_theta = len(theta)
                icut = 1  # Standard polar cut (phi fixed, theta varying)
                ncomp = 2  # Two field components
                
                f.write(f"{theta_start:.2f} {theta_step:.6f} {num_theta} {phi_val:.2f} {polarization_format} {icut} {ncomp}\n")
                
                # Convert field components based on polarization format
                for theta_idx in range(len(theta)):
                    if polarization_format == 1:
                        # Theta/phi format
                        comp1 = e_theta[freq_idx, theta_idx, phi_idx]
                        comp2 = e_phi[freq_idx, theta_idx, phi_idx]
                        
                    elif polarization_format == 2:
                        # RHCP/LHCP format
                        from .polarization import polarization_tp2rl
                        e_r, e_l = polarization_tp2rl(
                            phi_val,
                            e_theta[freq_idx, theta_idx, phi_idx:phi_idx+1],
                            e_phi[freq_idx, theta_idx, phi_idx:phi_idx+1]
                        )
                        comp1 = e_r[0]  # RHCP
                        comp2 = e_l[0]  # LHCP
                        
                    elif polarization_format == 3:
                        # X/Y format
                        from .polarization import polarization_tp2xy
                        e_x, e_y = polarization_tp2xy(
                            phi_val,
                            e_theta[freq_idx, theta_idx, phi_idx:phi_idx+1],
                            e_phi[freq_idx, theta_idx, phi_idx:phi_idx+1]
                        )
                        comp1 = e_x[0]  # X component
                        comp2 = e_y[0]  # Y component
                    
                    # Write complex components
                    f.write(f"{comp1.real:.6e} {comp1.imag:.6e} {comp2.real:.6e} {comp2.imag:.6e}\n")


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
    import logging
    logger = logging.getLogger(__name__)
    from .polarization import polarization_rl2tp, polarization_xy2tp
    
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
                    
                # This should be a header line (numeric data)
                header_parts = lines[line_index].strip().split()
                if len(header_parts) >= 7:  # Updated from 5 to 7
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
    header_flag = True
    first_flag = True
    data_counter = 0
    line_data_a = []
    line_data_b = []
    
    while line_index < total_lines:
        data_str = lines[line_index]
        line_index += 1
        
        if "MHz" in data_str:
            # This is a description line for a new cut
            header_flag = True
            continue
            
        if header_flag:
            # Parse header efficiently
            header_parts = data_str.strip().split()
            if len(header_parts) < 7:  # Updated from 5 to 7
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
                icut = int(header_parts[5])   # ICUT parameter (should be 1)
                ncomp = int(header_parts[6])  # NCOMP parameter (should be 2)
                
                if icomp not in [1, 2, 3]:
                    raise ValueError(f"Invalid polarization format (ICOMP): {icomp}")
                if icut != 1:
                    logger.warning(f"Unexpected ICUT value: {icut}. Expected 1 (standard polar cut)")
                if ncomp != 2:
                    logger.warning(f"Unexpected NCOMP value: {ncomp}. Expected 2 (two field components)")
                    
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

    theta_start, theta_stop, theta_points = map(float, theta_info[:3])
    theta_points = np.round(theta_points).astype(int)
    phi_start, phi_stop, phi_points = map(float, phi_info[:3])
    phi_points = np.round(phi_points).astype(int)
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
    
    
    for freq_idx in range(len(frequency_np)):
        # Process each theta/phi combination for this frequency
        for theta_idx in range(len(theta)):
            for phi_idx in range(len(phi)):
                # Reversed indexing formula
                data_idx = theta_idx * len(phi) + phi_idx
                
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

def write_ffd(pattern, file_path: Union[str, Path]) -> None:
    """
    Write an antenna pattern to HFSS far field data format (.ffd).
    
    Args:
        pattern: AntennaPattern object to save
        file_path: Path to save the file to
        
    Raises:
        OSError: If file cannot be written
    """
    file_path = Path(file_path)
    
    # Ensure .ffd extension
    if file_path.suffix.lower() != '.ffd':
        file_path = file_path.with_suffix('.ffd')
    
    # Get pattern data
    theta = pattern.theta_angles
    phi = pattern.phi_angles
    frequencies = pattern.frequencies
    e_theta = pattern.data.e_theta.values
    e_phi = pattern.data.e_phi.values
    
    with open(file_path, 'w') as f:
        # Write header lines
        f.write(f"{theta[0]} {theta[-1]} {len(theta)}\n")
        f.write(f"{phi[0]} {phi[-1]} {len(phi)}\n")
        f.write(f"Freq {len(frequencies)}\n")
        
        # Write data for each frequency
        for freq_idx, freq in enumerate(frequencies):
            f.write(f"Frequency {freq}\n")
            
            # Write field data for all theta/phi combinations
            # FFD format: theta is outer loop, phi is inner loop (opposite of what I had)
            for theta_idx in range(len(theta)):
                for phi_idx in range(len(phi)):
                    # Convert to HFSS units (multiply by sqrt(60))
                    eth = e_theta[freq_idx, theta_idx, phi_idx] * np.sqrt(60)
                    eph = e_phi[freq_idx, theta_idx, phi_idx] * np.sqrt(60)
                    
                    f.write(f"{eth.real:.6e} {eth.imag:.6e} {eph.real:.6e} {eph.imag:.6e}\n")

def create_pattern_from_swe(swe_data: Dict[str, Any],
                           theta_angles: Optional[np.ndarray] = None,
                           phi_angles: Optional[np.ndarray] = None) -> 'AntennaPattern':
    """
    Create a far field AntennaPattern from spherical mode coefficients.
    
    Args:
        swe_data: Dictionary from calculate_spherical_modes or load_swe_coefficients
        theta_angles: Theta angles in degrees (default: 0 to 180, 1° steps, sided convention)
        phi_angles: Phi angles in degrees (default: 0 to 360, 5° steps)
        
    Returns:
        AntennaPattern object with reconstructed far field
        
    Notes:
        Uses optimized far-field evaluation (no radial distance normalization).
        The pattern represents the far-field angular distribution.
        Theta angles must be in sided convention [0°, 180°] to match Hansen's
        spherical harmonic definitions.
    """
    from .spherical_expansion import evaluate_farfield_from_modes
    
    # Default angles - use SIDED convention (Hansen standard)
    if theta_angles is None:
        NTHE = swe_data.get('NTHE', 360)  # Default to 1° if not present
        angular_spacing_deg = 360.0 / NTHE
        n_theta_points = int(np.round(180.0 / angular_spacing_deg)) + 1
        theta_angles = np.linspace(0, 180, n_theta_points)
    if phi_angles is None:
        phi_angles = np.arange(0, 361, 5.0)
    
    # Create meshgrid
    theta_rad = np.radians(theta_angles)
    phi_rad = np.radians(phi_angles)
    THETA, PHI = np.meshgrid(theta_rad, phi_rad, indexing='ij')
    
    # Evaluate far field using optimized function
    E_theta, E_phi = evaluate_farfield_from_modes(
        swe_data['Q_coefficients'],
        swe_data['M'],
        swe_data['N'],
        swe_data['k'],
        THETA,
        PHI
    )
    
    # Create pattern (single frequency)
    frequencies = np.array([swe_data['frequency']])
    e_theta = E_theta[np.newaxis, :, :]  # Add frequency dimension
    e_phi = E_phi[np.newaxis, :, :]
    
    pattern = AntennaPattern(
        theta=theta_angles,
        phi=phi_angles,
        frequency=frequencies,
        e_theta=e_theta,
        e_phi=e_phi,
        polarization='theta'
    )
    
    # Attach SWE data
    pattern.swe = {swe_data['frequency']: swe_data}
    
    logger.info(f"Pattern created from SWE coefficients at f={swe_data['frequency']/1e9:.3f} GHz")
    return pattern

def write_ticra_sph(swe_data: Dict[str, Any], file_path: Union[str, Path],
                    program_tag: str = "AntPy", id_string: str = "SWE Export") -> None:
    """
    Write spherical mode coefficients to TICRA .sph format.
    
    Note: Phase convention conversion is now handled during extraction,
    so Q_coefficients are already in Hansen convention.
    """
    import datetime
    
    file_path = Path(file_path)
    
    # Ensure .sph extension
    if file_path.suffix.lower() != '.sph':
        file_path = file_path.with_suffix('.sph')
    
    # Extract data
    Q_coefficients = swe_data['Q_coefficients']  # Already in Hansen convention
    M = swe_data['M']
    N = swe_data['N']
    
    # Normalization factor for TICRA format: Q' = (1/√8π) * Q*
    norm_factor = 1.0 / np.sqrt(8.0 * np.pi)
    
    # Calculate NTHE and NPHI based on actual sampling grid
    # NTHE: number of theta samples over 360° (must be even)
    n_theta_samples = swe_data['n_theta_samples'] 
    # If theta goes 0-180° (half sphere), double it for 360°
    NTHE = 2 * (n_theta_samples - 1)
    # Ensure even
    if NTHE % 2 != 0:
        NTHE += 1

    # NPHI: number of phi samples over 360° (must be >= 3)
    n_phi_samples = swe_data['n_phi_samples']
    NPHI = max(3, 2*n_phi_samples)
    
    # Write file
    with open(file_path, 'w') as f:
        # Record 1: PRGTAG
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{program_tag} - {timestamp}\n")
        
        # Record 2: IDSTRG
        f.write(f"{id_string}\n")
        
        # Record 3: NTHE, NPHI, NMAX, MMAX
        NMAX = N
        MMAX = M
        f.write(f"{NTHE:6d}{NPHI:6d}{NMAX:6d}{MMAX:6d}\n")
        
        # Records 4-8: Dummy data
        f.write("Dummy text 1\n")
        f.write("     0.00000     0.00000     0.00000     0.00000     0.00000\n")
        f.write("     0.00000     0.00000     0.00000     0.00000     0.00000\n")
        f.write("Dummy text 2\n")
        f.write("Dummy text 3\n")
        
        # Mode coefficients - organized by |m|
        for m_abs in range(0, M + 1):
            # Calculate power in this azimuthal mode
            power_m = 0.0
            
            if m_abs == 0:
                # For m=0, only include m=0 coefficients
                for n in range(1, N + 1):
                    n_idx = n - 1
                    m_idx = M  # m=0 is at index M
                    for s_idx in range(2):
                        power_m += np.abs(Q_coefficients[s_idx, m_idx, n_idx])**2
            else:
                # For m≠0, include both +m and -m
                m_pos_idx = m_abs + M
                m_neg_idx = -m_abs + M
                for n in range(m_abs, N + 1):
                    n_idx = n - 1
                    for s_idx in range(2):
                        power_m += np.abs(Q_coefficients[s_idx, m_neg_idx, n_idx])**2
                        power_m += np.abs(Q_coefficients[s_idx, m_pos_idx, n_idx])**2
            
            # Record m.1: M, POWERM
            f.write(f"{m_abs:6d}  {power_m:14.6e}\n")
            
            # Record m.2: Coefficients
            if m_abs == 0:
                # For m=0: write Q1,0,n and Q2,0,n for n=1 to N
                m_idx = M
                for n in range(1, N + 1):
                    n_idx = n - 1
                    
                    Q1_0n = Q_coefficients[0, m_idx, n_idx]  # s=1 (TE)
                    Q2_0n = Q_coefficients[1, m_idx, n_idx]  # s=2 (TM)
                    
                    # Apply TICRA normalization and conjugation ONLY
                    # No phase correction needed - already in Hansen convention
                    Q1_file = Q1_0n.conjugate() * norm_factor
                    Q2_file = Q2_0n.conjugate() * norm_factor
                    
                    f.write(f"  {Q1_file.real:14.6e}  {Q1_file.imag:14.6e}  "
                        f"{Q2_file.real:14.6e}  {Q2_file.imag:14.6e}\n")
            else:
                # For m≠0: write Q(s,-m,n) then Q(s,+m,n) for each n
                m_neg_idx = -m_abs + M
                m_pos_idx = m_abs + M
                
                for n in range(m_abs, N + 1):
                    n_idx = n - 1
                    
                    # Get coefficients (already in Hansen convention)
                    Q1_neg = Q_coefficients[0, m_neg_idx, n_idx]  # s=1, -m, n (TE)
                    Q2_neg = Q_coefficients[1, m_neg_idx, n_idx]  # s=2, -m, n (TM)
                    Q1_pos = Q_coefficients[0, m_pos_idx, n_idx]  # s=1, +m, n (TE)
                    Q2_pos = Q_coefficients[1, m_pos_idx, n_idx]  # s=2, +m, n (TM)
                    
                    # Apply TICRA normalization and conjugation ONLY
                    Q1_neg_file = Q1_neg.conjugate() * norm_factor
                    Q2_neg_file = Q2_neg.conjugate() * norm_factor
                    Q1_pos_file = Q1_pos.conjugate() * norm_factor
                    Q2_pos_file = Q2_pos.conjugate() * norm_factor
                    
                    # Write -m coefficients
                    f.write(f"  {Q1_neg_file.real:14.6e}  {Q1_neg_file.imag:14.6e}  "
                        f"{Q2_neg_file.real:14.6e}  {Q2_neg_file.imag:14.6e}\n")
                    
                    # Write +m coefficients
                    f.write(f"  {Q1_pos_file.real:14.6e}  {Q1_pos_file.imag:14.6e}  "
                        f"{Q2_pos_file.real:14.6e}  {Q2_pos_file.imag:14.6e}\n")
    
    logger.info(f"SWE coefficients exported to TICRA format: {file_path}")

def read_ticra_sph(file_path: Union[str, Path], frequency: float) -> Dict[str, Any]:
    """
    Read spherical mode coefficients from TICRA .sph format.
    """
    from .utilities import frequency_to_wavelength
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Reading TICRA .sph file: {file_path}")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    prgtag = lines[0].strip()
    idstrg = lines[1].strip()
    header_values = lines[2].split()
    NTHE = int(header_values[0])
    NPHI = int(header_values[1])
    N = int(header_values[2])  # NMAX
    M = int(header_values[3])  # MMAX
    
    logger.info(f"  N={N}, M={M}, NTHE={NTHE}, NPHI={NPHI}")
    
    # Calculate wavelength
    wavelength = frequency_to_wavelength(frequency)
    k = 2 * np.pi / wavelength
    radius = N / k  # Rough estimate
    
    # Initialize: [s, m_index, n_index]
    # Note: n ranges from 1 to N, so we need N slots (indices 0 to N-1)
    Q_coefficients = np.zeros((2, 2*M + 1, N), dtype=complex)
    
    # Normalization
    norm_factor = np.sqrt(8.0 * np.pi)
    
    # Parse coefficients starting at line 9
    line_idx = 8
    
    for m_abs in range(0, M + 1):
        if line_idx >= len(lines):
            raise ValueError(f"Unexpected end of file at line {line_idx + 1}")
        
        mode_header = lines[line_idx].split()
        m_file = int(mode_header[0])
        
        if m_file != m_abs:
            raise ValueError(f"Expected |m|={m_abs}, got {m_file}")
        
        line_idx += 1
        
        if m_abs == 0:
            # m=0: read Q(s,0,n) for n=1..N
            m_idx = M
            
            for n in range(1, N + 1):
                if line_idx >= len(lines):
                    raise ValueError(f"Unexpected end of file at line {line_idx + 1}")
                
                n_idx = n - 1  # Array index for n
                
                if n_idx >= N:
                    raise ValueError(f"n_idx={n_idx} out of bounds (N={N})")
                
                values = lines[line_idx].split()
                Q1_re, Q1_im, Q2_re, Q2_im = map(float, values[:4])
                
                Q_coefficients[0, m_idx, n_idx] = norm_factor * complex(Q1_re, Q1_im)
                Q_coefficients[1, m_idx, n_idx] = norm_factor * complex(Q2_re, Q2_im)
                
                line_idx += 1
        else:
            # m≠0: read Q(s,-m,n) and Q(s,+m,n) for n=|m|..N
            m_neg_idx = M - m_abs
            m_pos_idx = M + m_abs
            
            # Check indices are valid
            if m_neg_idx < 0 or m_pos_idx >= 2*M + 1:
                raise ValueError(f"Invalid m indices: m_abs={m_abs}, M={M}")
            
            for n in range(m_abs, N + 1):
                if line_idx >= len(lines):
                    raise ValueError(f"Unexpected end of file at line {line_idx + 1}")
                
                n_idx = n - 1
                
                if n_idx >= N:
                    raise ValueError(f"n_idx={n_idx} out of bounds (N={N})")
                
                # Read -m coefficients
                values = lines[line_idx].split()
                Q1_neg_re, Q1_neg_im, Q2_neg_re, Q2_neg_im = map(float, values[:4])
                line_idx += 1
                
                # Read +m coefficients
                if line_idx >= len(lines):
                    raise ValueError(f"Unexpected end of file at line {line_idx + 1}")
                
                values = lines[line_idx].split()
                Q1_pos_re, Q1_pos_im, Q2_pos_re, Q2_pos_im = map(float, values[:4])
                line_idx += 1
                
                # Store with conversions
                Q_coefficients[0, m_neg_idx, n_idx] = norm_factor * complex(Q1_neg_re, Q1_neg_im)
                Q_coefficients[1, m_neg_idx, n_idx] = norm_factor * complex(Q2_neg_re, Q2_neg_im)
                Q_coefficients[0, m_pos_idx, n_idx] = norm_factor * complex(Q1_pos_re, Q1_pos_im)
                Q_coefficients[1, m_pos_idx, n_idx] = norm_factor * complex(Q2_pos_re, Q2_pos_im)

    # Apply Hansen normalization: TICRA coefficients lack the √[2/(n(n+1))] factor
    # that is embedded in Hansen's spherical wave function definitions
    logger.info("Applying Hansen normalization to coefficients...")
    for n in range(1, N + 1):
        n_idx = n - 1
        hansen_norm = np.sqrt(2.0 / (n * (n + 1)))
        
        for s in [0, 1]:  # s=1,2 → index 0,1
            m_min = max(-n, -M)
            m_max = min(n, M)
            for m in range(m_min, m_max + 1):
                m_idx = m + M
                Q_coefficients[s, m_idx, n_idx] *= hansen_norm
    
    logger.info(f"Successfully read coefficients: shape={Q_coefficients.shape}")

    # In read_ticra_sph, after parsing all coefficients:
    total_power = 0.5 * np.sum(np.abs(Q_coefficients)**2)
    logger.info(f"\n=== POWER CHECK ===")
    logger.info(f"Total radiated power from Q coefficients: {total_power:.6e} W")
    logger.info(f"This should equal 0.5 for normalized coefficients")
    logger.info(f"  or the actual radiated power if in physical units")

    
    logger.info(f"Successfully read coefficients: shape={Q_coefficients.shape}")
    
    return {
        'Q_coefficients': Q_coefficients,
        'M': M,
        'N': N,
        'frequency': frequency,
        'wavelength': wavelength,
        'k': k,
        'radius': radius,
        'NTHE': NTHE,
        'NPHI': NPHI,
        'n_theta_samples': NTHE // 2 + 1,
        'n_phi_samples': NPHI,
        'mode_power': 0.5 * np.sum(np.abs(Q_coefficients)**2)
    }