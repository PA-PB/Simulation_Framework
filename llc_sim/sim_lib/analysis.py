import numpy as np
import warnings 
from typing import Dict, Any, Optional

def compute_dead_time(time: np.ndarray, iDS: np.ndarray, fsw: float, 
                     threshold: float = 0.001) -> float:
    """
    Compute dead time before switch turn-on in the last switching cycle.
    
    Uses a robust method that works backwards from the simulation end time
    to find the start of the last cycle, avoiding assumptions about when
    the simulation started.
    
    Parameters
    ----------
    time : np.ndarray
        Simulation time array
    iDS : np.ndarray
        Switch current array
    fsw : float
        Switching frequency in Hz
    threshold : float, optional
        Current threshold to detect switch turn-on (A), default 0.001
        
    Returns
    -------
    float
        Dead time in seconds, or np.nan if computation fails
    """
    try:
        if len(time) < 2 or len(iDS) < 2:
            warnings.warn("Insufficient data points for dead time calculation", RuntimeWarning)
            return np.nan
        
        # Find start of last cycle by working backwards from end time
        # This is more robust than assuming simulation starts at t=0
        last_cycle_start_time = time[-1] - (1.0 / fsw)
        
        # Validate we have data in the last cycle
        if last_cycle_start_time < time[0]:
            warnings.warn(
                f"Last cycle start time ({last_cycle_start_time:.6f}s) is before "
                f"simulation start ({time[0]:.6f}s). Using full simulation.",
                RuntimeWarning
            )
            last_cycle_start_time = time[0]
        
        # Find the index closest to the last cycle start
        start_idx = np.argmin(np.abs(time - last_cycle_start_time))
        
        # Find midpoint of last cycle for dead time measurement
        midpoint_time = time[-1] - (0.5 / fsw)
        midpoint_idx = np.argmin(np.abs(time - midpoint_time))
        
        # Find next switch turn-on event after midpoint
        turn_on_indices = np.where(iDS[midpoint_idx:] > threshold)[0]
        if len(turn_on_indices) == 0:
            warnings.warn(
                f"No switch turn-on detected after t={time[midpoint_idx]:.6f}s "
                f"(threshold={threshold}A). May indicate insufficient dead time or ZVS loss.",
                RuntimeWarning
            )
            return np.nan
        
        next_on_idx = turn_on_indices[0]
        dead_time = time[midpoint_idx + next_on_idx] - time[midpoint_idx]
        
        # Sanity check: dead time shouldn't be > half cycle period
        half_period = 0.5 / fsw
        if dead_time > half_period:
            warnings.warn(
                f"Calculated dead time ({dead_time*1e6:.2f}µs) exceeds half period "
                f"({half_period*1e6:.2f}µs). Result may be invalid.",
                RuntimeWarning
            )
        
        return dead_time
        
    except (IndexError, ValueError) as e:
        warnings.warn(
            f"Could not compute dead time: {str(e)}. "
            f"Check simulation length and iDS waveform.",
            RuntimeWarning
        )
        return np.nan


def validate_analysis_signals(analysis: Any, required_signals: list) -> None:
    """
    Validate that all required signals are present in analysis results.
    
    Parameters
    ----------
    analysis : PySpice waveform object
        The simulation result object
    required_signals : list
        List of required signal names
        
    Raises
    ------
    ValueError
        If any required signal is missing
    """
    missing = []
    for signal in required_signals:
        # Check for 'time' attribute or other signals as dict keys
        if signal == 'time':
            if not hasattr(analysis, 'time'):
                missing.append(signal)
        else:
            try:
                _ = analysis[signal]
            except (KeyError, TypeError):
                missing.append(signal)
    
    if missing:
        raise ValueError(
            f"Missing required signals in analysis results: {', '.join(missing)}"
        )


def analyze(analysis: Any, 
           fsw: float, 
           TimeStep: float, 
           SimCycles: int, 
           return_arrays: bool = False,
           cycles_to_analyze: int = 30,
           vab_threshold: float = 0.1,
           dead_time_threshold: float = 0.001) -> Dict[str, Any]:
    """
    Analyze the results of a transient simulation for an LLC converter.

    Extracts key waveforms and calculates performance indicators.
    Optionally returns the full waveform arrays.

    Parameters
    ----------
    analysis : PySpice waveform object
        The result object returned by `simulator.transient(...)`.
    fsw : float
        Switching frequency in Hz.
    TimeStep : float
        Simulation time step in seconds.
    SimCycles : int
        Number of full switching cycles simulated.
    return_arrays : bool, optional
        If True, includes the full NumPy waveform arrays in the returned
        dictionary. Defaults to False (to save memory).
    cycles_to_analyze : int, optional
        Number of cycles at end of simulation to analyze. Default 20.
    vab_threshold : float, optional
        Voltage threshold for switch conduction detection (V). Default 0.1.
    dead_time_threshold : float, optional
        Current threshold for dead time detection (A). Default 0.001.

    Returns
    -------
    dict
        A dictionary containing calculated scalar metrics.
        If `return_arrays` is True, the dictionary also includes the full
        waveform arrays, keyed by their variable names (e.g., 'time', 'vab').

        Scalar Metrics (typically calculated over the last N cycles):
        - 'voutAVG': Average output voltage (V)
        - 'iSecRMS': RMS secondary side current (A)
        - 'iSecPK': Peak secondary current (A)
        - 'iD1PK': Peak diode D1 current (A)
        - 'iD1RMS': RMS diode D1 current (A)
        - 'iRRMS': RMS resonant current (A)
        - 'iRPK': Peak resonant current (A)
        - 'iDSRMS': RMS switch current (A)
        - 'iDSPK': Peak switch current (A)
        - 'iDSoff': Switch current at simulation end (A)
        - 'vCsRMS': RMS capacitor voltage (V)
        - 'vCsPK': Peak capacitor voltage (V)
        - 'ioutAVG': Average output current (A)
        - 'iLmAVG': Average magnetizing current (A)
        - 'vLmPK': Peak magnetizing voltage (V)
        - 'iCoRMS': RMS output capacitor current (A)
        - 'MaximumDeadTime': Estimated dead time before switch turn-on (s)

        Waveform Arrays (only included if return_arrays=True):
        - 'time': Simulation time array (s)
        - 'vab': Voltage across the switching bridge (V)
        - 'vpri': Primary side voltage (V)
        - 'vout': Output voltage (V)
        - 'iR': Resonant current (A)
        - 'iLm': Magnetizing current (A)
        - 'iD1', 'iD2': Diode currents (A)
        - 'iSec': Total secondary current (A)
        - 'iout': Output current (A)
        - 'vcs': Capacitor voltage (V)
        - 'ico': Output capacitor current (A)
        - 'iDS': Approximated switch current (A)
    """
    
    # Validate input parameters
    if fsw <= 0:
        raise ValueError(f"Switching frequency must be positive, got {fsw}")
    if TimeStep <= 0:
        raise ValueError(f"Time step must be positive, got {TimeStep}")
    if SimCycles < 1:
        raise ValueError(f"SimCycles must be >= 1, got {SimCycles}")
    if cycles_to_analyze < 1:
        raise ValueError(f"cycles_to_analyze must be >= 1, got {cycles_to_analyze}")
    
    # Validate required signals are present
    required_signals = ['time', 'vab', 'pri', 'vo', 'VLs_plus', 'VLm_plus', 
                       'VD1_cathode', 'VD2_cathode', 'VRload_plus', 'cs', 'Vco_plus']
    validate_analysis_signals(analysis, required_signals)
    
    # Extract and Derive Waveforms 
    time = np.array(analysis.time)
    vab = np.array(analysis['vab'])
    vpri = np.array(analysis['pri'])
    vout = np.array(analysis['vo'])
    iR = np.array(analysis['VLs_plus'])
    iLm = np.array(analysis['VLm_plus'])
    iD1 = np.array(analysis['VD1_cathode'])
    iD2 = np.array(analysis['VD2_cathode'])
    iSec = iD1 + iD2
    iout = np.array(analysis['VRload_plus'])
    vcs = np.array(analysis['vab'] - analysis['cs'])
    ico = np.array(analysis['Vco_plus'])
    MOSVds = np.array(analysis['N001'] - analysis['vab'])
    IMOS =  np.array(analysis['VSw1'])

    
    # Approximate switch current: conducts when bridge voltage is low
    iDS = np.where(vab < vab_threshold, -iR, 0)
    
    # Determine analysis slice (last N cycles) using robust time-based method
    # Work backwards from the end time to find where analysis should start
    analysis_start_time = time[-1] - (cycles_to_analyze / fsw)
    
    if analysis_start_time < time[0]:
        # Requested cycles exceed simulation length - use all data
        analysis_slice = slice(None, None)
        actual_cycles = (time[-1] - time[0]) * fsw
        warnings.warn(
            f"Requested {cycles_to_analyze} cycles for analysis, but simulation "
            f"only contains ~{actual_cycles:.1f} cycles. Using full simulation data.",
            RuntimeWarning
        )
    else:
        # Find the index closest to the analysis start time
        start_idx = np.argmin(np.abs(time - analysis_start_time))
        analysis_slice = slice(start_idx, None)
    
    # Calculate Scalar Metrics
    def safe_mean(arr):
        """Helper to safely compute mean"""
        return np.mean(arr).astype(float) if arr.size > 0 else np.nan
    
    def safe_rms(arr):
        """Helper to safely compute RMS"""
        return np.sqrt(np.mean(arr**2)).astype(float) if arr.size > 0 else np.nan
    
    def safe_max(arr):
        """Helper to safely compute max"""
        return np.max(arr).astype(float) if arr.size > 0 else np.nan
    
    metrics = {
        'voutAVG': safe_mean(vout[analysis_slice]),
        'iSecRMS': safe_rms(iSec[analysis_slice]),
        'iSecPK': safe_max(iSec[analysis_slice]),
        'iD1PK': safe_max(iD1[analysis_slice]),
        'iD1RMS': safe_rms(iD1[analysis_slice]),
        'iRRMS': safe_rms(iR[analysis_slice]),
        'iRPK': safe_max(iR[analysis_slice]),
        'iDSRMS': safe_rms(iDS[analysis_slice]),
        'iDSPK': safe_max(iDS[analysis_slice]),
        'iDSoff': iDS[-1].astype(float) if len(iDS) > 0 else np.nan,
        'vCsRMS': safe_rms(vcs[analysis_slice]),
        'vCsPK': safe_max(vcs[analysis_slice]),
        'ioutAVG': safe_mean(iout[analysis_slice]),
        'iLmAVG': safe_mean(iLm[analysis_slice]),
        'vLmPK': safe_max(vpri[analysis_slice]),
        'iCoRMS': safe_rms(ico[analysis_slice]),
        'MaximumDeadTime': float(compute_dead_time(time, iDS, fsw, 
                                                   dead_time_threshold)),
    }

    # Optionally Add Full Arrays to the Return Dictionary 
    if return_arrays:
        arrays = {
            'time': time,
            'vab': vab,
            'vpri': vpri,
            'vout': vout,
            'iR': iR,
            'iLm': iLm,
            'iD1': iD1,
            'iD2': iD2,
            'iSec': iSec,
            'iout': iout,
            'vcs': vcs,
            'ico': ico,
            'iDS': iDS,
            'MOSVds':MOSVds,
            'IMOS': IMOS
        }
        return {**metrics, **arrays}
    else:
        return metrics

