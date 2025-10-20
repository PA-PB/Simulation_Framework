# LLC Resonant Converter Design & Simulation Tool

A comprehensive Python-based simulation and design tool for LLC resonant converters using PySpice/ngspice. This tool enables automated frequency search, multi-operating point analysis, and parallel simulation capabilities for power electronics design.

## Features

- **Automated Frequency Search**: Finds optimal switching frequency to achieve target output voltage
- **Multi-Operating Point Testing**: Test hundreds of operating conditions in parallel
- **Steady-State Detection**: Intelligent simulation that stops when steady state is reached
- **Comprehensive Analysis**: Extracts key performance metrics (currents, voltages, dead time, etc.)
- **Parallel Processing**: Multi-core support for fast batch simulations
- **Visualization Tools**: Built-in plotting and reporting capabilities

## Installation

### Prerequisites

- Python 3.7 or higher
- ngspice (circuit simulator backend)

### Step 1: Clone the Repository

```bash
git clone <https://github.com/PA-PB/Simulation_Framework.git>
```

### Step 2: Install Python Dependencies

```bash
pip install -e .
```

This will automatically attempt to install the ngspice DLL. If it fails, run manually:

```bash
pyspice-post-installation --install-ngspice-dll
```


## Quick Start

### Basic Usage Example

```python
from llc_sim import build_llc_circuit, simulate, analyze

# Define circuit parameters
params = {
    'Rload': 700,      # Load resistance [Ω]
    'Cs': 6.8e-9,      # Series capacitor [F]
    'Ls': 150e-6,      # Series inductor [H]
    'Lm': 600e-6,      # Magnetizing inductor [H]
    'n': 2,            # Transformer turns ratio
    'Co': 10e-6,       # Output capacitor [F]
    'Vbus': 410,       # Input bus voltage [V]
    'fsw': 100e3       # Switching frequency [Hz]
}

SimCycles = 800
TimeStep = 100e-9

# Build circuit
circuit = build_llc_circuit(params, config=1)

# Run simulation
analysis = simulate(circuit, params['fsw'], TimeStep, SimCycles)

# Analyze results
results = analyze(analysis, params['fsw'], TimeStep, SimCycles)

print(f"Output Voltage: {results['voutAVG']:.2f} V")
print(f"Output Current: {results['ioutAVG']:.2f} A")
```

### Finding Target Voltage

```python
from llc_sim import find_vtarget

# Same parameters as above
params = {...}

# Find frequency for 100V output
results, fsw_khz = find_vtarget(
    params=params,
    Vtarget=100,           # Target output voltage [V]
    tol=0.1,               # Tolerance [V]
    SimCycles=800,
    TimeStep=100e-9,
    cycles_per_block=20,
    steady_state_tol=1e-3,
    avg_cycles=10,
    stable_blocks_required=3
)

print(f"Required frequency: {fsw_khz:.3f} kHz")
print(f"Achieved voltage: {results['voutAVG']:.4f} V")
```

### Testing Multiple Operating Points

```python
from llc_sim import test_edge_cases

# Define test cases
test_cases = {
    '410V/100V@1.5A',   # Format: Vbus/Vout@Iout
    '410V/100V@0.1A',
    '410V/200V@0.1A',
    '410V/200V@1.25A',
}

# Run parallel simulations
results = test_edge_cases(
    test_cases, 
    params, 
    SimCycles=800,
    TimeStep=100e-9,
    show_table=True,    # Display results table
    plot=True,          # Generate waveform plots
    save_csv=True,      # Save results to CSV
    tol=0.1
)
```

## Architecture

### Module Structure

```
llc_sim/
├── sim_lib/
│   ├── circuit_builder.py  # Circuit netlist generation
│   ├── simulation.py       # Transient simulation wrapper
│   ├── analysis.py         # Waveform extraction and metrics
│   └── transformer.py      # Transformer subcircuit model
└── tools/
    ├── f_vtarget.py        # Frequency search algorithm
    └── edge_cases.py       # Multi-point testing framework
```

### Circuit Configurations

**Config 1**: Full H-bridge with real switches (MOSFETs + gate drivers)
- Voltage-controlled switches with body diodes
- Dead-time modeling
- More accurate but slower simulation

**Config 2**: Ideal square-wave voltage source
- Simplified model for frequency search
- Faster convergence
- Used during optimization loops

## Key Parameters

### Circuit Parameters

| Parameter | Description | 
|-----------|-------------|
| `Vbus` | Input bus voltage 
| `Cs` | Series resonant capacitor | 
| `Ls` | Series resonant inductor | 
| `Lm` | Magnetizing inductance | 
| `n` | Transformer turns ratio | 
| `Co` | Output filter capacitor |
| `Rload` | Load resistance | 
| `fsw` | Switching frequency | 

### Simulation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `SimCycles` | Number of switching cycles | 800 |
| `TimeStep` | Simulation time step | 100 ns |
| `cycles_per_block` | Cycles per steady-state check | 20 |
| `avg_cycles` | Cycles to average for metrics | 10 |
| `stable_blocks_required` | Blocks needed for convergence | 3 |
| `steady_state_tol` | Convergence tolerance | 1e-3 |

## Results and Metrics

### Extracted Metrics

The `analyze()` function returns a dictionary with:

**Voltage Metrics:**
- `voutAVG`: Average output voltage
- `vCsRMS`, `vCsPK`: Resonant capacitor voltage (RMS, peak)
- `vLmPK`: Magnetizing inductor peak voltage

**Current Metrics:**
- `ioutAVG`: Average output current
- `iRRMS`, `iRPK`: Resonant current (RMS, peak)
- `iDSRMS`, `iDSPK`: Switch current (RMS, peak)
- `iD1RMS`, `iD1PK`: Diode current (RMS, peak)
- `iSecRMS`, `iSecPK`: Secondary current (RMS, peak)
- `iCoRMS`: Output capacitor RMS current

**Timing Metrics:**
- `MaximumDeadTime`: Available dead time before ZVS loss

### Optional Arrays

Set `return_arrays=True` to also get time-domain waveforms:
- `time`, `vout`, `vab`, `vpri`
- `iR`, `iLm`, `iDS`, `iSec`
- `iD1`, `iD2`, `ico`

## Advanced Features

### Parallel Batch Simulation

The tool automatically uses multiple CPU cores:

```python
# Process 1000 operating points in parallel
cases = generate_dense_operating_points(1000, method='latin_hypercube')

results = test_edge_cases(
    cases,
    params,
    SimCycles=500,
    TimeStep=100e-9,
    max_workers=None,  # Uses all available cores
    tol=0.1
)
```

### Steady-State Detection

Simulations automatically stop when steady state is reached using:
1. **Block mean stabilization**: Average over last N cycles converges
2. **Poincaré method**: Cycle-to-cycle repeatability

This significantly reduces simulation time compared to fixed-length simulations.

### Accuracy Tracking

```python
from profile_test_edge_cases import AccuracyTracker

tracker = AccuracyTracker(tolerance=0.1)  # 0.1V tolerance

for case in test_cases:
    v_target = parse_target_voltage(case)
    v_actual = run_simulation(case)
    tracker.add_result(case, v_target, v_actual)

# Generate comprehensive report
tracker.generate_accuracy_report()
tracker.plot_accuracy_analysis()
```

## Example Scripts

### Example.py

Basic demonstration of standalone functions:
- Circuit building
- Single-point simulation
- Multi-point testing

### profile_test_edge_cases.py

Advanced benchmarking script featuring:
- 1000-point operating space coverage
- Latin Hypercube sampling
- Performance tracking
- Accuracy analysis with visualization

Run with:
```bash
python profile_test_edge_cases.py
```


### Optimization Tips

3. **Adjust steady-state parameters**: Looser tolerances converge faster
4. **Reduce TimeStep**: Trade accuracy for speed (50-200 ns range)

## Troubleshooting

### Common Issues

**"ngspice DLL not found"**
```bash
pyspice-post-installation --install-ngspice-dll
```

**"Simulation failed to converge"**
- Increase `itl4` option (Newton iterations)
- Reduce `TimeStep` for stiff circuits
- Check if operating point is physically feasible

**"Steady state not reached"**
- Increase `MaxCycles` limit
- Relax `steady_state_tol`
- Reduce `stable_blocks_required`


## License

MIT License - see LICENSE file for details

## Author

GSEC Research Group

## References

3. PySpice Documentation: https://pyspice.fabrice-salvaire.fr/

