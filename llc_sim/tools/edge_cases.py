import os
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from llc_sim.tools import find_vtarget

# ─────────────────────────────────────────────────────────────────────

def create_comparison_table(test_cases: dict):
    """Creates a summary DataFrame and a formatted table from results."""
    # Lazy Import: Mova a importação para dentro da função
    import pandas as pd
    from tabulate import tabulate

    rows = {
        "Parameter": [
            "Po", "VoutAVG", "fsw", "dt_max", "dt_max(%)", "IDSpk", "IDS_off", "IDSrms",
            "ID1rms", "ID1pk", "IRrms", "IRpk", "VLmpk", "VCspk", "VCsrms", "ICorms"
        ]
    }
    vos = {}
    req = {"time", "vout", "iR", "vab", "iDS", "iSec"}

    for case_name, case_data in test_cases.items():
        results = case_data["results"]
        fsw_khz = case_data["fsw_khz"]

        try:
            _, rest = case_name.split("/")
            vout_str, iout_str = rest.split("@")
            voltage = float(vout_str.replace("V", "").strip())
            current = float(iout_str.replace("A", "").strip())
        except ValueError:
            print(f"Warning: Could not parse '{case_name}' for Po calculation. Setting Po to 0.")
            voltage, current = 0.0, 0.0

        po = voltage * current
        dt_max = results.get("MaximumDeadTime", 0.0)
        dt_pct = (dt_max * (fsw_khz * 1000.0)) * 50.0 if fsw_khz else 0.0

        rows[case_name] = [
            f"{po:.1f} W",
            f"{results.get('voutAVG', 0.0):.4f}",
            f"{fsw_khz:.4f} kHz",
            f"{dt_max*1e9:.4f} ns",
            f"{dt_pct:.4f}%",
            f"{results.get('iDSPK', 0):.4f} A",
            f"{results.get('iDSoff', 0):.4f} A",
            f"{results.get('iDSRMS', 0):.4f} A",
            f"{results.get('iD1RMS', 0):.4f} A",
            f"{results.get('iD1PK', 0):.4f} A",
            f"{results.get('iRRMS', 0):.4f} A",
            f"{results.get('iRPK', 0):.4f} A",
            f"{results.get('vLmPK', 0):.4f} V",
            f"{results.get('vCsPK', 0):.4f} V",
            f"{results.get('vCsRMS', 0):.4f} V",
            f"{results.get('iCoRMS', 0):.4f} A",
        ]

        if req.issubset(results):
            vos[case_name] = {k: results[k] for k in req}

    df = pd.DataFrame(rows).set_index("Parameter")
    return tabulate(df, headers="keys", tablefmt="fancy_grid"), df, vos


# ─────────────────────────────────────────────────────────────────────

def get_case_data(params, vtarget, SimCycles, TimeStep, tol, plot=False):
    """Wrapper to call the frequency finding function."""
    result, fsw_khz = find_vtarget(params=params, Vtarget=vtarget, tol=tol,
                                  SimCycles=SimCycles, TimeStep=TimeStep, Full_Arr=plot, stable_blocks_required=3, avg_cycles = 10, cycles_per_block=20,steady_state_tol=1e-3)
    return {"results": result, "fsw_khz": fsw_khz}

# ─────────────────────────────────────────────────────────────────────

def process_case(case_name, base_params, SimCycles, TimeStep, tol, plot):
    """Parses the case string, sets up, times, and runs the simulation."""
    try:
        vbus_str, rest = case_name.split("/")
        vout_str, iout_str = rest.split("@")
        vbus = float(vbus_str.replace("V", "").strip())
        vtarget = float(vout_str.replace("V", "").strip())
        current = float(iout_str.replace("A", "").strip())
        if vtarget < 0 or current < 0 or vbus <= 0:
            raise ValueError("Voltages and current must be positive.")
        rload = 1e12 if abs(current) < 1e-9 else vtarget / current
        params = {**base_params, "Rload": rload, "Vbus": vbus}
        print(f"--> {case_name}: Vbus={vbus:.1f}V, Vout={vtarget:.1f}V, I={current:.3f}A, "
              f"R={rload:.3f}Ω, Cycles={SimCycles}, dt={TimeStep}")

        start_time = time.perf_counter()
        case_data = get_case_data(params, vtarget, SimCycles, TimeStep,
                                  tol=tol, plot=plot)
        duration = time.perf_counter() - start_time
        print(f"<-- finished {case_name} in {duration:.2f}s")

        return case_name, case_data

    except Exception as e:
        print(f"!! Skipping {case_name}: Invalid format or error during processing. ({e})")
        return None, None

# ─────────────────────────────────────────────────────────────────────

def plot_detailed_case_waveforms(case_name, case_data, fsw_khz,
                                 base_params, num_cycles_zoom=4,
                                 output_dir="outputs/plots", plot_dpi=300,
                                 file_format='pdf'):
    """Saves plots of the key waveforms for a given simulation result."""
    # Lazy Import: Mova a importação para dentro da função de plotagem
    import matplotlib.pyplot as plt

    req = ["time", "vout", "iR", "iDS", "iSec", "vab"]
    if not all(k in case_data for k in req):
        print(f"Waveforms incomplete for {case_name}, skipping plot.")
        return
    t, vout, Ir, Ids, Isec, vab = (np.asarray(case_data[k]) for k in req)
    if t.size < 2:
        print(f"Not enough data points for {case_name}, skipping plot.")
        return
    tm = t * 1e3
    fig, axes = plt.subplots(4, 1, figsize=(12, 15), sharex=False)
    fig.suptitle(f'LLC {case_name} (fsw={fsw_khz:.3f} kHz)')
    axes[0].plot(tm, vout, color='blue', label="Vout")
    axes[0].set(title="Output Voltage (Vout)", xlabel="Time (ms)", ylabel="Voltage (V)")
    axes[0].legend(); axes[0].grid(True)
    axes[1].plot(tm, vout, color='blue', label="Vout")
    axes[1].set(title="Vout & Resonator Current (Ir)", xlabel="Time (ms)", ylabel="Voltage (V)")
    axes[1].tick_params(axis='y', labelcolor='blue')
    ax1b = axes[1].twinx(); ax1b.plot(tm, Ir, color='red', label="Ir")
    ax1b.set_ylabel("Current (A)", color='red'); ax1b.tick_params(axis='y', labelcolor='red')
    lines, labels = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1b.legend(lines + lines2, labels + labels2, loc='upper right'); axes[1].grid(True)
    if fsw_khz > 0:
        Tms = 1 / fsw_khz
        zoom_mask = tm >= tm[-1] - num_cycles_zoom * Tms
    else:
        zoom_mask = slice(-500, None)
    axes[2].plot(tm[zoom_mask], Ids[zoom_mask], color='green', label="Ids")
    axes[2].set(title="Primary Current (Ids) & Voltage (Vab) (Zoom)", xlabel="Time (ms)", ylabel="Current (A)")
    axes[2].tick_params(axis='y', labelcolor='green')
    ax2b = axes[2].twinx(); ax2b.plot(tm[zoom_mask], vab[zoom_mask], color='purple', linestyle='--', label="Vab")
    ax2b.set_ylabel("Voltage (V)", color='purple'); ax2b.tick_params(axis='y', labelcolor='purple')
    lines, labels = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2b.legend(lines + lines2, labels + labels2, loc='upper right'); axes[2].grid(True)
    axes[3].plot(tm[zoom_mask], Isec[zoom_mask], color='orange', label="Isec")
    axes[3].set(title="Secondary Current (Isec) (Zoom)", xlabel="Time (ms)", ylabel="Current (A)")
    axes[3].legend(); axes[3].grid(True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    safe_name = case_name.replace("@", "_at_").replace("/", "_out_")
    file_name = f"waves_{safe_name}.{file_format}"
    file_path = os.path.join(output_dir, file_name)
    fig.savefig(file_path, dpi=plot_dpi)
    plt.close(fig)
    print(f"Plot saved: {file_name}")

# ─────────────────────────────────────────────────────────────────────

def test_edge_cases(desired_cases_identifiers, base_params,
                             SimCycles=500, TimeStep=10e-9,
                             plot=False, show_table=False, save_csv=False,
                             max_workers=None, tol=0.1,
                             return_arrays=False):
    """
    Simulates a list of LLC operating points IN PARALLEL and prints total runtime.
    """
    import os, time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Validate input
    if not isinstance(desired_cases_identifiers, (list, set)):
        print("Error: 'desired_cases_identifiers' must be a list or set.")
        return []
    identifiers = [c for c in desired_cases_identifiers if isinstance(c, str)]
    if not identifiers:
        print("Warning: No valid string identifiers found in the input list.")
        return []

    generate_full_arrays = plot or return_arrays
    test_cases = {}

    # --- Parallel execution with timing ---
    workers = max_workers or os.cpu_count() or 1
    print(f"\nRunning in PARALLEL with max_workers={workers}...")
    start_total = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                process_case,
                c,
                base_params,
                SimCycles,
                TimeStep,
                tol,
                generate_full_arrays
            ): c for c in identifiers
        }

        for fut in as_completed(futs):
            name, data = fut.result()
            if name and data:
                test_cases[name] = data

    total_runtime = time.perf_counter() - start_total
    print(f"\nTotal simulation runtime (parallel): {total_runtime:.2f} seconds")

    if not test_cases:
        print("\nNo cases were successfully processed.")
        return []

    print(f"\n{len(test_cases)} cases were successfully processed.")

    # Build table (only if requested)
    df = None
    if show_table or save_csv:
        table, df, vos = create_comparison_table(test_cases)
        if show_table:
            print("\n" + "="*80)
            print("SIMULATION SUMMARY")
            print("="*80)
            print(table)
            print("="*80 + "\n")

    # Plots (only if requested)
    if plot:
        _, _, vos = create_comparison_table(test_cases)
        if vos:
            out_dir = os.path.join(os.curdir, "outputs", "plots")
            os.makedirs(out_dir, exist_ok=True)
            print(f"Saving plots to '{out_dir}'...")
            for cname, data in vos.items():
                plot_detailed_case_waveforms(
                    cname,
                    data,
                    test_cases[cname]["fsw_khz"],
                    base_params,
                    num_cycles_zoom=5,
                    output_dir=out_dir
                )
        else:
            print("Plotting was requested, but no waveform data was generated.")

    # CSV (only if requested)
    if save_csv and df is not None and not df.empty:
        out_dir = os.path.join(os.curdir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        fname = f'test_cases_{datetime.now():%Y%m%d_%H%M%S}.csv'
        full_path = os.path.join(out_dir, fname)
        df.to_csv(full_path)
        print(f"CSV file saved to {full_path}")

    # Return compact list of results (order by case name for determinism)
    return [{"case": n, "fsw_khz": d["fsw_khz"], **d["results"]}
            for n, d in sorted(test_cases.items(), key=lambda x: x[0])]
