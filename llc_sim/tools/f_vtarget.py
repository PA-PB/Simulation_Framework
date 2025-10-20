import numpy as np
from functools import partial
from scipy.optimize import minimize_scalar
import traceback

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

from llc_sim import build_llc_circuit, analyze, simulate



def simulate_until_steady_state(
    circuit, fsw, TimeStep, MaxCycles,
    cycles_per_block, SteadyStateTol,
    avg_cycles, stable_blocks_required,
    SteadyStateNode='vo'
):
    """
    Simulate circuit until steady state is reached by monitoring:
      - block mean stabilization over the last `avg_cycles` cycles, and
      - Poincaré (cycle-to-cycle) repeat at the end of each block.

    Returns:
        (final_analysis, cycles_to_stable) or (None, None) if simulation fails
    """

    # ---- helpers --------------------
    def get_node(analysis, name):
        # Try 'v(name)', 'name', 'V(name)', then analysis.nodes[name]
        for key in (f'v({name})', name, f'V({name})'):
            try:
                return np.asarray(analysis[key], dtype=float)
            except Exception:
                pass
        try:
            return np.asarray(analysis.nodes[name], dtype=float)
        except Exception:
            pass
        raise KeyError(f"Node vector not found for {name}")

    def get_ind_current(analysis, lname):
        # Try @L[i], i(L), L#branch, then analysis.branches['L']
        for key in (f'@{lname}[i]', f'i({lname})', f'{lname}#branch'):
            try:
                return np.asarray(analysis[key], dtype=float)
            except Exception:
                pass
        try:
            return np.asarray(analysis.branches[lname], dtype=float)
        except Exception:
            pass
        raise KeyError(f"Inductor current not found for {lname}")

    def apply_element_ic(circ, cap_ic=None, ind_ic=None):
        # Best-effort: set element initial_condition on C and L
        if cap_ic:
            for name, V0 in cap_ic.items():
                try:
                    circ.element(name).initial_condition = float(V0)
                except Exception:
                    pass
        if ind_ic:
            for name, I0 in ind_ic.items():
                try:
                    circ.element(name).initial_condition = float(I0)
                except Exception:
                    pass

    # ----------------------------------------------------------------
    Tsw = 1.0 / fsw
    current_cycle = 0
    previous_avg = None
    prev_end_value = None         
    final_analysis = None
    stable_block_counter = 0

    initial_conditions = {}

    while current_cycle < MaxCycles:
        simulator = circuit.simulator()
        simulator.options(
            method='gear',
            maxord=3,
            reltol=1e-3,
            abstol=1e-9,
            vntol=1e-6,
            chgtol=1e-16,
            itl4=500,
            plotwinsize=0
        )

        # Save only what's needed (robust spellings)
        save_list = []
        # steady node + nodes needed for Cs and Co reconstruction
        for n in {SteadyStateNode, 'cs', 'vab', 'va'}:
            save_list += [f'v({n})', n]
        # inductor currents
        for lname in ('Ls', 'Lm'):
            save_list += [f'@{lname}[i]', f'i({lname})', f'{lname}#branch']
        simulator.save(save_list)

        # carry node .ic (voltage) if available
        if initial_conditions:
            simulator.initial_condition(**initial_conditions)

        block_duration = cycles_per_block * Tsw

        try:
            analysis = simulator.transient(
                step_time=TimeStep,
                end_time=block_duration,
                use_initial_condition=True  # honor element ICs (uic)
            )
        except Exception:
            return None, None

        # minimal sanity
        try:
            time_vector = np.asarray(analysis.time, dtype=float)
        except Exception:
            return None, None
        if time_vector.size < 2:
            return None, None

        final_analysis = analysis
        current_cycle += cycles_per_block

        # --- monitored node vector (robust) ---
        try:
            node_vector = get_node(analysis, SteadyStateNode)
        except Exception:
            return None, None

        # --- mean over last `avg_cycles` cycles ---
        time_to_avg = avg_cycles / fsw
        block_end_time = float(time_vector[-1])
        avg_start_time = max(0.0, block_end_time - time_to_avg)
        avg_indices = (time_vector >= avg_start_time)
        final_waveform_segment = node_vector[avg_indices]
        current_avg = float(np.mean(final_waveform_segment)) if final_waveform_segment.size else float(node_vector[-1])

        # --- Poincaré: end-of-block value ---
        end_value = float(node_vector[-1])

        # --- compare with previous block ---
        ok_avg = False
        ok_poincare = False

        if previous_avg is not None:
            denom_mu = max(abs(previous_avg), 1e-9)
            avg_error = abs((current_avg - previous_avg) / denom_mu)
            ok_avg = (avg_error < SteadyStateTol)

        if prev_end_value is not None:
            denom_end = max(abs(prev_end_value), 1e-9)
            end_error = abs((end_value - prev_end_value) / denom_end)
            ok_poincare = (end_error < SteadyStateTol)

        ok = ok_avg and ok_poincare  # require BOTH gates

        stable_block_counter = (stable_block_counter + 1) if ok else 0
        if stable_block_counter >= stable_blocks_required:
            return final_analysis, current_cycle

        # ---------------- state handoff to next block ----------------
        previous_avg = current_avg
        prev_end_value = end_value

        initial_conditions.clear()
        try:
            for nd in analysis.nodes.values():
                initial_conditions[str(nd.name)] = float(nd[-1])
        except Exception:
            initial_conditions = {}

        # 2) element ICs (best-effort): Cs, Co from node Vs; Ls, Lm from currents
        cap_ic = {}
        ind_ic = {}

        # Co: between vo and 0
        try:
            v_vo = get_node(analysis, 'vo')
            cap_ic['Co'] = float(v_vo[-1])
        except Exception:
            pass

        # Cs: between (vab,cs) or (va,cs) — try vab first, then va
        try:
            v_cs = get_node(analysis, 'cs')
            try:
                v_pos = get_node(analysis, 'vab')
            except Exception:
                v_pos = get_node(analysis, 'va')
            cap_ic['Cs'] = float(v_pos[-1] - v_cs[-1])
        except Exception:
            pass

        # Ls, Lm currents
        for lname in ('Ls', 'Lm'):
            try:
                iL = get_ind_current(analysis, lname)
                ind_ic[lname] = float(iL[-1])
            except Exception:
                pass

        apply_element_ic(circuit, cap_ic=cap_ic, ind_ic=ind_ic)
        # ----------------------------------------------------------------
    return final_analysis, current_cycle



class ToleranceReached(Exception):
    """Exception raised when voltage tolerance is met during optimization."""
    def __init__(self, fsw_khz, results, vout, error):
        self.fsw_khz = fsw_khz
        self.results = results
        self.vout = vout
        self.error = error


def voltage_error(fsw_khz, params, Vtarget, tol, SimCycles, TimeStep, Full_Arr, 
                  cycles_per_block, steady_state_tol, 
                  avg_cycles, stable_blocks_required):
    """
    Calculate voltage error for given switching frequency.
    Uses the simplified circuit (config=2) during the search to speed up iterations.
    """
    current_params = params.copy()
    current_params["fsw"] = fsw_khz * 1e3  # Convert kHz to Hz

    fsw_min = params.get('fsw_min', 10e3)
    fsw_max = params.get('fsw_max', 1000e3)
    if not (fsw_min <= current_params["fsw"] <= fsw_max):
        return 1e6

    try:
        # --- use simplified circuit for the loop ---
        circuit = build_llc_circuit(current_params, Vtarget, config=2)

        analysis_fast, _cycles_to_stable = simulate_until_steady_state(
            circuit=circuit,
            fsw=current_params["fsw"],
            TimeStep=TimeStep,
            MaxCycles=20e3,
            cycles_per_block=cycles_per_block,
            SteadyStateNode='vo',
            SteadyStateTol=steady_state_tol,
            avg_cycles=avg_cycles,
            stable_blocks_required=stable_blocks_required
        )

        if analysis_fast is None:
            print(f"  > Simulation failed for fsw = {fsw_khz:.3f} kHz (config=2)")
            return 1e6

        fsw = current_params["fsw"]  # Hz
        Tsw = 1.0 / fsw
        time_vec = analysis_fast.time
        vo_vec = analysis_fast['vo']

        block_end_time = float(time_vec[-1])
        avg_start_time = max(0.0, block_end_time - avg_cycles * Tsw)
        avg_indices = time_vec >= (avg_start_time @ u_s)

        final_segment = np.asarray(vo_vec[avg_indices], dtype=float)
        if final_segment.size == 0 or not np.all(np.isfinite(final_segment)):
            final_segment = np.asarray(vo_vec, dtype=float)

        vout_fast = float(np.mean(final_segment))
        if not np.isfinite(vout_fast):
            return 1e6

        error = abs(vout_fast - Vtarget)

        # Early exit when voltage tolerance is met
        eps = max(1e-6, 1e-6 * abs(Vtarget))
        if tol > 0 and error <= tol + eps:
            raise ToleranceReached(fsw_khz, results=None, vout=vout_fast, error=error)

        return error

    except ToleranceReached:
        raise
    except Exception:
        print(f"[ERROR] Unexpected failure at fsw = {fsw_khz:.3f} kHz (config=2). Details below:")
        traceback.print_exc()
        return 1e6



def find_vtarget(params, Vtarget, tol, SimCycles, TimeStep, 
                 cycles_per_block, steady_state_tol, 
                 avg_cycles, stable_blocks_required, Full_Arr=False):
    """
    Find switching frequency that achieves target output voltage.
    Uses config=2 (simplified) during the search/steady-state loop for speed,
    then runs a final high-fidelity simulation with config=1.
    """
    Voafo = params["Vbus"] / (2 * params["n"])
    f1 = (1 / (2 * np.pi * np.sqrt(params["Cs"] * (params["Ls"] + params["Lm"])))) / 1e3
    fo = (1 / (2 * np.pi * np.sqrt(params["Cs"] * params["Ls"]))) / 1e3

    if Vtarget >= Voafo:
        low, high = f1, fo
    else:
        low, high = fo, 3*fo

    margin = 10.0
    low -= margin
    high += margin

    fsw_min_khz = params.get('fsw_min', 10e3) / 1e3
    fsw_max_khz = params.get('fsw_max', 1e6) / 1e3
    low = max(low, fsw_min_khz)
    high = min(high, fsw_max_khz)

    print(f"Starting optimization in bounds ({low:.3f}, {high:.3f}) kHz for Vtarget = {Vtarget} V")

    objective_function = partial(
        voltage_error,
        params=params, 
        Vtarget=Vtarget, 
        tol=tol,
        SimCycles=SimCycles, 
        TimeStep=TimeStep, 
        Full_Arr=Full_Arr,
        cycles_per_block=cycles_per_block,
        steady_state_tol=steady_state_tol,
        avg_cycles=avg_cycles,
        stable_blocks_required=stable_blocks_required
    )

    try:
        result = minimize_scalar(
            objective_function,
            bounds=(low, high),
            method='bounded',
            options={'xatol': 1e-3, 'maxiter': 100}
        )

        fsw_khz = float(result.x)
        print("\nOptimization finished without reaching the exact voltage tolerance.")
        print(f"Best fsw = {fsw_khz:.6f} kHz, |err| ≈ {result.fun:.6g} V")

        # --- Final high-fidelity sim with config=1 ---
        try:
            current_params = params.copy()
            current_params["fsw"] = fsw_khz * 1e3  # Hz

            # Use simplified circuit again just to estimate cycles_to_stable quickly
            circuit_fast = build_llc_circuit(current_params, Vtarget, config=2)
            _, cycles_to_stable = simulate_until_steady_state(
                circuit=circuit_fast,
                fsw=current_params["fsw"],
                TimeStep=TimeStep,
                MaxCycles=20e3,
                cycles_per_block=cycles_per_block,
                SteadyStateNode='vo',
                SteadyStateTol=steady_state_tol,
                avg_cycles=avg_cycles,
                stable_blocks_required=stable_blocks_required
            )

            if cycles_to_stable is None:
                final_sim_cycles = max(cycles_per_block, 5*avg_cycles)
            else:
                final_sim_cycles = max(cycles_per_block, int(cycles_to_stable * 1))

            # Now build the full circuit for the final analysis
            circuit_full = build_llc_circuit(current_params, Vtarget, config=1)

            final_analysis = simulate(
                circuit=circuit_full,
                fsw=current_params["fsw"],
                TimeStep=TimeStep,
                SimCycles=final_sim_cycles,
                TimeStartSave=0
            )

            final_results = analyze(
                final_analysis, 
                current_params["fsw"], 
                TimeStep,
                final_sim_cycles, 
                return_arrays=True
            )

        except Exception:
            final_results = {
                'note': 'final dynamic-length sim (config=1) failed or was skipped',
                'error': result.fun
            }

        return final_results, fsw_khz

    except ToleranceReached as e:
        print(f"\nFinal solution found! fsw = {e.fsw_khz:.3f} kHz "
              f"(Vout={e.vout:.4f} V, |err|={e.error:.6g} V ≤ tol={tol})")

        try:
            current_params = params.copy()
            current_params["fsw"] = e.fsw_khz * 1e3

            # Estimate length with simplified model
            circuit_fast = build_llc_circuit(current_params, Vtarget, config=2)
            _, cycles_to_stable = simulate_until_steady_state(
                circuit=circuit_fast,
                fsw=current_params["fsw"],
                TimeStep=TimeStep,
                MaxCycles=20e3,
                cycles_per_block=cycles_per_block,
                SteadyStateNode='vo',
                SteadyStateTol=steady_state_tol,
                avg_cycles=avg_cycles,
                stable_blocks_required=stable_blocks_required
            )

            if cycles_to_stable is None:
                final_sim_cycles = max(cycles_per_block, 5*avg_cycles)
            else:
                final_sim_cycles = max(cycles_per_block, int(cycles_to_stable * 1))

            # Final high-fidelity run
            circuit_full = build_llc_circuit(current_params, Vtarget, config=1)

            final_analysis = simulate(
                circuit=circuit_full,
                fsw=current_params["fsw"],
                TimeStep=TimeStep,
                SimCycles=final_sim_cycles,
                TimeStartSave=0
            )

            final_results = analyze(
                final_analysis, 
                current_params["fsw"], 
                TimeStep,
                final_sim_cycles, 
                return_arrays=True
            )

        except Exception:
            final_results = {
                'voutAVG': e.vout,
                'note': 'early stop; final dynamic-length sim (config=1) failed or skipped'
            }

        return final_results, e.fsw_khz

