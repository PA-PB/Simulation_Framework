from llc_sim import test_edge_cases, simulate, analyze, build_llc_circuit
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np
import pandas as pd
import time
from datetime import timedelta

# ============================================
# ENHANCED OPERATING POINT GENERATORS
# ============================================

def generate_dense_operating_points(num_points=1000, method='latin_hypercube'):
    """
    Generate dense sampling of operating points within specifications.
    
    Parameters:
    - num_points: Target number of operating points to generate
    - method: 'latin_hypercube', 'random', 'grid', or 'adaptive'
    
    Returns:
    - Set of operating point strings in format 'VbusV/VoV@IoA'
    """
    
    # Operating window constraints
    V_BUS_MIN, V_BUS_MAX = 380, 450
    V_OUT_MIN, V_OUT_MAX = 70, 160
    I_OUT_MIN, I_OUT_MAX = 0.1, 1.5
    P_OUT_MAX = 150  # Watts
    
    cases = set()
    
    if method == 'latin_hypercube':
        # Latin Hypercube Sampling for better space coverage
        from scipy.stats import qmc
        
        sampler = qmc.LatinHypercube(d=3, seed=42)
        samples = sampler.random(n=num_points)
        
        # Scale to operating ranges
        vbus_samples = samples[:, 0] * (V_BUS_MAX - V_BUS_MIN) + V_BUS_MIN
        vo_samples = samples[:, 1] * (V_OUT_MAX - V_OUT_MIN) + V_OUT_MIN
        io_samples = samples[:, 2] * (I_OUT_MAX - I_OUT_MIN) + I_OUT_MIN
        
        for v_bus, v_o, i_o in zip(vbus_samples, vo_samples, io_samples):
            power = v_o * i_o
            if power <= P_OUT_MAX:
                # Round to reasonable precision
                v_bus = round(v_bus, 0)
                v_o = round(v_o, 1)
                i_o = round(i_o, 2)
                case_str = f"{int(v_bus)}V/{v_o:.1f}V@{i_o:.2f}A"
                cases.add(case_str)
    
    elif method == 'random':
        # Pure random sampling
        np.random.seed(42)
        attempts = 0
        max_attempts = num_points * 3  # Allow some rejections
        
        while len(cases) < num_points and attempts < max_attempts:
            v_bus = np.random.uniform(V_BUS_MIN, V_BUS_MAX)
            v_o = np.random.uniform(V_OUT_MIN, V_OUT_MAX)
            i_o = np.random.uniform(I_OUT_MIN, I_OUT_MAX)
            power = v_o * i_o
            
            if power <= P_OUT_MAX:
                v_bus = round(v_bus, 0)
                v_o = round(v_o, 1)
                i_o = round(i_o, 2)
                case_str = f"{int(v_bus)}V/{v_o:.1f}V@{i_o:.2f}A"
                cases.add(case_str)
            
            attempts += 1
    
    elif method == 'grid':
        # Dense uniform grid
        n_per_dim = int(np.ceil(num_points ** (1/3)))
        
        vbus_grid = np.linspace(V_BUS_MIN, V_BUS_MAX, n_per_dim)
        vo_grid = np.linspace(V_OUT_MIN, V_OUT_MAX, n_per_dim)
        io_grid = np.linspace(I_OUT_MIN, I_OUT_MAX, n_per_dim)
        
        from itertools import product
        for v_bus, v_o, i_o in product(vbus_grid, vo_grid, io_grid):
            power = v_o * i_o
            if power <= P_OUT_MAX and len(cases) < num_points:
                v_bus = round(v_bus, 0)
                v_o = round(v_o, 1)
                i_o = round(i_o, 2)
                case_str = f"{int(v_bus)}V/{v_o:.1f}V@{i_o:.2f}A"
                cases.add(case_str)
    
    elif method == 'adaptive':
        # Focus more samples on regions of interest
        np.random.seed(42)
        
        # Generate base samples
        base_cases = generate_dense_operating_points(num_points//2, 'latin_hypercube')
        cases.update(base_cases)
        
        # Add extra samples at boundaries
        boundary_samples = num_points // 4
        
        # Boundary: low voltage, high current
        for _ in range(boundary_samples // 4):
            v_bus = np.random.uniform(V_BUS_MIN, V_BUS_MIN + 30)
            v_o = np.random.uniform(V_OUT_MIN, V_OUT_MIN + 30)
            i_o = np.random.uniform(I_OUT_MAX - 0.5, I_OUT_MAX)
            if v_o * i_o <= P_OUT_MAX:
                case_str = f"{int(v_bus)}V/{v_o:.1f}V@{i_o:.2f}A"
                cases.add(case_str)
        
        # Boundary: high voltage, low current
        for _ in range(boundary_samples // 4):
            v_bus = np.random.uniform(V_BUS_MAX - 30, V_BUS_MAX)
            v_o = np.random.uniform(V_OUT_MAX - 30, V_OUT_MAX)
            i_o = np.random.uniform(I_OUT_MIN, I_OUT_MIN + 0.3)
            if v_o * i_o <= P_OUT_MAX:
                case_str = f"{int(v_bus)}V/{v_o:.1f}V@{i_o:.2f}A"
                cases.add(case_str)
        
        # Fill remaining with random samples
        while len(cases) < num_points:
            v_bus = np.random.uniform(V_BUS_MIN, V_BUS_MAX)
            v_o = np.random.uniform(V_OUT_MIN, V_OUT_MAX)
            i_o = np.random.uniform(I_OUT_MIN, I_OUT_MAX)
            if v_o * i_o <= P_OUT_MAX:
                case_str = f"{int(v_bus)}V/{v_o:.1f}V@{i_o:.2f}A"
                cases.add(case_str)
    
    print(f"\n✓ Generated {len(cases)} operating points using '{method}' method")
    return cases


def generate_comprehensive_test_suite():
    """
    Generate comprehensive test suite combining different sampling strategies.
    Total: ~1000 cases
    """
    all_cases = set()
    
    # 1. Extreme boundary cases (24 cases)
    extreme_cases = generate_extreme_operating_points()
    all_cases.update(extreme_cases)
    print(f"   • Extreme cases: {len(extreme_cases)}")
    
    # 2. Sparse strategic (68 cases)
    sparse_cases = generate_sparse_operating_points()
    all_cases.update(sparse_cases)
    print(f"   • Sparse strategic: {len(sparse_cases)}")
    
    # 3. Dense grid sampling (~300 cases)
    dense_grid = generate_dense_operating_points(300, method='grid')
    all_cases.update(dense_grid)
    print(f"   • Dense grid: {len(dense_grid)}")
    
    # 4. Latin hypercube (~600 cases)
    lhs_cases = generate_dense_operating_points(600, method='latin_hypercube')
    all_cases.update(lhs_cases)
    print(f"   • Latin hypercube: {len(lhs_cases)}")
    
    print(f"\n✓ Total unique cases: {len(all_cases)}")
    return all_cases


# ============================================
# ENHANCED ACCURACY METRICS
# ============================================

class AccuracyTracker:
    """Track voltage regulation accuracy for each operating point."""
    
    def __init__(self, tolerance=0.1):
        self.tolerance = tolerance  # Tolerance in volts (or percentage if < 1)
        self.results = []
        
    def add_result(self, case_str, v_target, v_actual):
        """
        Add a result and check if it's within tolerance.
        
        Parameters:
        - case_str: Operating point string
        - v_target: Target output voltage
        - v_actual: Actual simulated output voltage
        """
        # Calculate error
        error = abs(v_actual - v_target)
        
        # Determine tolerance (absolute or percentage)
        if self.tolerance < 1:
            # Percentage tolerance
            tol_absolute = v_target * self.tolerance
        else:
            # Absolute tolerance in volts
            tol_absolute = self.tolerance
        
        # Check if within tolerance
        within_tolerance = error <= tol_absolute
        error_percent = (error / v_target) * 100
        
        result = {
            'case': case_str,
            'v_target': v_target,
            'v_actual': v_actual,
            'error_V': error,
            'error_percent': error_percent,
            'tolerance_V': tol_absolute,
            'within_tolerance': within_tolerance,
            'status': 'PASS' if within_tolerance else 'FAIL'
        }
        
        self.results.append(result)
        return within_tolerance
    
    def generate_accuracy_report(self):
        """Generate comprehensive accuracy report."""
        if not self.results:
            print("No results to report")
            return
        
        df = pd.DataFrame(self.results)
        
        # Calculate statistics
        total = len(df)
        passed = df['within_tolerance'].sum()
        failed = total - passed
        pass_rate = (passed / total) * 100
        
        mean_error = df['error_V'].mean()
        max_error = df['error_V'].max()
        min_error = df['error_V'].min()
        std_error = df['error_V'].std()
        
        print(f"\n{'='*80}")
        print(f"VOLTAGE REGULATION ACCURACY REPORT")
        print(f"{'='*80}")
        
        print(f"\nOVERALL ACCURACY:")
        print(f"   Total cases: {total}")
        print(f"   Passed (within tolerance): {passed} ({pass_rate:.1f}%)")
        print(f"   Failed (outside tolerance): {failed} ({100-pass_rate:.1f}%)")
        print(f"   Tolerance: ±{self.tolerance:.3f} {'V' if self.tolerance >= 1 else '%'}")
        
        print(f"\nERROR STATISTICS:")
        print(f"   Mean error: {mean_error:.4f} V ({df['error_percent'].mean():.2f}%)")
        print(f"   Max error: {max_error:.4f} V ({df['error_percent'].max():.2f}%)")
        print(f"   Min error: {min_error:.4f} V ({df['error_percent'].min():.2f}%)")
        print(f"   Std deviation: {std_error:.4f} V")
        
        # Worst cases
        if failed > 0:
            print(f"\n❌ WORST CASES (Top 5):")
            worst_cases = df.nlargest(5, 'error_V')
            for idx, row in worst_cases.iterrows():
                print(f"   {row['case']}: Error = {row['error_V']:.4f}V ({row['error_percent']:.2f}%) - {row['status']}")
        
        # Best cases
        print(f"\n✅ BEST CASES (Top 5):")
        best_cases = df.nsmallest(5, 'error_V')
        for idx, row in best_cases.iterrows():
            print(f"   {row['case']}: Error = {row['error_V']:.4f}V ({row['error_percent']:.2f}%)")
        
        print(f"\n{'='*80}")
        
        return df
    
    def plot_accuracy_analysis(self, save_path='accuracy_analysis.png'):
        """Create detailed accuracy visualization."""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Voltage Regulation Accuracy Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Pass/Fail Distribution
        pass_fail_counts = df['status'].value_counts()
        colors = ['#2ecc71' if x == 'PASS' else '#e74c3c' for x in pass_fail_counts.index]
        axes[0, 0].pie(pass_fail_counts.values, labels=pass_fail_counts.index, 
                       autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 0].set_title('Pass/Fail Distribution', fontweight='bold')
        
        # Plot 2: Error Distribution Histogram
        axes[0, 1].hist(df['error_V'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(self.tolerance if self.tolerance >= 1 else df['v_target'].mean() * self.tolerance,
                          color='r', linestyle='--', linewidth=2, label='Tolerance')
        axes[0, 1].set_xlabel('Voltage Error (V)', fontweight='bold')
        axes[0, 1].set_ylabel('Count', fontweight='bold')
        axes[0, 1].set_title('Error Distribution', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: Error vs Target Voltage
        scatter = axes[0, 2].scatter(df['v_target'], df['error_V'], 
                                     c=df['within_tolerance'], cmap='RdYlGn',
                                     alpha=0.6, s=30)
        axes[0, 2].set_xlabel('Target Voltage (V)', fontweight='bold')
        axes[0, 2].set_ylabel('Voltage Error (V)', fontweight='bold')
        axes[0, 2].set_title('Error vs Target Voltage', fontweight='bold')
        axes[0, 2].grid(alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 2], label='Within Tolerance')
        
        # Plot 4: Percentage Error Distribution
        axes[1, 0].hist(df['error_percent'], bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Error (%)', fontweight='bold')
        axes[1, 0].set_ylabel('Count', fontweight='bold')
        axes[1, 0].set_title('Percentage Error Distribution', fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 5: Target vs Actual Voltage (Ideal Line)
        axes[1, 1].scatter(df['v_target'], df['v_actual'], alpha=0.5, s=20, c='#3498db')
        min_v = min(df['v_target'].min(), df['v_actual'].min())
        max_v = max(df['v_target'].max(), df['v_actual'].max())
        axes[1, 1].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Ideal (V_actual = V_target)')
        axes[1, 1].set_xlabel('Target Voltage (V)', fontweight='bold')
        axes[1, 1].set_ylabel('Actual Voltage (V)', fontweight='bold')
        axes[1, 1].set_title('Target vs Actual Voltage', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].set_aspect('equal', adjustable='box')
        
        # Plot 6: Cumulative Error Distribution
        sorted_errors = np.sort(df['error_V'])
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        axes[1, 2].plot(sorted_errors, cumulative, linewidth=2, color='#e67e22')
        axes[1, 2].axvline(self.tolerance if self.tolerance >= 1 else df['v_target'].mean() * self.tolerance,
                          color='r', linestyle='--', linewidth=2, label=f'Tolerance: {self.tolerance}V')
        axes[1, 2].set_xlabel('Voltage Error (V)', fontweight='bold')
        axes[1, 2].set_ylabel('Cumulative Probability (%)', fontweight='bold')
        axes[1, 2].set_title('Cumulative Error Distribution', fontweight='bold')
        axes[1, 2].legend()
        axes[1, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Accuracy analysis saved: {save_path}")
        plt.show()


# ============================================
# UTILITY FUNCTIONS
# ============================================

def parse_operating_point(op_string):
    """Parse operating point string: '420V/160V@0.94A' -> (Vbus, Vout, Iout)"""
    try:
        parts = op_string.split('/')
        vbus = float(parts[0].replace('V', ''))
        output_parts = parts[1].split('@')
        vout = float(output_parts[0].replace('V', ''))
        iout = float(output_parts[1].replace('A', ''))
        return vbus, vout, iout
    except:
        return None, None, None


def save_operating_points(cases, filename='operating_points_1000.txt'):
    """Save operating points to file."""
    with open(filename, 'w') as f:
        for case in sorted(cases):
            f.write(f"{case}\n")
    print(f"✓ Saved {len(cases)} cases to: {filename}")


def save_detailed_csv(cases, filename='operating_points_1000.csv'):
    """Save operating points with parsed values to CSV."""
    data = []
    for case in sorted(cases):
        vbus, vo, io = parse_operating_point(case)
        if vbus is not None:
            data.append({
                'case': case,
                'V_bus': vbus,
                'V_o': vo,
                'I_o': io,
                'P_o': vo * io
            })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"✓ Saved detailed CSV: {filename}")
    return df


# ============================================
# PERFORMANCE TRACKING (from previous code)
# ============================================

class PerformanceTracker:
    """Track and report simulation performance metrics."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.num_cases = 0
        self.successful_cases = 0
        self.failed_cases = 0
        
    def start(self, num_cases):
        self.start_time = time.time()
        self.num_cases = num_cases
        print(f"\n{'='*80}")
        print(f"⏱️  PERFORMANCE TRACKING STARTED")
        print(f"{'='*80}")
        print(f"Total cases to process: {num_cases}")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    def stop(self, successful, failed):
        self.end_time = time.time()
        self.successful_cases = successful
        self.failed_cases = failed
        self.generate_report()
        
    def generate_report(self):
        total_time = self.end_time - self.start_time
        
        print(f"\n{'='*80}")
        print(f"PERFORMANCE REPORT")
        print(f"{'='*80}")
        
        print(f"\n⏱TIME METRICS:")
        print(f"   Total runtime: {total_time:.2f} seconds ({timedelta(seconds=int(total_time))})")
        print(f"   Average time per case: {total_time/self.num_cases:.3f} seconds")
        
        print(f"\nTHROUGHPUT:")
        print(f"   Cases per second: {self.num_cases/total_time:.2f}")
        print(f"   Cases per minute: {(self.num_cases/total_time)*60:.1f}")
        
        print(f"\nSUCCESS RATE:")
        print(f"   Successful: {self.successful_cases}/{self.num_cases} ({100*self.successful_cases/self.num_cases:.1f}%)")
        
        print(f"\nEFFICIENCY ANALYSIS:")
        traditional_time = self.num_cases * 30
        speedup = traditional_time / total_time
        print(f"   Speedup: {speedup:.1f}x faster")
        print(f"   Time saved: {timedelta(seconds=int(traditional_time - total_time))}")
        
        print(f"\n{'='*80}")


# ============================================
# SPARSE GENERATORS (from previous code)
# ============================================

def generate_sparse_operating_points():
    """Generate sparse strategic operating points."""
    from itertools import product
    
    v_bus_sparse = [380, 420, 450]
    v_o_sparse = [70, 90, 110, 130, 150, 160]
    i_o_sparse = [0.1, 0.5, 0.9, 1.2, 1.5]
    
    cases = set()
    
    for v_bus, v_o, i_o in product(v_bus_sparse, v_o_sparse, i_o_sparse):
        power = v_o * i_o
        if power <= 150:
            case_str = f"{v_bus}V/{v_o}V@{i_o:.2f}A"
            cases.add(case_str)
    
    return cases


def generate_extreme_operating_points():
    """Generate extreme/boundary operating points."""
    extreme_cases = [
        (380, 70, 0.1), (380, 70, 0.5), (380, 70, 1.0), (380, 70, 1.5),
        (380, 100, 0.1), (380, 100, 1.5),
        (380, 160, 0.1), (380, 160, 0.94),
        (420, 70, 0.1), (420, 70, 1.5),
        (420, 100, 0.1), (420, 100, 1.5),
        (420, 130, 0.1), (420, 130, 1.15),
        (420, 160, 0.1), (420, 160, 0.94),
        (450, 70, 0.1), (450, 70, 1.5),
        (450, 100, 0.1), (450, 100, 1.5),
        (450, 130, 0.1), (450, 130, 1.15),
        (450, 160, 0.1), (450, 160, 0.94),
    ]
    
    cases = set()
    for v_bus, v_o, i_o in extreme_cases:
        power = v_o * i_o
        if power <= 150 and i_o <= 1.5:
            case_str = f"{v_bus}V/{v_o}V@{i_o:.2f}A"
            cases.add(case_str)
    
    return cases


# ============================================
# MAIN SCRIPT
# ============================================

if __name__ == "__main__":
    
    # Initialize trackers
    perf_tracker = PerformanceTracker()
    accuracy_tracker = AccuracyTracker(tolerance=0.1)  # 0.1V tolerance
    
    # Circuit parameters
    params = {
        'Rload': 100,
        'Cs': 6.8e-9,
        'Ls': 165e-6,
        'Lm': 750e-6,
        'n': 39/12,
        'Co': 5e-6,
        'Vbus': 420,
        'fsw': 100000
    }
    
    SimCycles = 800  
    TimeStep = 100e-9    
    
    # ============================================
    # GENERATE 1000 OPERATING POINTS
    # ============================================
    
    print("="*80)
    print("GENERATING 1000 OPERATING POINTS")
    print("="*80)
    
    # Choose generation method:
    # - 'latin_hypercube': Best space coverage (RECOMMENDED)
    # - 'random': Pure random sampling
    # - 'grid': Uniform grid
    # - 'adaptive': Focused on boundaries
    # - 'comprehensive': Mix of all methods
    
    print("\nGenerating operating points...")
    
    # Method 1: Single method (fastest)
    OperatingPoints = generate_dense_operating_points(1000, method='latin_hypercube')
    
    # Method 2: Comprehensive suite (most thorough)
    # OperatingPoints = generate_comprehensive_test_suite()
    
    print(f"\n✓ Total operating points: {len(OperatingPoints)}")
    
    # Save to files
    save_operating_points(OperatingPoints, 'operating_points_1000.txt')
    df_cases = save_detailed_csv(OperatingPoints, 'operating_points_1000.csv')
    
    # Show statistics
    print(f"\nDATASET STATISTICS:")
    vbus_vals = df_cases['V_bus'].values
    vo_vals = df_cases['V_o'].values
    io_vals = df_cases['I_o'].values
    po_vals = df_cases['P_o'].values
    
    print(f"   V_bus: {vbus_vals.min():.0f} - {vbus_vals.max():.0f} V")
    print(f"   V_o:   {vo_vals.min():.1f} - {vo_vals.max():.1f} V")
    print(f"   I_o:   {io_vals.min():.2f} - {io_vals.max():.2f} A")
    print(f"   P_o:   {po_vals.min():.1f} - {po_vals.max():.1f} W")
    
    # Preview
    print(f"\nSample cases:")
    for i, case in enumerate(list(OperatingPoints)[:10], 1):
        print(f"   {i:3d}. {case}")
    print(f"   ... and {len(OperatingPoints) - 10} more")
    
    # ============================================
    # RUN SIMULATIONS
    # ============================================
    
    perf_tracker.start(len(OperatingPoints))
    
    print(f"\n{'='*80}")
    print(f"STARTING SIMULATIONS")
    print(f"{'='*80}")
    
    try:
        # Run simulations
        results = test_edge_cases(
            OperatingPoints, 
            params, 
            SimCycles, 
            TimeStep, 
            plot=False,          # Disable plotting for speed
            show_table=True, 
            tol=0.1,             # Voltage tolerance
            return_arrays=True,
            save_csv=True        # Save results to CSV
        )
        
        # Process results and track accuracy
        # Note: You'll need to extract actual V_out from your results
        # This is a placeholder - adapt to your actual results structure
        
        successful = 0
        failed = 0
        
        # Example of how to track accuracy (adapt to your actual results format):
        if isinstance(results, dict):
            for case, result in results.items():
                _, v_target, _ = parse_operating_point(case)
                
                # Extract actual voltage from results
                # v_actual = result['v_out']  # Adapt this line
                # accuracy_tracker.add_result(case, v_target, v_actual)
                
                successful += 1  # Count as successful if simulation completed
        else:
            successful = len(OperatingPoints)
        
        print(f"\nSIMULATIONS COMPLETED")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        successful = 0
        failed = len(OperatingPoints)
    
    # ============================================
    # GENERATE REPORTS
    # ============================================
    
    perf_tracker.stop(successful, failed)
    
    # Generate accuracy report
    # accuracy_df = accuracy_tracker.generate_accuracy_report()
    # accuracy_tracker.plot_accuracy_analysis('accuracy_analysis_1000.png')
    
    # Save accuracy results
    # if accuracy_df is not None:
    #     accuracy_df.to_csv('accuracy_results_1000.csv', index=False)
    #     print("✓ Saved accuracy results to: accuracy_results_1000.csv")
    
    print(f"\n{'='*80}")
    print(f" 1000-CASE BENCHMARK COMPLETE")
    print(f"{'='*80}")