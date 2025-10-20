from llc_sim import build_llc_circuit, simulate, analyze, test_edge_cases, find_vtarget

import matplotlib.pyplot as plt

params = {

    'Rload': 700,
    'Cs': 6.8e-9,
    'Ls': 150e-6,
    'Lm': 600e-6,
    'n': 2,
    'Co': 10e-6,
    'Vbus': 410,
    'fsw':100e3
}


SimCycles = 800
TimeStep = 100e-9


#Example syntax for standalone function use
'''
circuit = build_llc_circuit(params, config=1)
analysis = simulate(circuit, params['fsw'], TimeStep, SimCycles)
results = analyze(analysis, params['fsw'], TimeStep, SimCycles)
plt.plot(analysis.time,analysis['vo'])
plt.show() 

'''
test = {
    f'{params['Vbus']}V/100V@1.5A',
    f'{params['Vbus']}V/100V@0.1A',
    f'{params['Vbus']}V/200V@0.1A',
    f'{params['Vbus']}V/200V@1.25A',
    
}


#test_edge_cases syntax

if __name__ == "__main__":
    test_edge_cases(test, params, SimCycles, TimeStep, show_table=True, tol=0.1) 

