def simulate(circuit, fsw, TimeStep, SimCycles, TimeStartSave = 0 , show_node_names = False):

    """
    Run a transient simulation of an LLC converter circuit.

    Parameters
    ----------
    circuit : Circuit
        PySpice Circuit object to be simulated.
    fsw : float
        Switching frequency in Hz.
    TimeStep : float
        Simulation time step in seconds.
    SimCycles : int
        Number of switching cycles to simulate.
    TimeStartSave : float, optional
        Time to start saving results in simulation cycles(default is 0).
    show_node_names : bool, optional
        If True, prints the available node names (default is False).

    Returns
    -------
    analysis : SimulationResult
        Result object containing all node voltages and currents from the simulation.
    """


    TimeEnd = SimCycles * 1 / fsw
    TimeStartSave = TimeStartSave * 1/fsw

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    simulator.options(
            method='gear',      
            maxord=3,          
            reltol=1e-3,        # Relative tolerance
            abstol=1e-9,        # Current absolute tolerance (A)
            vntol=1e-6,         # Voltage absolute tolerance (V)
            chgtol=1e-16,       # Helps with capacitor charge bookkeeping
            itl4=500,           # Newton iterations per timepoint
            plotwinsize=0,       # Disable data compression: keeps raw time points exact
        )

    simulator.initial_condition(vo=0.1)
    analysis = simulator.transient(step_time=TimeStep, end_time=TimeEnd, start_time=TimeStartSave)

    if show_node_names:
        print("📍 Available Node Names:")
        for key in analysis.nodes:
            print(f"{key}")

    return analysis