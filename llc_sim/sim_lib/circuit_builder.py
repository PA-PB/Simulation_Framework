from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from .transformer import TrxSubCir_CT
from PySpice.Spice.Library import SpiceLibrary

def build_llc_circuit(params: dict, Vtarget=None, config  = 2) -> Circuit:

    """
    Build a full LLC resonant converter circuit using PySpice.

    Parameters
    ----------
    fsw : float
        Switching frequency in Hz.
    Vbus : float
        Input bus voltage [V].
    Cs : float
        Series resonant capacitor value [F].
    Ls : float
        Series resonant inductor value [H].
    Lm : float
        Magnetizing inductor value [H].
    Co : float
        Output capacitor value [F].
    Rload : float
        Load resistance [Ohms].
    n : float
        Transformer turns ratio (primary/secondary).
    Vtarget : float, optional
        Initial target voltage for the output capacitor [V].
    L1 : float, optional
        Transformer primary inductance
    Rc1 : float, optional
        Secondary primary inductance
    Rc2 : float, optional
        Copper resistance
        
    Returns
    -------
    Circuit
        A PySpice `Circuit` object representing the configured LLC converter,
        including input stage, resonant tank, transformer, rectifier, output filter,
        and current probes for key components.
    """

    fsw = params['fsw']
    Vbus = params['Vbus']
    CsValue = params['Cs']
    LsValue = params['Ls']
    LmValue = params['Lm']
    CoValue = params['Co']
    RL = params['Rload']
    n = params['n']
    L1= params.get('L1', 100)
    Rc1= params.get('Rc1', 0.01)
    Rc2= params.get('Rc2', 0.01)

    period = 1 / fsw 
    D = 0.5
    duty_cycle = D * period 
    circuit = Circuit('LLC')

    circuit.model('MyDiode', 'D', RON=.1, ROFF=1e6, VFWD=.5, VREV=2000)

    
    
    circuit.model('Switch', 'SW',
              Ron=0.02,     
              Roff=1e9,     
              Vt=0.5, Vh=0.4)


    if config == 1:

        circuit.V(1, 'N001', circuit.gnd, Vbus)
       
        circuit.VCS(1, 'N001', 'ISw1', 'N003', circuit.gnd, model='Switch', initial_state='off')
        circuit.V('Sw1','vab','ISw1', 0)
        circuit.Diode('Sw1','ISw1','N001',model='MyDiode')
        circuit.C('Sw1','N001','ISw1',100e-15)
        circuit.VCS(2, 'vab',circuit.gnd, 'N004', circuit.gnd, model='Switch', initial_state='on')
        circuit.Diode('Sw2',circuit.gnd,'vab',model='MyDiode')
        circuit.C('Sw2','vab',circuit.gnd,100e-15)

    
        dead = 100e-9

        Ton_top = max(duty_cycle - 2*dead, 0.0)  # safety clamp

        circuit.PulseVoltageSource('pulse', 'N003', circuit.gnd,
            initial_value=0, pulsed_value=5,
            delay_time=dead,
            pulse_width=Ton_top,
            period=period,
            rise_time=100e-9, fall_time=100e-9)


        low_width = max(duty_cycle + dead, 0.0)

        circuit.PulseVoltageSource('pulse_comp', 'N004', circuit.gnd,
            initial_value=5, pulsed_value=0,
            delay_time=0,
            pulse_width=low_width,
            period=period,
            rise_time=100e-9, fall_time=100e-9)

        
        circuit.C('s', 'vab', 'cs', CsValue)
        circuit.L('s', 'cs', 'pri', LsValue)
        circuit.L('m', 'pri', circuit.gnd, LmValue)

        trx = TrxSubCir_CT('TRX_LLC', turn_ratio=n, primary_inductance=L1,
                       copper_resistance_primary=Rc1, copper_resistance_secondary=Rc2)
        circuit.subcircuit(trx)
        circuit.X(1, 'TRX_LLC', 'pri', circuit.gnd, 'secn1', circuit.gnd, 'secn2')

    
        circuit.Diode(1, 'secn1', 'vo', model='MyDiode')
        circuit.Diode(2, 'secn2', 'vo', model='MyDiode')
        circuit.C('o', 'vo', circuit.gnd, CoValue)
        circuit.R('load', 'vo', circuit.gnd, RL)
     
    else:
        circuit.PulseVoltageSource("Ideal", 'vab', circuit.gnd,
                               initial_value=0,
                               pulsed_value=Vbus,
                               pulse_width=1/(2*fsw),
                               period=1/fsw,
                               rise_time=20e-9,
                               fall_time=20e-9)
    
        circuit.C('s', 'vab', 'cs', CsValue)
        circuit.L('s', 'cs', 'pri', LsValue)
        circuit.L('m', 'pri', circuit.gnd, LmValue)

        trx = TrxSubCir_CT('TRX_LLC', turn_ratio=n, primary_inductance=L1,
                       copper_resistance_primary=Rc1, copper_resistance_secondary=Rc2)
        circuit.subcircuit(trx)
        circuit.X(1, 'TRX_LLC', 'pri', circuit.gnd, 'secn1', circuit.gnd, 'secn2')

    
        circuit.Diode(1, 'secn1', 'vo', model='MyDiode')
        circuit.Diode(2, 'secn2', 'vo', model='MyDiode')
        circuit.C('o', 'vo', circuit.gnd, CoValue)
        circuit.R('load', 'vo', circuit.gnd, RL)
        


    circuit.Ls.plus.add_current_probe(circuit)
    circuit.Lm.plus.add_current_probe(circuit)
    circuit.Rload.plus.add_current_probe(circuit)
    circuit.D1.plus.add_current_probe(circuit)
    circuit.D2.plus.add_current_probe(circuit)
    circuit.Co.plus.add_current_probe(circuit)

    



    
    return circuit


