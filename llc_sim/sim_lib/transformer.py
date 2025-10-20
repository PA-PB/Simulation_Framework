from PySpice.Spice.Netlist import SubCircuit

class TrxSubCir_CT(SubCircuit):
            __nodes__= ('input_plus', 'input_minus','output_s1', 'output_gnd', 'output_s2')
            def __init__(self, name, turn_ratio, primary_inductance = 100, copper_resistance_primary = 0.01, copper_resistance_secondary = 0.01):

                SubCircuit.__init__(self,name, *self.__nodes__)
                secondary_inductance = primary_inductance / float(turn_ratio**2)

                # Primary
                self.R('primary', 'input_plus', 1, copper_resistance_primary)
                primary_inductor = self.L('primary', 1, 'input_minus', primary_inductance)
                
                # Secondary S1
                self.R('secondary_s1', 2, 'output_s1', copper_resistance_secondary)
                secondary_inductor_s1 = self.L('secondary_s1', 2, 'output_gnd', secondary_inductance)

                # Secondary S2
                self.R('secondary_s2',3,'output_s2', copper_resistance_secondary)
                secondary_inductor_s2 = self.L('secondary_s2', 'output_gnd', 3, secondary_inductance)       
                
                # Coupling: 
                self.raw_spice='Kcoupling Lprimary Lsecondary_s1 1\nKcoupling2 Lprimary Lsecondary_s2 1\nKcoupling3 Lsecondary_s1 Lsecondary_s2 1'
                
                return




