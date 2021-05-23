import time

from mscproject.simulation.fba_model import FBAModel, dir_path
from pytfa import ThermoModel
from pytfa.io import load_thermoDB, read_lexicon, read_compartment_data, annotate_from_lexicon, apply_compartment_data


class TFAModel(FBAModel):
    def __init__(self, model_name='ecoli', solver_name='gurobi', conditions=None, pfba=False, sampling_n=0):
        start_time = time.time()
        super().__init__(model_name, solver_name, conditions, sampling_n, pfba)
        if model_name == 'ecoli':
            self.model = ThermoModel(load_thermoDB(dir_path + '/data/ecoli/thermo/thermo_data.thermodb'), self.model)
            annotate_from_lexicon(self.model, read_lexicon(dir_path + '/data/ecoli/thermo/lexicon.csv'))
            compartment_data = read_compartment_data(dir_path + '/data/ecoli/thermo/compartment_data.json')
            apply_compartment_data(self.model, compartment_data)
            self.model.prepare()
            self.model.convert()
            self.model.print_info()
        print(f"Build TFA model for {time.time() - start_time:.2f} seconds!")


if __name__ == '__main__':
    tfa_model = TFAModel()
    solution = tfa_model.solve()
