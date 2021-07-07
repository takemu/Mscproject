import time

from optlang.exceptions import SolverError

from mscproject.simulation.fba_model import FBAModel, dir_path, growth_reaction_id
from pytfa import ThermoModel
from pytfa.io import load_thermoDB, read_lexicon, read_compartment_data, annotate_from_lexicon, apply_compartment_data
from pytfa.optim import relax_dgo

glc_uptake = 7.54
glc_uptake_std = 0.56
observed_growth_std = 0.02
observed_growth = 0.61


class TFAModel(FBAModel):
    def __init__(self, model_name='ecoli', solver_name='gurobi', conditions=None, pfba=False):
        start_time = time.time()
        super().__init__(model_name, solver_name, conditions, pfba)
        if model_name == 'ecoli':
            # self.model.reactions.EX_glc__D_e.lower_bound = -1 * glc_uptake - glc_uptake_std
            # self.model.reactions.EX_glc__D_e.upper_bound = -1 * glc_uptake + glc_uptake_std
            thermo_db = load_thermoDB(dir_path + '/data/thermo/thermo_data.thermodb')
            self.model = ThermoModel(thermo_db, self.model)
            self.model.name = 'iJO1366_TFA'
            self.model.sloppy = True
            apply_compartment_data(self.model, read_compartment_data(dir_path + '/data/thermo/compartment_data.json'))
            self.apply_annotation_data()
            self.model.prepare()
            self.model.convert()
            # self.model.reactions.get_by_id(
            #     growth_reaction_id).lower_bound = observed_growth - 1 * observed_growth_std
            self.model.repair()
            try:
                self.model.optimize()
            except (AttributeError, SolverError):
                self.model, _, _ = relax_dgo(self.model, in_place=True)
                self.model.reactions.get_by_id(growth_reaction_id).lower_bound = 0
            print(f"Build TFA model for {time.time() - start_time:.2f} seconds!")
            self.model.print_info()

    def apply_annotation_data(self):
        for met in self.model.metabolites:
            if 'seed.compound' in met.annotation:
                met.annotation = {'seed_id': met.annotation['seed.compound'][0]}
        annotate_from_lexicon(self.model, read_lexicon(dir_path + '/data/thermo/lexicon.csv'))
        # for met in self.model.metabolites:
        #     print(met.annotation)


if __name__ == '__main__':
    tfa_model = TFAModel()
    res = tfa_model.solve()
    res.to_csv('../../output/tfa_fluxes.csv', float_format='%.3f')
