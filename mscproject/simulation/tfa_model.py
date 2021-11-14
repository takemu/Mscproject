from os.path import join

import pandas as pd
from optlang.exceptions import SolverError

from mscproject.simulation.data import data_dir
from mscproject.simulation.fba_model import FBAModel, pfba_biomass
from pytfa import ThermoModel
from pytfa.io import load_thermoDB, read_lexicon, read_compartment_data, annotate_from_lexicon, apply_compartment_data
from pytfa.optim import relax_dgo


class TFAModel(FBAModel):
    def __init__(self, model_code='ecoli:iJO1366', solver='gurobi', min_biomass=0.55):
        super().__init__(model_code, solver, min_biomass)
        if self.species == 'ecoli':
            # self.model.reactions.EX_glc__D_e.lower_bound = -1 * glc_uptake - glc_uptake_std
            # self.model.reactions.EX_glc__D_e.upper_bound = -1 * glc_uptake + glc_uptake_std
            thermo_db = load_thermoDB(join(data_dir, 'thermo/thermo_data.thermodb'))
            self.model = ThermoModel(thermo_db, self.model)
            self.model.name = self.model_name
            self.model.sloppy = True
            apply_compartment_data(self.model, read_compartment_data(join(data_dir, 'thermo/compartment_data.json')))
            self.apply_annotation_data()
            self.model.prepare()
            self.model.convert()
            # self.model.reactions.get_by_id(self.objective).lower_bound = objective_lb
            self.model.repair()
            try:
                self.model.optimize()
            except (AttributeError, SolverError):
                self.model, _, _ = relax_dgo(self.model, in_place=True)
                # self.model.reactions.get_by_id(self.objective).lower_bound = 0
            self.model.print_info()

    def apply_annotation_data(self):
        # for met in self.model.metabolites:
        #     if 'seed.compound' in met.annotation:
        #         met.annotation = {'seed_id': met.annotation['seed.compound'][0]}
        annotate_from_lexicon(self.model, read_lexicon(join(data_dir, 'thermo/lexicon.csv')))
        # for met in self.model.metabolites:
        #     print(met.annotation)


if __name__ == '__main__':
    tfa_model = TFAModel(model_code='ecoli:iML1515', min_biomass=0.1)
    # tfa_model.solve(alg='pfba', conditions=pd.read_csv('data/perturbations.csv')).to_csv('output/ptfa_fluxes.csv')
    # min_biomass = [0.1 * bm for bm in pfba_biomass]
    # print(min_biomass)
    tfa_model.solve(alg='pfba', conditions=pd.read_csv('data/glc_uptakes.csv')).to_csv(
        'output/glc_ptfa_fluxes.csv')
