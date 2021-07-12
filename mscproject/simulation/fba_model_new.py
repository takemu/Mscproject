import logging
import multiprocessing
import os
import pandas as pd
from cobra.flux_analysis import pfba
from cobra.io import load_json_model
from cobra.sampling import sample
from pandas import Series

EPSILON = 1e-9
max_flux = 1000
default_uptake = 10
special_uptakes = {'EX_glc__D_e': default_uptake, 'EX_o2_e': 18.5, 'EX_cbl1_e': 0.1}


class FBAModel:
    def __init__(self, model_name='ecoli', objective='biomass', solver='gurobi'):
        self.logger = logging.getLogger(FBAModel.__name__)
        self.logger.setLevel(logging.INFO)

        if model_name == 'ecoli':
            self.model = load_json_model(os.path.dirname(os.path.abspath(__file__)) + '/data/ecoli/iJO1366.json')
            if objective == 'biomass':
                self.model.objective = 'BIOMASS_Ec_iJO1366_core_53p95M'
            self.model.solver = solver
        self._init_model()
        self._print_summary()

        self.reserved_bounds = {}
        self.dummy_reactions = set()

    def _init_model(self):
        for e_react_id, uptake in special_uptakes.items():
            if self.model.reactions.has_id(e_react_id):
                e_reaction = self.model.reactions.get_by_id(e_react_id)
                e_reaction.lower_bound = -uptake

    def solve(self, alg='fba', sampling_n=0, conditions=pd.DataFrame([])):
        self.logger.info('<< Condition: Control >>')
        results = self._calc_fluxes(alg, sampling_n).rename("control").to_frame()
        for _, condition in conditions.iterrows():
            condition = condition.dropna()
            c_name = '-'.join(condition)
            self.logger.info(f'<< Condition: {c_name} >>')
            self._modify_model(condition)
            results = pd.concat([results, self._calc_fluxes(alg, sampling_n).rename(c_name)], axis=1)
            self._revert_model()
        results = results[(results.T > EPSILON).any()]
        results = results.sort_index()
        results = results.fillna(0)
        results.loc["net_flux"] = results.sum()
        return results

    def _calc_fluxes(self, alg='fba', sampling_n=0):
        if sampling_n == 0:
            self.model.solver.problem.write(self.__class__.__name__ + '.lp')
            if alg == 'pfba':
                solution = pfba(self.model)
            else:
                solution = self.model.optimize()
            reversible_fluxes = solution.fluxes
            self.logger.info(f"Objective: {solution.objective_value}")
        else:
            reversible_fluxes = sample(self.model, n=sampling_n, thinning=10,
                                       processes=multiprocessing.cpu_count()).mean(axis=0)
        irreversible_fluxes = {}
        for reaction, flux in reversible_fluxes.iteritems():
            if flux < 0:
                irreversible_fluxes[reaction + '_b'] = -flux
            else:
                irreversible_fluxes[reaction] = flux
        return Series(irreversible_fluxes.values(), index=irreversible_fluxes.keys())

    def _modify_model(self, condition):
        for met_id in condition:
            e_met_id = met_id + '_e'
            if self.model.metabolites.has_id(e_met_id):
                e_react_id = 'EX_' + e_met_id
                if self.model.reactions.has_id(e_react_id):
                    e_reaction = self.model.reactions.get_by_id(e_react_id)
                    self.reserved_bounds[e_reaction.id] = e_reaction.bounds
                    e_reaction.bounds = -default_uptake, max_flux
            else:
                self.logger.warning(f"There is no metabolite {e_met_id}")

    def _revert_model(self):
        for react_id, bounds in self.reserved_bounds.items():
            reaction = self.model.reactions.get_by_id(react_id)
            reaction.bounds = bounds
        self.reserved_bounds = {}

    def _print_summary(self):
        summary = pd.DataFrame(columns=["Number"])
        summary.loc["Constraints"] = len(self.model.constraints)
        summary.loc["Variables"] = len(self.model.variables)
        summary.loc["Metabolites"] = len(self.model.metabolites)
        summary.loc["Reactions"] = len(self.model.reactions)
        self.logger.info(f"\nModel summary:\n{summary.to_markdown()}")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fba_model = FBAModel()
    res = fba_model.solve(alg='pfba')
    # res = fba_model.solve(alg='pfba', batch=pd.read_csv('data/perturbations.csv'))
    res.to_csv('../../output/fba_fluxes_new.csv', float_format='%.6f')
