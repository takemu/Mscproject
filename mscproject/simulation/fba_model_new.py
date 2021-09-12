import logging
import multiprocessing
import os
import pandas as pd
from cobra.exceptions import OptimizationError
from cobra.flux_analysis import pfba
from cobra.io import load_json_model
from cobra.sampling import sample
from optlang.exceptions import SolverError
from optlang.interface import OPTIMAL
from pandas import Series

from pytfa.utils.logger import get_bistream_logger

# EPSILON = 1e-2
max_flux = 1000
default_uptake = 10
special_uptakes = {'EX_glc__D_e': default_uptake, 'EX_o2_e': 18.5, 'EX_cbl1_e': 0.1}


class FBAModel:
    def __init__(self, model_name='ecoli', solver='gurobi', min_biomass=0.55):
        self.logger = get_bistream_logger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        if model_name == 'ecoli':
            self.model = load_json_model(os.path.dirname(os.path.abspath(__file__)) + '/data/ecoli/iJO1366.json')
            self.biomass_reaction = 'BIOMASS_Ec_iJO1366_core_53p95M'
            self.model.objective = self.biomass_reaction
            self.model.reactions.get_by_id(self.biomass_reaction).lower_bound = min_biomass
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

    def solve(self, alg='fba', decimals=2, sampling_n=0, conditions=pd.DataFrame([])):
        self.logger.info('<< Condition: Control >>')
        results = self._calc_fluxes(alg, sampling_n).rename("control").to_frame()
        for _, condition in conditions.iterrows():
            condition = condition.dropna()
            c_name = '-'.join(condition)
            self.logger.info(f'<< Condition: {c_name} >>')
            self._modify_model(condition)
            results = pd.concat([results, self._calc_fluxes(alg, sampling_n).rename(c_name)], axis=1)
            self._revert_model()
        results = results[(results >= 10 ** -decimals).any(axis=1) | (results.index == self.biomass_reaction)]
        results = results.sort_index()
        results = results.fillna(0)
        results.loc["net_flux"] = results.sum()
        results = results.round(decimals=decimals)
        return results

    def _calc_fluxes(self, alg='fba', sampling_n=0):
        try:
            if sampling_n == 0:
                self.model.solver.problem.write(self.__class__.__name__ + '.lp')
                if alg == 'pfba':
                    solution = pfba(self.model)
                else:
                    solution = self.model.optimize()
                if solution.status == OPTIMAL:
                    reversible_fluxes = solution.fluxes
                    self.logger.info(f"Optimal objective: {solution.objective_value:.2f}")
                else:
                    raise OptimizationError(solution.status)
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
        except (AttributeError, SolverError, OptimizationError) as e:
            self.logger.error(f"{str(e).capitalize()} solution!")
            return Series([], dtype=object)

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
    # logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fba_model = FBAModel()
    fba_model.solve().to_csv('output/fba_fluxes.csv')
    # fba_model.solve(decimals=1).to_csv('output/fba_fluxes_0.1.csv')
    fba_model.solve(alg='pfba', conditions=pd.read_csv('data/perturbations.csv')).to_csv(
        'output/pfba_fluxes_batch.csv')
    # fba_model.solve(alg='pfba', decimals=1, conditions=pd.read_csv('data/perturbations.csv')).to_csv(
    #     'output/pfba_fluxes_batch_0.1.csv')
