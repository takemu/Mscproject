import logging
import multiprocessing
import os

import numpy as np
import pandas as pd
from cobra.exceptions import OptimizationError
from cobra.flux_analysis import pfba
from cobra.io import load_json_model
from cobra.sampling import sample
from optlang.exceptions import SolverError
from optlang.interface import OPTIMAL
from pandas import Series

from pytfa.utils.logger import get_bistream_logger

max_flux = 1000
default_uptake = 10
special_uptakes = {'EX_glc__D_e': default_uptake, 'EX_o2_e': 18.5, 'EX_cbl1_e': 0.1}

pfba_biomass = [0, 0.15, 0.25, 0.34, 0.43, 0.52, 0.61, 0.7, 0.75, 0.78, 0.82, 0.85, 0.89, 0.92, 0.94]


def reformat_condition(condition):
    met_ids = []
    met_conditions = []
    i = 0
    for _, value in condition.items():
        if i % 2 == 1:
            if isinstance(met_id, str):
                met_ids.append(met_id)
                if np.isnan(value):
                    value = default_uptake
                met_conditions.append((met_id, value))
        else:
            met_id = value
        i += 1
    return '-'.join(met_ids), met_conditions


class FBAModel:
    def __init__(self, model_code='ecoli:iJO1366', solver='gurobi', min_biomass=0.55):
        self.logger = get_bistream_logger((model_code + ':' + self.__class__.__name__).replace(':', '_'))
        self.logger.setLevel(logging.INFO)
        self.species = model_code.split(':')[0]
        self.model_name = model_code.split(':')[-1]
        if self.species.lower() == 'ecoli':
            if self.model_name.lower() == 'iml1515':
                self.model = load_json_model(os.path.dirname(os.path.abspath(__file__)) + '/data/ecoli/iML1515.json')
                self.biomass_reaction = 'BIOMASS_Ec_iML1515_core_75p37M'
            else:
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

    def solve(self, alg='fba', decimals=2, sampling_n=0, conditions=pd.DataFrame([]), min_biomass=[]):
        if len(min_biomass) > 0:
            assert (conditions.shape[0] == len(min_biomass))
        self.logger.info('<< Condition: Control >>')
        results = self._calc_fluxes(alg, sampling_n).rename("control").to_frame()
        conditions = conditions.apply(reformat_condition, axis=1)
        for i, (c_name, condition) in conditions.items():
            # c_name = combined_condition[0]
            self.logger.info(f'<< Condition: {c_name} >>')
            if len(min_biomass) > 0:
                self.model.reactions.get_by_id(self.biomass_reaction).lower_bound = min_biomass[i]
                self.logger.info(f'Biomass LB: {min_biomass[i]}')
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
            self.logger.error(f'{str(e).capitalize()}')
            return Series([], dtype=object)

    def _modify_model(self, condition):
        for met_id, value in condition:
            e_met_id = met_id + '_e'
            if self.model.metabolites.has_id(e_met_id):
                e_react_id = 'EX_' + e_met_id
                if self.model.reactions.has_id(e_react_id):
                    e_reaction = self.model.reactions.get_by_id(e_react_id)
                    self.reserved_bounds[e_reaction.id] = e_reaction.bounds
                    e_reaction.bounds = -value, max_flux
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
    fba_model = FBAModel(model_code='ecoli:iML1515', min_biomass=0.1)
    # fba_model.solve().to_csv('output/fba_fluxes.csv')
    # fba_model.solve(alg='pfba', conditions=pd.read_csv('data/perturbations.csv')).to_csv('output/pfba_fluxes.csv')
    fba_model.solve(alg='pfba', conditions=pd.read_csv('data/glc_uptakes.csv')).to_csv('output/glc_pfba_fluxes.csv')

    # fba_model = FBAModel(model_code='ecoli:iJO1366', min_biomass=0.1)
    # fba_model.solve(alg='pfba', conditions=pd.read_csv('data/perturbations.csv')).to_csv('output/pfba_iJO1366_fluxes.csv')
    # fba_model.solve(alg='pfba', conditions=pd.read_csv('data/glc_uptakes.csv')).to_csv('output/glc_pfba_iJO1366_fluxes.csv')
