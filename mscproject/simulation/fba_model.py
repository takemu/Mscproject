import json
import multiprocessing
import os
import time
import pandas as pd
from cobra.flux_analysis import pfba
from cobra.io import load_json_model
from cobra.sampling import sample

ex_uptake = 1000
growth_reaction_id = 'BIOMASS_Ec_iJO1366_core_53p95M'
dir_path = os.path.dirname(os.path.abspath(__file__))


class FBAModel:
    def __init__(self, model_name='ecoli', solver_name='gurobi', conditions=None, sampling_n=1, p_fba=False):
        start_time = time.time()
        if model_name == 'ecoli':
            self.model = load_json_model(dir_path + '/data/ecoli/iJO1366.json')
            self.model.objective = growth_reaction_id
            self.model.solver = solver_name
            self.reaction_ids = pd.read_csv(dir_path + '/data/ecoli/ex_reactions.csv', header=None)[0].tolist()
        if conditions is None:
            self.conditions = pd.DataFrame()
        else:
            self.conditions = conditions
        self.sampling_n = sampling_n
        self.p_fba = p_fba
        self._print_info()
        print(f"Build FBA model for {time.time() - start_time:.2f} seconds!")
        self.model.solver.problem.write('fba.lp')

    def solve(self):
        print('<< Condition: Control >>')
        self._reset_reaction_bounds()
        results = self._get_fluxes().rename("control").to_frame()

        for i in range(len(self.conditions)):
            condition = self.conditions.iloc[i, :].dropna().tolist()
            column_name = '-'.join(condition)
            print('<< Condition:', column_name, '>>')
            self._reset_reaction_bounds()
            for metabolite in condition:
                reaction_id = 'EX_' + metabolite + '_e'
                self._relax_reaction_bounds(reaction_id, ex_uptake)
            results[column_name] = self._get_fluxes()

        # results = results[(results.T != 0).any()]
        return results

    def _get_fluxes(self):
        start_time = time.time()
        if self.p_fba:
            fluxes = pfba(self.model).fluxes
        else:
            fluxes = self.model.optimize().fluxes
        print(f"Solve LP for {time.time() - start_time:.2f} seconds!")
        if self.sampling_n > 1:
            start_time = time.time()
            fluxes = sample(self.model, n=self.sampling_n, thinning=100, processes=multiprocessing.cpu_count()).mean(
                axis=0)
            print(f"Run OptGPSampler for {(time.time() - start_time):.2f} seconds!")
        return fluxes

    def _reset_reaction_bounds(self):
        for reaction_id in self.reaction_ids:
            reaction = self.model.reactions.get_by_id(reaction_id)
            # reaction.bounds = 0, 0

    def _relax_reaction_bounds(self, reaction_id, upper_bound):
        try:
            reaction = self.model.reactions.get_by_id(reaction_id)
            reaction.upper_bound = upper_bound
            print('Relax bounds of reaction', reaction.id, 'to', reaction.bounds)
        except KeyError:
            print("There is no reaction " + reaction_id + '!')

    def _print_info(self):
        print("num constraints", len(self.model.constraints))
        print("num variables", len(self.model.variables))
        print("num metabolites", len(self.model.metabolites))
        print("num reactions", len(self.model.reactions))


if __name__ == '__main__':
    fba_model = FBAModel()
    res = fba_model.solve()
