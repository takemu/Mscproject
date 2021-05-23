import multiprocessing
import os
import time
import pandas as pd
from cobra import Reaction
from cobra.flux_analysis import pfba
from cobra.io import load_json_model
from cobra.sampling import sample

max_flux = 1000
ex_uptake = 10
growth_reaction_id = 'BIOMASS_Ec_iJO1366_core_53p95M'
# growth_reaction_id = 'BIOMASS_Ec_iML1515_core_75p37M'
dir_path = os.path.dirname(os.path.abspath(__file__))


class FBAModel:
    def __init__(self, model_name='ecoli', solver_name='gurobi', conditions=None, sampling_n=1, p_fba=False):
        start_time = time.time()
        if model_name == 'ecoli':
            self.model = load_json_model(dir_path + '/data/ecoli/iJO1366.json')
            self.model.objective = growth_reaction_id
            self.model.solver = solver_name
        if conditions is None:
            self.conditions = pd.DataFrame()
        else:
            self.conditions = conditions
        self.sampling_n = sampling_n
        self.p_fba = p_fba
        self.reserved_bounds = {}
        self.temp_reactions = set()
        print(f"Build FBA model for {time.time() - start_time:.2f} seconds!")
        self._print_info()

    def solve(self):
        self._init_reaction_bounds()
        print('<< Condition: Control >>')
        results = self._calc_fluxes().rename("control").to_frame()

        for _, condition in self.conditions.iterrows():
            condition = condition.dropna()
            c_name = '-'.join(condition)
            print('<< Condition:', c_name, '>>')
            self._relax_reaction_bounds(condition)
            results = pd.concat([results, self._calc_fluxes().rename(c_name)], axis=1)
            self._reset_reaction_bounds(condition)

        results = results[(results.T != 0).any()]
        results = results.sort_index()
        results = results.fillna(0)
        results.loc["net_flux"] = results.sum()
        return results

    def _calc_fluxes(self):
        start_time = time.time()
        if self.p_fba:
            solution = pfba(self.model)
            fluxes = solution.fluxes
        else:
            solution = self.model.optimize()
            fluxes = solution.fluxes
        print(f"Solve LP for {time.time() - start_time:.2f} seconds!")
        if self.sampling_n > 1:
            start_time = time.time()
            fluxes = sample(self.model, n=self.sampling_n, thinning=100, processes=multiprocessing.cpu_count()).mean(
                axis=0)
            print(f"Run OptGPSampler for {(time.time() - start_time):.2f} seconds!")
        print("Objective:", solution.objective_value)
        self.model.solver.problem.write(self.__class__.__name__ + '.lp')
        return fluxes

    def _init_reaction_bounds(self):
        self.model.reactions.get_by_id('EX_glc__D_e').lower_bound = -ex_uptake
        self.model.reactions.get_by_id('EX_o2_e').lower_bound = -18.5
        self.model.reactions.get_by_id('EX_cbl1_e').lower_bound = -0.1

        for reaction in self.model.reactions:
            if reaction.upper_bound < max_flux:
                self.reserved_bounds[reaction.id] = reaction.upper_bound
            if reaction.reversibility:
                b_reaction = self.add_backward_reaction(reaction)
                if b_reaction.upper_bound < max_flux:
                    self.reserved_bounds[b_reaction.id] = b_reaction.upper_bound

    def add_backward_reaction(self, reaction, temp=False):
        b_react_id = reaction.id + '_b'
        if self.model.reactions.has_id(b_react_id):
            b_reaction = self.model.reactions.get_by_id(b_react_id)
        else:
            b_reaction = Reaction(b_react_id, upper_bound=-reaction.lower_bound)
            b_reaction.add_metabolites({met_id: -coef for met_id, coef in reaction.metabolites.items()})
            self.model.add_reactions([b_reaction])
            if temp:
                self.temp_reactions.add(b_reaction.id)
        reaction.lower_bound = 0
        return b_reaction

    def _relax_reaction_bounds(self, condition):
        for met_id in condition:
            e_met_id = met_id + '_e'
            if self.model.metabolites.has_id(e_met_id):
                e_react_id = 'EX_' + e_met_id
                if self.model.reactions.has_id(e_react_id):
                    e_reaction = self.model.reactions.get_by_id(e_react_id)
                    e_reaction.upper_bound = max_flux
                    b_e_reaction = self.add_backward_reaction(e_reaction, True)
                    b_e_reaction.upper_bound = ex_uptake
            else:
                print('There is no metabolite', e_met_id)

    def _reset_reaction_bounds(self, condition):
        for met_id in condition:
            e_met_id = met_id + '_e'
            self._reset_reaction('EX_' + e_met_id)
            self._reset_reaction('EX_' + e_met_id + '_b')
        self.temp_reactions = set()

    def _reset_reaction(self, react_id):
        if self.model.reactions.has_id(react_id):
            reaction = self.model.reactions.get_by_id(react_id)
            if react_id in self.temp_reactions:
                self.model.remove_reactions([reaction])
            elif react_id in self.reserved_bounds:
                reaction.upper_bound = self.reserved_bounds[react_id]
            else:
                reaction.upper_bound = max_flux

    def _print_info(self):
        print("num constraints", len(self.model.constraints))
        print("num variables", len(self.model.variables))
        print("num metabolites", len(self.model.metabolites))
        print("num reactions", len(self.model.reactions))


if __name__ == '__main__':
    fba_model = FBAModel(conditions=pd.read_csv('data/perturbations.csv'))
    res = fba_model.solve()
    res = res.sort_index()
    res.to_csv('../../output/fba_fluxes.csv')
