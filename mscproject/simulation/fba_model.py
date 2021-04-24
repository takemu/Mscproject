from etfl.data.ecoli import get_model
from pytfa.utils.logger import get_timestr
from mscproject.simulation.utils import glc_uptake, glc_uptake_std

solver = 'optlang-gurobi'


def make_fba_model():
    ecoli = get_model(solver)
    ecoli.reactions.EX_glc__D_e.lower_bound = -1 * glc_uptake - glc_uptake_std
    ecoli.reactions.EX_glc__D_e.upper_bound = -1 * glc_uptake + glc_uptake_std

    # ecoli.objective = growth_reaction_id
    ecoli.optimize()

    from cobra.io.json import save_json_model

    save_json_model(ecoli, 'models/iJO1366_T0E0N0_{}.json'.format(get_timestr()))


if __name__ == '__main__':
    make_fba_model()
