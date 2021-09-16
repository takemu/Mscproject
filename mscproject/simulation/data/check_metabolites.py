import pandas as pd

from mscproject.simulation.fba_model import FBAModel


def run(model_code='ecoli:iJO1366'):
    model = FBAModel(model_code).model
    conditions = pd.read_csv('perturbations.csv')

    df = pd.DataFrame(columns=('met_id', 'e_met_id', 'p_met_id', 'c_met_id'))
    for _, condition in conditions.iterrows():
        condition = condition.dropna()
        for met_id in condition:
            row = [met_id]
            e_met_id = met_id + '_e'
            if model.metabolites.has_id(e_met_id):
                row.append(e_met_id)
            else:
                row.append('-')
            p_met_id = met_id + '_p'
            if model.metabolites.has_id(p_met_id):
                row.append(p_met_id)
            else:
                row.append('-')
            c_met_id = met_id + '_c'
            if model.metabolites.has_id(c_met_id):
                row.append(c_met_id)
            else:
                row.append('-')
            df.loc[len(df), :] = row
    df = df.drop_duplicates()
    model_code = model_code.replace(':', '_')
    df.to_csv(f'output/{model_code}_check_metabolites.csv', index=False)


if __name__ == '__main__':
    run()
    run('ecoli:iML1515')
