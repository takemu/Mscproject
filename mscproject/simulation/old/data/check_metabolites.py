import pandas as pd
from cobra.io import load_json_model


class CheckMetabolites:
    def __init__(self):
        self.model = load_json_model('ecoli/iJO1366_old.json')
        self.conditions = pd.read_csv('perturbations.csv')

    def run(self):
        df = pd.DataFrame(columns=('met_id', 'e_met_id', 'p_met_id', 'c_met_id'))
        for _, condition in self.conditions.iterrows():
            condition = condition.dropna()
            for met_id in condition:
                row = [met_id]
                e_met_id = met_id + '_e'
                if self.model.metabolites.has_id(e_met_id):
                    row.append(e_met_id)
                else:
                    row.append('')
                p_met_id = met_id + '_p'
                if self.model.metabolites.has_id(p_met_id):
                    row.append(p_met_id)
                else:
                    row.append('')
                c_met_id = met_id + '_c'
                if self.model.metabolites.has_id(c_met_id):
                    row.append(c_met_id)
                else:
                    row.append('')
                df.loc[len(df), :] = row
        df = df.drop_duplicates()
        df.to_csv('../../../output/check_metabolites.csv', index=False)


if __name__ == '__main__':
    check_metabolites = CheckMetabolites()
    check_metabolites.run()
