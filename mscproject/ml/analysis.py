import os

import pandas as pd
from cobra.io import load_json_model


def analyze(data='our', model='iML1515'):
    coefs = pd.read_csv(f'output/{data}_coefs.csv', index_col=0)
    coefs = (coefs - coefs.min()) / (coefs.max() - coefs.min()) * 100
    coefs = coefs[(coefs > 0).any(axis=1)]
    coefs.to_csv(f'output/{data}_results.csv')
    model = load_json_model(os.path.dirname(os.path.abspath(__file__)) + f'/../simulation/data/ecoli/{model}.json')

    reactions = []
    for react_id in coefs.index:
        if react_id.endswith('_b'):
            react_id = react_id[:-2]
        reaction = model.reactions.get_by_id(react_id)
        subsystem = reaction.subsystem
        subsystem = subsystem.replace(',', ':')
        kegg_react_id = ''
        if 'kegg.reaction' in reaction.annotation:
            kegg_react_id = reaction.annotation['kegg.reaction'][0]
        reactions.append([react_id, kegg_react_id, subsystem])

    results = pd.DataFrame(reactions, columns=['id', 'kegg.reaction', 'subsystem'])
    results = results[~(results['kegg.reaction'] == '')].set_index('id')
    results = results[~results.index.duplicated()]
    results.to_csv(f'output/{data}_results.csv')


if __name__ == '__main__':
    analyze()
    analyze('yangs', 'iJO1366')
