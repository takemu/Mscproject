import os

import pandas as pd
from cobra.io import load_json_model


def analyze(name='ptfa', model='iML1515', threshold=20):
    coefs = pd.read_csv(f'output/{name}_coefs.csv', index_col=0)
    coefs = (coefs - coefs.min()) / (coefs.max() - coefs.min()) * 100
    coefs = coefs[(coefs > threshold).all(axis=1)]
    coefs.to_csv(f'output/{name}_results.csv')
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
    results.to_csv(f'output/{name}_results.csv')
    results['kegg.reaction'].to_csv(f'output/{name}_results.txt', sep=' ', index=False, header=False)


if __name__ == '__main__':
    analyze('yangs_lr', 'iJO1366')
    analyze('ptfa_lr')
    analyze('ptfa_mlp')
