import os

import pandas as pd
from cobra.io import load_json_model


def analyze(name='ptfa_mlp', model='iML1515', threshold=0):
    coefs = pd.read_csv(f'output/{name}_coefs.csv', index_col=0)
    # coefs = (coefs - coefs.min()) / (coefs.max() - coefs.min()) * 100
    coefs = coefs[(coefs >= threshold).all(axis=1)]
    model = load_json_model(os.path.dirname(os.path.abspath(__file__)) + f'/../simulation/data/ecoli/{model}.json')

    reactions = []
    for react_id in coefs.index:
        score = coefs.loc[react_id, 'All_IC50']
        if react_id.endswith('_b'):
            react_id = react_id[:-2]
        if model.reactions.has_id(react_id):
            reaction = model.reactions.get_by_id(react_id)
            subsystem = reaction.subsystem
            subsystem = subsystem.replace(',', ':')
            kegg_react_id = ''
            biocyc_react_id = ''
            if 'kegg.reaction' in reaction.annotation:
                kegg_react_id = reaction.annotation['kegg.reaction'][0]
            if 'biocyc' in reaction.annotation:
                biocyc_react_id = reaction.annotation['biocyc'][0][5:]
            reactions.append([react_id, subsystem, kegg_react_id, biocyc_react_id, score])

    results = pd.DataFrame(reactions, columns=['id', 'subsystem', 'kegg.reaction', 'biocyc', 'score'])
    results = results[~((results['kegg.reaction'] == '') | (results['biocyc'] == ''))].set_index('id')
    results = results[~results.index.duplicated()]
    results.to_csv(f'output/reactions/{name}_scores.csv')
    results['kegg.reaction'].to_csv(f'output/reactions/{name}_kegg.txt', sep=' ', index=False, header=False)
    results['biocyc'].to_csv(f'output/reactions/{name}_biocyc.txt', sep=' ', index=False, header=False)


if __name__ == '__main__':
    analyze(name='yangs_lr', model='iJO1366')
    # analyze(name='etfl_lr')
    analyze(name='etfl_mlp')
