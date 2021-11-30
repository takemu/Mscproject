import pandas as pd


def analyze(name='ptfa_mlp', method='sum'):
    react_pathways = pd.read_csv(f'data/ecocyc/{name}_pathways.txt', sep='\t', index_col=0)
    react_pathways = react_pathways.fillna('')
    pathway_reactions = {}
    for index, row in react_pathways.iterrows():
        for field in row:
            if field != '':
                pathways = field.split(' // ')
                for pathway in pathways:
                    if pathway not in pathway_reactions:
                        pathway_reactions[pathway] = set([index])
                    else:
                        pathway_reactions[pathway].add(index)
    pathway_scores = {}
    reaction_scores = pd.read_csv(f'output/reactions/{name}_scores.csv', index_col=0)
    kegg_reactions = []
    for pathway, reactions in pathway_reactions.items():
        score = 0
        for reaction in list(reactions):
            score += reaction_scores[reaction_scores['biocyc'] == reaction]['score'].tolist()[0]
        if method == 'mean':
            score /= len(list(reactions))
        pathway_scores[pathway] = score
        for reaction in list(reactions):
            kegg_reaction = reaction_scores.loc[reaction_scores['biocyc'] == reaction, 'kegg.reaction'][0]
            kegg_reactions.append([pathway, score, kegg_reaction])
    results = pd.DataFrame.from_dict(pathway_scores, orient='index', columns=['score'])
    results = results.round(decimals=2)
    results = results.sort_values(by='score', ascending=False)
    results.to_csv(f'output/pathways/{name}_{method}_scores.csv')

    results2 = pd.DataFrame(kegg_reactions, columns=['pathway', 'score', 'reaction'])
    results2 = results2.sort_values(by='score', ascending=False)
    results2 = results2.reset_index(drop=True)
    results2.to_csv(f'output/pathways/{name}_kegg_reactions.csv')


if __name__ == '__main__':
    # analyze('yangs_lr', method='mean')
    # analyze('etfl_lr', method='sum')
    analyze('etfl_mlp', method='sum')
    analyze('etfl_mlp', method='mean')
