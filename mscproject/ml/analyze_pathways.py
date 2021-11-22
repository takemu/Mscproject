import pandas as pd


def analyze(name='ptfa_mlp'):
    react_pathways = pd.read_csv(f'data/ecocyc/{name}_pathways.csv', sep='\t', index_col=0)
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
    reaction_scores = pd.read_csv(f'output/results/{name}_results.csv', index_col=0)
    for pathway, reactions in pathway_reactions.items():
        score = 0
        for reaction in list(reactions):
            score += reaction_scores[reaction_scores['biocyc'] == reaction]['score'].tolist()[0]
        # score /= len(list(reactions))
        pathway_scores[pathway] = score
    results = pd.DataFrame.from_dict(pathway_scores, orient='index', columns=['score'])
    results = results.round(decimals=2)
    results = results.sort_values(by='score', ascending=False)
    results.to_csv(f'output/results/{name}_pathway_scores.csv')


if __name__ == '__main__':
    analyze('ptfa_mlp')
