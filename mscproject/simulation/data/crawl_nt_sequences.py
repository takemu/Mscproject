import numpy as np
import pandas as pd
import requests

from mscproject.simulation.fba_model import FBAModel

kegg_url = "http://rest.kegg.jp/get/{org}:{gene_name}/ntseq"

ecoli_rnap_genes = pd.Series(['eco:b3295', 'eco:b3649', 'eco:b3987', 'eco:b3988'])
ecoli_rrna_genes = pd.Series(['eco:b3851', 'eco:b3854', 'eco:b3855'])
ecoli_rpeptide_genes = pd.read_csv('ecoli/ribosomal_proteins.tsv', delimiter='\t', header=None).iloc[:, 0]


def get_from_kegg(gene_id):
    org, gene_code, gene_name = gene_id.split(':')
    response = requests.post(kegg_url.format(org=org, gene_name=gene_code))
    if not response.ok:
        response = requests.post(kegg_url.format(org=org, gene_name=gene_name))
    if response.ok:
        eol_ix = response.text.find('\n')
        text = response.text[eol_ix + 1:].replace('\n', '')
        print(gene_id)
        return text
    else:
        return np.nan


# def run(model_code='ecoli:iJO1366'):
#     fba_model = FBAModel(model_code)
#     gene_ids = [gene.id for gene in fba_model.model.genes]
#     gene_ids.sort()
#     with open(f'{fba_model.species}/{fba_model.model_name}_nt_seq_kegg.csv', 'w+') as out_file:
#         for gene_id in gene_ids:
#             print(gene_id)
#             url = 'https://www.genome.jp/entry/-f+-n+n+eco:' + gene_id
#             web_page = urllib.request.urlopen(url)
#
#
#             text = web_page.read().decode("utf8")
#             web_page.close()
#             read_sequence = False
#             sequence = ""
#             for line in text.split('\n'):
#                 if read_sequence:
#                     if '</pre>' in line:
#                         read_sequence = False
#                     else:
#                         sequence += line
#                 else:
#                     if f'<!-- bget:db:genes --><!-- eco:{gene_id} -->' in line:
#                         read_sequence = True
#             out_file.write(gene_id + ',' + sequence + '\n')

def run(model_code='ecoli:iJO1366', database='kegg'):
    fba_model = FBAModel(model_code)
    if fba_model.species == 'ecoli':
        gene_ids = pd.Series([f'eco:{gene.id}:{gene.name}' for gene in fba_model.model.genes])
        gene_ids = pd.concat([gene_ids, ecoli_rnap_genes + ':', ecoli_rrna_genes + ':', ecoli_rpeptide_genes + ':'])
        gene_ids = gene_ids.sort_values()
        gene_ids = gene_ids.drop_duplicates()
    if database == 'kegg':
        nt_sequences = gene_ids.apply(get_from_kegg)
    nt_sequences.index = gene_ids.str.split(':').apply(lambda x: x[1])
    nt_sequences = nt_sequences.dropna()
    nt_sequences.to_csv(f'{fba_model.species}/{fba_model.model_name}_nt_seq_kegg.csv', header=False)


if __name__ == '__main__':
    run()
    # run('ecoli:iML1515')
