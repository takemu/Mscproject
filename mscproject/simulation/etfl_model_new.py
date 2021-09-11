import logging
import os
import time

from cobra import Metabolite, Reaction

from etfl.core.allocation import add_interpolation_variables, add_protein_mass_requirement, add_rna_mass_requirement, \
    add_dna_mass_requirement
from fba_model_new import FBAModel
from pytfa.io import load_thermoDB, read_lexicon, annotate_from_lexicon, read_compartment_data, apply_compartment_data
from pytfa.optim.relaxation import relax_dgo
from etfl.core import ThermoMEModel, MEModel
from pytfa.utils.logger import get_timestr
from etfl.data.ecoli import get_coupling_dict, get_mrna_dict, get_rib, get_rnap, get_monomers_dict, get_nt_sequences, \
    get_ratios, get_neidhardt_data, get_mrna_metrics, get_enz_metrics, remove_from_biomass_equation, \
    get_ecoli_gen_stats, get_essentials
from optlang.exceptions import SolverError

# McCloskey2014 values
glc_uptake = 7.54
glc_uptake_std = 0.56
observed_growth_std = 0.02
observed_growth = 0.61
mu_range = [0, 3.5]
n_mu_bins = 128
growth_reaction_id = 'BIOMASS_Ec_iJO1366_core_53p95M'
dir_path = os.path.dirname(os.path.abspath(__file__))


class ETFLModel(FBAModel):
    def __init__(self, model_name='ecoli', objective='biomass', solver='gurobi', objective_lb=0.1):
        # start_time = time.time()
        super().__init__(model_name, objective, solver)
        if model_name == 'ecoli':
            # Add cystein -> selenocystein transformation for convenience
            selcys = Metabolite(id='selcys__L_c', compartment='c', formula='C3H7NO2Se')
            selcys_rxn = Reaction(id='PSEUDO_selenocystein_synthase', name='PSEUDO Selenocystein_Synthase')
            selcys_rxn.add_metabolites({self.model.metabolites.cys__L_c: -1, selcys: +1})
            self.model.add_reactions([selcys_rxn])

            self._sanitize_varnames()
            # self.model.reactions.EX_glc__D_e.lower_bound = -1 * glc_uptake - glc_uptake_std
            # self.model.reactions.EX_glc__D_e.upper_bound = -1 * glc_uptake + glc_uptake_std

            # time_str = get_timestr()
            coupling_dict = get_coupling_dict(self.model, mode='kmax', atps_name='ATPS4rpp', infer_missing_enz=True)
            aa_dict, rna_nucleotides, rna_nucleotides_mp, dna_nucleotides = get_monomers_dict()
            essentials = get_essentials()

            # if has_thermo:
            thermo_db = load_thermoDB(dir_path + '/data/thermo/thermo_data.thermodb')
            self.model = ThermoMEModel(thermo_db,
                                       model=self.model,
                                       growth_reaction=growth_reaction_id, mu_range=mu_range,
                                       n_mu_bins=n_mu_bins)
            self.model.name = 'iJO1366_ETFL'
            # annotate_from_lexicon(self.model, read_lexicon(dir_path + '/data/thermo/lexicon.csv'))
            # compartment_data = read_compartment_data(dir_path + '/data/thermo/compartment_data.json')
            # apply_compartment_data(self.model, compartment_data)
            apply_compartment_data(self.model, read_compartment_data(dir_path + '/data/thermo/compartment_data.json'))
            annotate_from_lexicon(self.model, read_lexicon(dir_path + '/data/thermo/lexicon.csv'))
            self.model.prepare()
            # self.model.reactions.MECDPS.thermo['computed'] = False
            # self.model.reactions.NDPK4.thermo['computed'] = False
            # self.model.reactions.TMDPP.thermo['computed'] = False
            # self.model.reactions.ARGAGMt7pp.thermo['computed'] = False
            self.model.convert()
            # else:
            #     self.model = MEModel(model=self.model, growth_reaction=growth_reaction_id, mu_range=mu_range,
            #                          n_mu_bins=n_mu_bins, name=name)
            mrna_dict = get_mrna_dict(self.model)
            nt_sequences = get_nt_sequences()
            rnap = get_rnap()
            rib = get_rib()

            # Remove nucleotides and amino acids from biomass reaction as they will be
            # taken into account by the expression
            remove_from_biomass_equation(model=self.model, nt_dict=rna_nucleotides, aa_dict=aa_dict,
                                         essentials_dict=essentials)

            self.model.add_nucleotide_sequences(nt_sequences)
            self.model.add_essentials(essentials=essentials, aa_dict=aa_dict, rna_nucleotides=rna_nucleotides,
                                      rna_nucleotides_mp=rna_nucleotides_mp)
            self.model.add_mrnas(mrna_dict.values())
            self.model.add_ribosome(rib, free_ratio=0.2)
            # http://bionumbers.hms.harvard.edu/bionumber.aspx?id=102348&ver=1&trm=rna%20polymerase%20half%20life&org=
            # Name          Fraction of active RNA Polymerase
            # Bionumber ID  102348
            # Value 	    0.17-0.3 unitless
            # Source        Bremer, H., Dennis, P. P. (1996) Modulation of chemical composition and other parameters of the cell by growth rate.
            #               Neidhardt, et al. eds. Escherichia coli and Salmonella typhimurium: Cellular
            #                       and Molecular Biology, 2nd ed. chapter 97 Table 1
            self.model.add_rnap(rnap, free_ratio=0.75)

            self.model.build_expression()
            self.model.add_enzymatic_coupling(coupling_dict)

            # if has_neidhardt:
            #     nt_ratios, aa_ratios = get_ratios()
            #     chromosome_len, gc_ratio = get_ecoli_gen_stats()
            #     kdeg_mrna, mrna_length_avg = get_mrna_metrics()
            #     kdeg_enz, peptide_length_avg = get_enz_metrics()
            #     neidhardt_mu, neidhardt_rrel, neidhardt_prel, neidhardt_drel = get_neidhardt_data()
            #
            #     add_interpolation_variables(self.model)
            #     self.model.add_dummies(nt_ratios=nt_ratios, mrna_kdeg=kdeg_mrna, mrna_length=mrna_length_avg,
            #                            aa_ratios=aa_ratios, enzyme_kdeg=kdeg_enz, peptide_length=peptide_length_avg)
            #     add_protein_mass_requirement(self.model, neidhardt_mu, neidhardt_prel)
            #     add_rna_mass_requirement(self.model, neidhardt_mu, neidhardt_rrel)
            #     add_dna_mass_requirement(self.model, mu_values=neidhardt_mu, dna_rel=neidhardt_drel, gc_ratio=gc_ratio,
            #                              chromosome_len=chromosome_len, dna_dict=dna_nucleotides)

            # Need to put after, because dummy has to be taken into account if used.
            self.model.populate_expression()
            self.model.add_trna_mass_balances()
            self.model.growth_reaction.lower_bound = objective_lb
            self.model.repair()
            try:
                self.model.optimize()
            except (AttributeError, SolverError):
                self.model, _, _ = relax_dgo(self.model)
            # self.model.growth_reaction.lower_bound = 0
            # print(f"Build ETFL model for {time.time() - start_time:.2f} seconds!")
            self.model.print_info()

    def _sanitize_varnames(self):
        for met in self.model.metabolites:
            if met.id[0].isdigit():
                met.id = '_' + met.id
        for rxn in self.model.reactions:
            if rxn.id[0].isdigit():
                rxn.id = '_' + rxn.id
        self.model.repair()


if __name__ == '__main__':
    etfl_model = ETFLModel()
    solution = etfl_model.solve()
    solution.to_csv('../../output/etfl_fluxes.csv')
