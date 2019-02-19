import sys
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import LeavePOut
import pdb

import filemanager
import config

sys.path.insert(0, config.pyNetMelt_dir+"src/lib/")
import networks
# import algorithms
# import evaluators
# import integrators
# from optimizers import Optimizer
# import visualizators


class DiGeNet:
    def __init__(self, group_by="by_diseases"):
        self.group_by = group_by # if we will group diseases by similarity (to make balanced sets and/or do PRINCE)
        self.disease_groups = dict()  # which diseases are in each group
        self.disease_to_group = dict()  # which group each disease belong

        self.df_diseases = Load.diseases()
        self.df_diseases[["score", "geneId"]] = self.df_diseases[["score", "geneId"]].apply(pd.to_numeric)

    def filter_columns(self, columns=["diseaseId", "geneId", "score", "diseaseName"]):
        # ------filter columns that will be used---------------
        self.df_diseases = self.df_diseases[columns]

    def filter_genes(self, geneids=None):
        # -------filter genes not in external list----------
        if geneids is not None:
            self.df_diseases = self.df_diseases.loc[self.df_diseases["geneId"].isin(geneids)]

    def filter_genes_byscore(self, score_lower_threshold):
        # ---------filter genes by threshold--------------
        self.df_diseases = self.df_diseases.groupby(["diseaseId", "geneId", "diseaseName"]).sum().reset_index()
        self.df_diseases = self.df_diseases.loc[self.df_diseases["score"] >= score_lower_threshold, :]

    def filter_diseases(self, diseaseids=None):
        # --------- filter diseases by external diseases ---------
        if diseaseids is not None:
            self.df_diseases = self.df_diseases.loc[self.df_diseases["diseaseId"].isin(diseaseids)]

    def filter_diseases_bynum(self, num_genes_lower_threshold):
        # ------- filter diseases with low number of genes ------------
        cant_genes_per_dis = self.df_diseases[["diseaseId", "geneId"]].groupby("diseaseId").count().reset_index()
        disseses_set = cant_genes_per_dis.loc[cant_genes_per_dis["geneId"] >= num_genes_lower_threshold, "diseaseId"]
        self.df_diseases = self.df_diseases.loc[self.df_diseases["diseaseId"].isin(disseses_set), :]

    def __str__(self):
        return "Remaining diseases: {}".format(len(self.df_diseases["diseaseId"].unique())) + "\n" \
                "Remaining genes: {}".format(len(self.df_diseases["geneId"].unique()))

    def get_disease_gen_ids(self, disease):
        return self.df_diseases.loc[self.df_diseases["diseaseId"].isin([disease]), "geneId"].unique()

    def get_gen_ids(self):
        return list(set([gen for dis in self.get_diseases_ids() for gen in self.get_disease_gen_ids(dis)]))

    def get_diseases_ids(self):
        return self.df_diseases["diseaseId"].unique()

    def get_cant_diseases(self):
        return len(self.df_diseases["diseaseId"].unique())

    def get_disease_name(self, disease_id):
        return self.df_diseases.loc[self.df_diseases["diseaseId"] == disease_id, "diseaseName"].unique()[0]

    def make_groups_by_diseases(self):
        """
        Each gene is its own group. This is the naive partition of diseases.
        :return: dictionary: groups as keys, list of diseses as values.
        """
        for i, disease in enumerate(self.get_diseases_ids()):
            self.disease_groups[i] = [disease]
            self.disease_to_group[disease] = i

    def make_groups_by_similarity(self):
        """
        TODO: this method
        Using similarity matrix between diseases, divide them in groups of similar diseses: those that share many genes.
        :return: dictionary: groups as keys, list of diseses as values.
        """
        raise Exception("Method still not implemented")
        pass

    def make_disease_groups(self):
        if len(self.disease_groups) > 0:
            print("diseases groups already made in mode: {}".format(self.group_by))
        else:
            if self.group_by == "by_diseases":
                self.make_groups_by_diseases()
            elif self.group_by == "by_similarity":
                self.make_groups_by_similarity()

    def get_weight_by_similarity(self):
        pass

    def get_disease_gens_weight(self, disease):
        if self.group_by == "by_diseases":
            # all the genes in the disease have weight=1; there are no extra genes.
            diseasegens_weight = {gen: 1 for gen in self.get_disease_gen_ids(disease)}
            extragens_weight = {}
        elif self.group_by == "by_similarity":
            group = self.disease_to_group[disease]
            pass

        return diseasegens_weight, extragens_weight


class GeneLinkage:
    def __init__(self, gene_code="entrezid", gen_interval_len=100):
        self.gene_code = gene_code  # it can be "entrezid" or "ensembl"
        self.gen_interval_len = gen_interval_len
        self.linkage = {}

        self.gene_sequence = filemanager.Load.gene_sequence_entrez_data()
        self.gene_sequence = self.gene_sequence[[self.gene_code, "chromosome"]]

        print("Removing copies of duplicated genes.")
        print("Duplicated genes: ", np.sum(self.gene_sequence["entrezid"].duplicated()))
        for ix, gen in self.gene_sequence.loc[self.gene_sequence["entrezid"].duplicated(), "entrezid"].iteritems():
            print("gen: ", gen,\
                  "\t -> \t index in sequence of duplicates: ", list(np.where(self.gene_sequence["entrezid"] == gen)[0]))
        self.gene_sequence = self.gene_sequence.loc[~self.gene_sequence["entrezid"].duplicated(), :]
        self.gene_sequence = self.gene_sequence.reset_index(drop=True)

        # self.gene_sequence.insert(0, 'availability', True)
        # print(self.gene_sequence.head())

    def get_genes_in_sequence(self):
        return list(set(self.gene_sequence[self.gene_code].tolist()))

    def chromosome_has_enough_genes(self, cromosome):
        return self.gen_interval_len < (self.gene_sequence["chromosome"] == cromosome).sum()

    def is_gene_available(self, gene):
        return gene in self.gene_sequence[self.gene_code].values
        # return self.gene_sequence["availability"][self.gene_sequence[self.gene_code] == gene].all()

    def filter_genes(self, gene_ids=None):
        if gene_ids is not None:  # filter genes present in the datasets.
            self.gene_sequence = self.gene_sequence.loc[self.gene_sequence[self.gene_code].isin(gene_ids), :]
            self.gene_sequence = self.gene_sequence.reset_index(drop=True)
            # returns the genes that are both in the gene sequence and the gene ids.

            # modify availability of genes to be used in evaluation.
            # when in a chromosome there are not enough genes to complete not even a linkage then all chromosome is
            # wiped but only for evaluation of dropped genes, for networks the genes still are useful.

            print("Erasing chromosomes with less number of genes that the linkage size.")
            for chromosome in self.gene_sequence["chromosome"].unique():
                if not self.chromosome_has_enough_genes(chromosome):
                    self.gene_sequence = self.gene_sequence.loc[self.gene_sequence["chromosome"] != chromosome, :]
                    self.gene_sequence = self.gene_sequence.reset_index(drop=True)
                    # self.gene_sequence.loc[self.gene_sequence["chromosome"] == chromosome, "availability"] = False

    def create_linkage_interval(self, genes):
        """
        TODO: filter seeds in the linkage or not??

        Creates a dictionary gene -> linkage of genes that contains the information of what genes are in the DNA
        sequence sourroundings. It will be useful to create the target/seed matrices.
        :param entrez_genes: the list of genes that will be used as targets.
        :return:
        """
        genes = set(genes).difference(set(self.linkage.keys()))  # only do the genes not yet in linkage
        print("Generating linkage interval for {} genes...".format(len(genes)), end="\n")

        max_ix = self.gene_sequence.shape[0]
        for count, gene in enumerate(genes):
            print("\r done {}%".format((100*count)//len(genes)), end="\t")
            # takes only the first one if there are many.
            ix = self.gene_sequence.index[self.gene_sequence[self.gene_code] == gene].tolist()[0]
            chromosome = self.gene_sequence.loc[ix, "chromosome"]

            linkage_temp = []
            i = 1
            while len(linkage_temp) < self.gen_interval_len:
                # add genes above in the sequence
                if ix + i < max_ix and self.gene_sequence.loc[ix + i, "chromosome"] == chromosome:
                    gene_to_add = self.gene_sequence.loc[ix + i, self.gene_code]
                    linkage_temp += [gene_to_add]
                # add genes below in the sequence
                if ix - i >= 0 and self.gene_sequence.loc[ix - i, "chromosome"] == chromosome:
                    gene_to_add = self.gene_sequence.loc[ix - i, self.gene_code]
                    linkage_temp += [gene_to_add]
                i += 1

            if len(linkage_temp) > self.gen_interval_len:
                linkage_temp = linkage_temp[:self.gen_interval_len]
            self.linkage[gene] = linkage_temp + [gene]  # add the linkage central gene
            assert gene in self.linkage[gene]
            assert len(self.linkage[gene]) == self.gen_interval_len+1

        print("Some genes may be duplicated")
        print("Finished.")


########################################################################################################################
#   Prioritization class to create the train test sets accordingly.
class Prioritization:
    def __init__(self, bipartite_network, query_nodes_name, leave_p_out=1, max_evals=np.inf):
        self.leave_p_out = leave_p_out
        self.max_evals = max_evals # makes evals per disease group up to the maximum with no repetition.
        self.query_nodes_name = query_nodes_name  # would be the diseases. But if you whant to do inverse...

        self.groups = dict()
        self.make_groups(bipartite_network)  # make groups of diseases if they haven't been made already.
        self.score_same_dis = 1

    def make_groups(self, bipartite_network):
        """
        Each disease-node is its own group. This is the naive partition.
        :return: dictionary: groups as keys, list of diseses as values.
        """
        for i, node in enumerate(bipartite_network.get_nodes_ids(self.query_nodes_name)):
            self.groups[i] = [node]

    def get_nodes_weights(self, bipartite_network, source_node):
        target_nodes = bipartite_network.get_neighborhood(source_node, self.query_nodes_name)
        return {node: self.score_same_dis for node in target_nodes}, {}  # all with the same weight; no extra gens

    def generate_evaluation_sets(self, bipartite_network, gene_linkage, rand_seed=0):
        np.random.seed(rand_seed)
        lpo = LeavePOut(p=self.leave_p_out)

        number_of_groups = len(self.groups)
        evals_per_group = self.max_evals/number_of_groups

        seeds = []
        seeds_weight = []
        dropped = []
        groups = []
        # self.groups is a dictionary with a list of diseases per group
        for group_name, diseases_in_group in self.groups.items():
            # to balance evaluation over groups
            number_of_diseases_in_group = len(diseases_in_group)
            evals_per_disease = evals_per_group/number_of_diseases_in_group
            for dis in diseases_in_group:
                # get weights of the genes in the disease and extra genes if the mode allows it (PRINCE).
                disgens_weight, extragens_weight = self.get_nodes_weights(bipartite_network, dis)
                disease_genes = list(set(list(disgens_weight.keys())))

                # to avoid repeted evaluations
                evals_in_this_disease = int(np.min([lpo.get_n_splits(disease_genes), evals_per_disease]))
                # select subset of evaluations
                i = 0
                for seeds_ix, dropped_ix in lpo.split(disease_genes):
                    if i > evals_in_this_disease:  # to avoid adding too many evals
                        break

                    # could be more than 1 gene if leave many out
                    dropped_genes = [disease_genes[drop_ix] for drop_ix in dropped_ix]
                    # if all the genes to drop are linkageables
                    if all([gene_linkage.is_gene_available(gene) for gene in dropped_genes]):
                        # add the disease genes to the seeds and also the extra genes given by PRINCE method except for
                        # those genes already in the dropped set.

                        seeds_genes = [disease_genes[seed] for seed in seeds_ix]
                        seeds.append(seeds_genes + [gen for gen in extragens_weight.keys() if gen not in dropped_genes])
                        seeds_weight.append([disgens_weight[gen] for gen in seeds_genes] +
                                            [weight for gen, weight in extragens_weight.items() if
                                             gen not in dropped_genes])

                        dropped.append(dropped_genes)
                        groups.append(group_name)
                        i += 1

        # force the cases to be exactly the number of evals.
        chosen_ixes = np.random.choice(len(seeds_weight), int(np.min([self.max_evals, len(seeds_weight)])), replace=False)
        print("Number of test cases to perform: ", len(chosen_ixes))
        seeds_weight = [seeds_weight[chosen_ix] for chosen_ix in chosen_ixes]
        seeds = [seeds[chosen_ix] for chosen_ix in chosen_ixes]
        dropped = [dropped[chosen_ix] for chosen_ix in chosen_ixes]
        groups = [groups[chosen_ix] for chosen_ix in chosen_ixes]

        return seeds, seeds_weight, dropped, groups

    def generate_data_to_prioritize(self, bipartite_network, gene_linkage):
        print("Generate data to prioritize: seeds, seed weights, targets and true targets.")
        l_seeds, l_seeds_weight, l_true_target_genes, l_groups = self.generate_evaluation_sets(bipartite_network, gene_linkage)
        gene_linkage.create_linkage_interval(list(set(list(itertools.chain(*l_true_target_genes)))))

        l_target_genes = []
        for genes in l_true_target_genes:
            linkaged_genes = list(set([g for gene in genes for g in gene_linkage.linkage[gene]]))
            l_target_genes.append(linkaged_genes)  # genes to be targeted together with the true ones.

        return l_seeds, l_seeds_weight, l_target_genes, l_true_target_genes


class PrioritizationPRINCE(Prioritization):
    def get_nodes_weights(self, bipartite_network, source_node):
        # The weight assigned to the opposite nodes connected to the source node is 1.
        # For the rest is the maximum value of weight that in the similar nodes that node is neighbour

        # --- for the similar diseases to the queried one ---
        extra_gens_weights = pd.DataFrame([], columns=["gen", "weight"])
        # the scores of other diseases can't be greater than the disease itself.
        assert (np.array(list(self.dict_similar_diseases[source_node].values())) <= self.score_same_dis).all()
        for dis, weight in self.dict_similar_diseases[source_node].items():
            if dis == source_node:
                # source_node_weight = weight # to normalize the other weights.
                continue

            genes_in_disease = bipartite_network.get_neighborhood(source_node=dis,
                                                                  bipartite_source_column_name=self.query_nodes_name)
            extra_gens_weights = extra_gens_weights.append(pd.DataFrame.from_dict({"gen": genes_in_disease,
                                                                                   "weight": weight}),
                                                           ignore_index=True)

        # --- for the disease queried ---
        genes_in_disease = bipartite_network.get_neighborhood(source_node, self.query_nodes_name)
        gens_weights = {gen: self.score_same_dis for gen in genes_in_disease}

        # --- filter repeted genes; get the max value of score; filter genes in the original disease ---
        extra_gens_weights = extra_gens_weights.groupby(["gen"]).max()
        extra_gens_weights = extra_gens_weights[~extra_gens_weights.index.isin(list(gens_weights.keys()))]
        # extra_gens_weights = extra_gens_weights/source_node_weight*self.score_same_dis # to normalize scores

        return gens_weights, extra_gens_weights.to_dict()["weight"]

    def __init__(self, bipartite_network, query_nodes_name, leave_p_out=1, max_evals=np.inf,
                 similarity_lower_threshold=0, mode="one_mode_proyection", to_undirected=True):
        # should be -1 and False to guarantee that the self similarity is the maximum possible and the normalization
        # valid: when dividing by the self-weight.
        self.default_laplacian_exponent = -1
        self.simetrize = False

        Prioritization.__init__(self, bipartite_network=bipartite_network, query_nodes_name=query_nodes_name,
                                leave_p_out=leave_p_out, max_evals=max_evals)

        proy = bipartite_network.get_proyection(query_nodes_name, mode=mode,
                                                laplacian_exponent=self.default_laplacian_exponent,
                                                simetrize=self.simetrize, to_undirected=to_undirected)
        # normalize columns by self-weight
        proy = proy.divide(np.diag(proy), axis=1) # divide columns by the value of the self-weigth to normalize.
        self.dict_similar_diseases = bipartite_network.get_similar_nodes(proy, similarity_lower_threshold)


def get_gene_universe(gene_linkage, dict_of_networks={}, list_of_list_of_genes=[], mode="intersect"):
    genes_in_linkage = gene_linkage.get_genes_in_sequence()
    genes_in_networks = [network.node_names for network in dict_of_networks.values()]

    print("Generating gene common universe by: {}".format(mode))
    if mode == "intersect":
        genes_to_use = set(genes_in_linkage)
        if len(genes_in_networks) > 0:  # intersect only if there is something to intersect
            genes_to_use = genes_to_use.intersection(*genes_in_networks)
        if len(list_of_list_of_genes) > 0:  # intersect only if there is something to intersect
            genes_to_use = genes_to_use.intersection(*list_of_list_of_genes)
        # genes_to_use = genes_to_use.intersection(genes_in_disease)
    else:
        pass
        # genes_to_use = list(set(genes_in_linkage).union(*genes_to_use))

    return list(genes_to_use)


if __name__ == "__main__":
    # --------------------------------------------
    # parameters
    max_evals = 10  # np.inf
    leave_p_out = 1
    gen_interval_len = 2

    # ---------Select genes 2 use in common---------------
    # --- using most restricting params

    # bip_diseases = networks.Bipartite(filemanager.Load.DisGeNet(columns=["diseaseId", "geneId", "score"]))
    # gl = GeneLinkage(gene_code="entrezid", gen_interval_len=gen_interval_len)
    #
    # genes_in_data = get_gene_universe(dict_of_networks={}, gene_linkage=gl)
    # genes_in_data = list(set(genes_in_data).intersection(bip_diseases.get_nodes_ids("geneId")))
    # print("Genes in common: {}".format(len(genes_in_data)))
    #
    #
    # # --- filter diseases ---
    # bip_diseases.filter_nodes_by_intersection("geneId", genes_in_data)
    # bip_diseases.filter_edges_by_score(0.2)
    # bip_diseases.filter_nodes_by_degree("diseaseId", degree_lower_threshold=8)
    # bip_diseases.filter_nodes_by_intersection("diseaseId", None)
    #
    # # --- filter gene sequence ---
    # genes_in_data = list(set(genes_in_data).intersection(bip_diseases.get_nodes_ids("geneId")))
    # print(len(genes_in_data))
    #
    # bip_diseases.filter_nodes_by_intersection("geneId", genes_in_data)
    # gl.filter_genes(genes_in_data)
    #
    # genes2linkage = [gene for gene in genes_in_data if gl.is_gene_available(gene)]
    # print(len(genes2linkage))

    # ---------- Test prioritization ------------------

    gl = GeneLinkage(gene_code="entrezid", gen_interval_len=gen_interval_len)
    gl.gene_sequence = pd.DataFrame([["g0", "Y"],
                                     ["g1", "Y"],
                                     ["g2", "Y"],
                                     ["g3", "Y"],
                                     ["g4", "Y"]],
                                    columns=['entrezid', 'chromosome'])

    edgelist = pd.DataFrame([["d1", "g1", 0.2],
                             ["d1", "g2", 0.2],
                             ["d2", "g2", 0.2],
                             ["d2", "g3", 0.2]],
                            columns=["diseaseId", "geneId", "score"])
    bip_diseases = networks.Bipartite(edgelist)

    genes_in_data = get_gene_universe(dict_of_networks={}, gene_linkage=gl)
    genes_in_data = list(set(genes_in_data).intersection(bip_diseases.get_nodes_ids("geneId")))
    print("Genes in common: {}".format(len(genes_in_data)))

    # gl.filter_genes(genes_in_data)
    gl.create_linkage_interval(genes_in_data)

    prioritize = Prioritization(bipartite_network=bip_diseases,
                                query_nodes_name="diseaseId",  # which column of the bipartite is used for quering prioritization: diseases in this case and in general.
                                leave_p_out=leave_p_out,
                                max_evals=max_evals)
    l_seeds, l_seeds_weight, l_target_genes, l_true_target_genes = prioritize.generate_data_to_prioritize(bip_diseases, gl)

    print(l_seeds)
    print(l_seeds_weight)
    print(l_target_genes)
    print(l_true_target_genes)


    # ------ PRINCE ------
    print("\n\n PRINCE \n")
    prioritize = PrioritizationPRINCE(bipartite_network=bip_diseases,
                                      query_nodes_name="diseaseId",
                                      leave_p_out=leave_p_out,
                                      max_evals=max_evals,
                                      similarity_lower_threshold=0,
                                      mode="one_mode_proyection",
                                      to_undirected=False)
    prioritize.get_nodes_weights(bip_diseases, bip_diseases.get_nodes_ids("diseaseId")[0])
    l_seeds, l_seeds_weight, l_target_genes, l_true_target_genes = prioritize.generate_data_to_prioritize(bip_diseases, gl)

    print(l_seeds)
    print(l_seeds_weight)
    print(l_target_genes)
    print(l_true_target_genes)

