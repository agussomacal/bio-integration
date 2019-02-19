import pandas as pd
import numpy as np
from time import time
import os

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import Counter


from config import *
import filemanager


all_tissues = ['Adipose - Subcutaneous', 'Adipose - Visceral (Omentum)', 'Adrenal Gland', 'Artery - Aorta',
    'Artery - Coronary', 'Artery - Tibial', 'Brain - Amygdala', 'Brain - Anterior cingulate cortex (BA24)',
    'Brain - Caudate (basal ganglia)', 'Brain - Cerebellar Hemisphere', 'Brain - Cerebellum', 'Brain - Cortex',
    'Brain - Frontal Cortex (BA9)', 'Brain - Hippocampus', 'Brain - Hypothalamus',
               'Brain - Nucleus accumbens (basal ganglia)',
    'Brain - Putamen (basal ganglia)', 'Brain - Spinal cord (cervical c-1)', 'Brain - Substantia nigra',
               'Breast - Mammary Tissue',
    'Cells - EBV-transformed lymphocytes', 'Cells - Transformed fibroblasts', 'Colon - Sigmoid', 'Colon - Transverse',
    'Esophagus - Gastroesophageal Junction', 'Esophagus - Mucosa', 'Esophagus - Muscularis', 'Heart - Atrial Appendage',
    'Heart - Left Ventricle', 'Liver', 'Lung', 'Muscle - Skeletal', 'Nerve - Tibial', 'Pancreas', 'Pituitary', 'Prostate',
    'Skin - Not Sun Exposed (Suprapubic)', 'Skin - Sun Exposed (Lower leg)', 'Spleen', 'Stomach', 'Testis', 'Thyroid',
               'Whole Blood']


########################################################################################################################

###############################################################################
class BioPreProcessing:
    @staticmethod
    def from_edgelist_to_matrix(data, source_column, target_column, score_column):
        node_names = set(data[source_column]).union(set(data[target_column]))
        df_matrix = pd.pivot_table(data, values=score_column, index=[source_column], columns=[target_column],
                                aggfunc=np.sum)
        df_matrix.fillna(0, inplace=True)

        forgotten_column_nodes = list(set(node_names).difference(set(df_matrix.columns)))
        df_matrix[forgotten_column_nodes] = pd.DataFrame(0, columns=forgotten_column_nodes, index=df_matrix.index)
        forgotten_index_nodes = list(set(node_names).difference(set(df_matrix.index)))
        df_matrix = df_matrix.append(pd.DataFrame(0, columns=df_matrix.columns, index=forgotten_index_nodes), ignore_index=False)

        df_matrix = df_matrix.loc[node_names, node_names]
        df_matrix = df_matrix + df_matrix.T
        np.fill_diagonal(df_matrix.values, np.diag(df_matrix.values) / 2)

        return df_matrix
    
    @staticmethod
    def get_isolated_nodes(df_matrix):
        return list(df_matrix.index[(df_matrix.sum(axis=0) == 0)])

    def get_gigant_commponent(self):
        _, membership = connected_components(csgraph=csr_matrix(self.data), directed=False, return_labels=True)
        components_size = Counter(list(membership))
        gigant_component_ix = list(components_size.keys())[np.argmax(list(components_size.values()))]
        return list(np.array(self.gene_names)[np.where(membership == gigant_component_ix)[0]])


class PPI:
    def __init__(self, ppi_df):
        self.data = ppi_df
        self.data["entrezid1"] = self.data["entrezid1"].apply(int)
        self.data["entrezid2"] = self.data["entrezid2"].apply(int)
        self.gene_names = list(set(self.data["entrezid1"]).union(set(self.data["entrezid2"])))

    def filter_gene_names(self, gene_names):
        if self.data.shape[0] > self.data.shape[1]:  # we have the data in edgelist form
            self.gene_names = [gene for gene in gene_names if gene in self.gene_names]
            self.data = self.data[
                self.data["entrezid1"].isin(self.gene_names) & self.data["entrezid2"].isin(self.gene_names)]
        elif self.data.shape[0] == self.data.shape[1]:
            self.data = pd.DataFrame(self.data, columns=self.gene_names, index=self.gene_names)
            self.data = self.data.loc[gene_names, gene_names].fillna(0).values
            self.gene_names = gene_names

        print("Number of genes at end: {}".format(len(self.gene_names)))
        print("Shape of ppi after filtering: {}".format(self.data.shape))

    def apply_threshold(self, umbral):
        self.data = self.data.groupby(["entrezid1", "entrezid2"]).sum()
        self.data = self.data[self.data["score"] >= umbral]
        self.data = self.data.reset_index()  # para sacarle el index herarquico y quede solo 2 columnas mas legibles

        print("ppi shape after thresholding: {}".format(self.data.shape))

    def binarize(self):
        self.data["score"] = 1
        # self.data = self.data[["entrezid1","entrezid2"]]

    def to_np(self, node_names):
        """
        TODO: know why those 2 2 2 2 appear in th matrix
        :param node_names:
        :return:
        """
        self.data = BioPreProcessing.from_edgelist_to_matrix(self.data, source_column="entrezid1",
                                                             target_column="entrezid2", score_column="score")
        self.data = self.data.loc[node_names, node_names].fillna(0).values
        print("Warning: there are {} matrix entries with values > 1.".format((self.data > 1).sum()))
        self.gene_names = node_names

    def get_isolated_nodes(self):
        return [self.gene_names[i] for i, value in enumerate(self.data.sum(axis=0) == 0) if value is True]

    def get_gigant_commponent(self):
        _, membership = connected_components(csgraph=csr_matrix(self.data), directed=False, return_labels=True)
        components_size = Counter(list(membership))
        gigant_component_ix = list(components_size.keys())[np.argmax(list(components_size.values()))]
        return list(np.array(self.gene_names)[np.where(membership == gigant_component_ix)[0]])

    def isolate_chaperones(self, deg_threshold):
        deg = np.squeeze(np.array(np.sum(self.data > 0, axis=0)))  # para sumar degree y no strength se pone el >0
        print("Isolating {} chaperon genes".format(np.sum(deg >= deg_threshold)))
        # pone a 0 las interacciones de los genes chaperones
        self.data[deg >= deg_threshold, :][:, deg >= deg_threshold] = 0
        # self.gene_names = list(np.array(self.gene_names)[deg<deg_threshold])


###############################################################################
# ------------------Coexpression data---------------------
class COEXP:
    def __init__(self, coexp_df):
        self.gene_names = list(set(coexp_df.columns).union(set(coexp_df.index)))
        self.data = coexp_df
        # -------put autocorrelatin to 0-------------
        for i in range(self.data.shape[0]):
            self.data.iloc[i, i] = 0

    def filter_gene_names(self, gene_names):
        self.gene_names = [gene for gene in gene_names if gene in self.gene_names]
        self.data = self.data.loc[self.data.index.isin(self.gene_names), self.data.columns.isin(self.gene_names)]

        print("Number of genes at end: {}".format(len(self.gene_names)))
        print("Shape of COEXP after filtering: {}".format(self.data.shape))

    def apply_threshold(self, umbral):
        self.data[self.data <= umbral] = 0

    def binarize(self):
        self.data[self.data != 0] = 1

    def to_sparse(self):
        self.data = csc_matrix(self.data.as_matrix())

    def convert_nan_to_zero(self):
        self.data = self.data.fillna(0)

    def to_np(self, node_names):
        self.data = self.data.loc[node_names, node_names].fillna(0).values
        self.gene_names = node_names


###############################################################################
# ------------------NN data---------------------
class NN:
    def __init__(self, gen_mean):
        self.gene_names = gen_mean.index
        self.data = gen_mean

    def filter_gene_names(self, gene_names):
        self.gene_names = [gene for gene in gene_names if gene in self.gene_names]
        self.data = self.data[self.data.index.isin(self.gene_names)]

        print("Number of genes at end: {}".format(len(self.gene_names)))
        print("Shape of NN after filtering: {}".format(self.data.shape))

    def apply_threshold(self, umbral, r):
        self.data[self.data <= umbral] = r
        self.data[self.data > umbral] = 1

    def make_network_np(self, node_names):
        self.data = np.outer(self.data.loc[node_names].values, self.data.loc[node_names].values)
        self.gene_names = node_names

###############################################################################
# ------------------Ontology data---------------------
class Ontology:
    def __init__(self, ontology_df):
        self.data = ontology_df
        self.gene_names = self.data.columns

    def filter_gene_names(self, gene_names):
        self.gene_names = [gene for gene in gene_names if gene in self.gene_names]
        self.data = self.data.loc[self.data.index.isin(self.gene_names), self.data.columns.isin(self.gene_names)]

        print("Number of genes at end: {}".format(len(self.gene_names)))
        print("Shape of ontology after filtering: {}".format(self.data.shape))

    def apply_threshold(self, umbral):
        self.data[self.data <= umbral] = 0

    def binarize(self):
        self.data[self.data != 0] = 1

    def convert_nan_to_zero(self):
        self.data = self.data.fillna(0)

    def to_np(self, node_names):
        self.data = self.data.loc[node_names, node_names].fillna(0).values
        self.gene_names = node_names


###############################################################################
# -------------------Expression Data----------------------
class EXP:
    def __init__(self):
        self.gen_mean = filemanager.Load.gen_meansd_data(filename="gene_mean_activation")
        self.gen_sd = filemanager.Load.gen_meansd_data(filename="gene_sd_activation")
        self.gen_mean_by_tissue = filemanager.Load.gen_meansd_bytissue_data(filename="gene_mean_activation_by_tissue")
        self.gen_sd_by_tissue = filemanager.Load.gen_meansd_bytissue_data(filename="gene_sd_activation_by_tissue")
        self.gene_names = list(self.gen_mean.index)
        self.ngenes = len(self.gene_names)


if __name__ == "__main__":
    ppi = PPI(filemanager.Load.ppi())
    genes_in_data = ppi.gene_names
    # genes_in_data = list(np.random.choice(ppi.gene_names, size=1000))

    # --- ppi ---
    ppi.filter_gene_names(genes_in_data)
    ppi.apply_threshold(0.75)
    if True:
        ppi.binarize()

    ppi.to_np(genes_in_data)
    genes_in_data = ppi.get_gigant_commponent()
    #genes_in_data = list(set(genes_in_data).difference(set(ppi.get_isolated_nodes())))
    ppi.filter_gene_names(genes_in_data)
    # ppi.isolate_chaperones(ppi_params["degree threshold"])
    print(ppi.data.shape)
    assert (ppi.data > 1).sum() == 0

    # --- test convertion edgelist to matrix ---
    data = pd.DataFrame([[1, 2, 0.5], [1, 5, 0.1], [2, 3, 0.7], [1, 1, 1]], columns=["A", "B", "C"])
    BioPreProcessing.from_edgelist_to_matrix(data, source_column="A", target_column="B", score_column="C")








