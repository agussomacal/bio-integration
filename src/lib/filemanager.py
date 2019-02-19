import rpy2.robjects as robjects
import pandas as pd
import numpy as np
from time import time
import os

from config import *

# TODO: make all loadings with a standard output that includes time of loading and the print of having finished.

class Load:
    @staticmethod
    def gene_sequence_bed_data(path=GTEx_dir):
        data = pd.read_csv("{}/genes_hsa_sort.bed".format(path), sep="\t", header=None)
        data.columns = ["cromosoma", "pos_inicial", "pos_final", "ensembl", "score", "where_to_read"]
        return data

    @staticmethod
    def gene_sequence_entrez_data(path=GTEx_dir):
        print("Loading gene sequence data...", end="\t")
        data = pd.read_csv("{}/genes_hsa_sort_entrez.csv".format(path), sep=",")
        print("Loaded.")
        return data

    @staticmethod
    def ontology_similarity_data(ontology="BP", path=GTEx_dir):
        data = robjects.r['readRDS']("{}/MultiplexNetworks/{}adjacency_matrix_entrez.rds".format(path, ontology))
        return pd.DataFrame(np.array(data), columns=np.array(data.colnames, dtype=int),
                            index=np.array(data.rownames, dtype=int))

    @staticmethod
    def ppi(path=GTEx_dir, filename="Human-PPI-HIPPIE.txt"):
        ppi_df = pd.read_csv("{}/{}".format(path, filename), sep="\t",
                             names=["protein1", "entrezid1", "protein2", "entrezid2", "score", "meassure_type"])
        ppi_df = ppi_df[["entrezid1", "entrezid2", "score"]]
        ppi_df["entrezid1"] = ppi_df["entrezid1"].apply(int)
        ppi_df["entrezid2"] = ppi_df["entrezid2"].apply(int)
        return ppi_df

    @staticmethod
    def coexp(tissue, dataset="reads.dgelist", path=byTissue_data_dir):
        data = robjects.r['readRDS']("{}/{}/coexp_{}.RDS".format(path, dataset, tissue))
        return pd.DataFrame(np.array(data), columns=np.array(data.colnames, dtype=int),
                            index=np.array(data.rownames, dtype=int))

    @staticmethod
    def gen_meansd_data(filename):
        data = robjects.r['readRDS']("{}/{}.RDS".format(byTissue_data_dir, filename))
        return pd.DataFrame(np.array(data), columns=["global"], index=np.array(data.names, dtype=int))

    @staticmethod
    def gen_meansd_bytissue_data(filename):
        data = robjects.r['readRDS']("{}/{}.RDS".format(byTissue_data_dir, filename))
        return pd.DataFrame(np.array(data), columns=np.array(data.colnames), index=np.array(data.rownames, dtype=int))

    @staticmethod
    def DisGeNet(path=GTEx_dir, columns=None):
        data = robjects.r("""
                            load_dataGV <- function(filepath){
                                load(paste0(filepath,"/data.table.genAlt.Rdata"))
                                as.data.frame(dataGV[dataGV$associationType %in% genAlt,])
                            }
                        """)(path)
        df_diseases = pd.DataFrame(np.transpose(np.array(data)), columns=np.array(data.colnames))
        df_diseases[["score", "geneId"]] = df_diseases[["score", "geneId"]].apply(pd.to_numeric)
        if columns is not None:
            df_diseases = df_diseases[columns]  # ["diseaseId", "geneId", "score", "diseaseName"]
        return df_diseases

    @staticmethod
    def W(path, filename):
        t0 = time()
        fname = "{}/{}".format(path, filename)
        if fname[-3:] != ".gz":
            fname += ".gz"

        if not os.path.isfile(fname):
            print("W not in folder")
            return None
        else:
            print("loading W...")
            w = pd.read_csv(fname, compression='gzip', index_col=0)
            w.columns = pd.to_numeric(w.columns)  # de otro modo las deja como strings
            w.index = pd.to_numeric(w.index)  # de otro modo las deja como strings
        print("Duracion: {}".format(time() - t0))
        return w


class Save:

    @staticmethod
    def W(w, path, filename):
        if type(w) == pd.core.frame.DataFrame:
            t0 = time()
            fname = "{}/{}".format(path, filename)
            if fname[-3:] != ".gz":
                fname += ".gz"

            if not os.path.isfile(fname):
                print("Saving W network...")
                w.to_csv(fname, compression='gzip')
            else:
                print("Already calculated...")
            print("Duracion: {}".format(time() - t0))


