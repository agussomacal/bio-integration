import os
import sys
from time import time
import matplotlib.pylab as plt
import numpy as np
from hyperopt import hp

runs_dir = os.getcwd()
lib_dir = runs_dir+"/../lib/"

sys.path.insert(0, lib_dir)
import config
import biolib
import diseaseslib
import filemanager

sys.path.insert(0, config.pyNetMelt_dir+"src/lib/")
#os.chdir(config.pyNetMelt_dir+"src/lib/")
import algorithms
import networks
import evaluators
import integrators
from optimizers import Optimizer
import visualizators


########################################################################################################################
#            parameters

global_params = dict()
bambu_params = dict()
preprocess_params = dict()
ppi_params = dict()
coexp_params = dict()
nn_params = dict()
bp_params = dict()
cc_params = dict()
diseases_params = dict()
priorization_params = dict()
optimizer_params = dict()

# ----------------------------------
global_params["analysisname"] = "Experiments_PRINCE_7"

global_params["dataset2use"] = "reads.dgelist"#"prepareForCoexp"#puede ser prepareForCoexp o reads.dgelist, special es usar prepareForCoexp sino es reads.edgelist que tiene todo

# -----------------------------------
preprocess_params["eps"] = 1         #epsiolon apra aplicar el logreads

# ----------------------------------
ppi_params["threshold"] = 0.75#0.75#0.63
ppi_params["binarize"] = True
ppi_params["degree threshold"] = 200#200#1000
ppi_params["color"] = "red"

# ----------------------------------
coexp_params["threshold"] = 0.6
coexp_params["binarize"] = False
coexp_params["similarity"] = "corr"#"gtom1"#"corr"
coexp_params["color"] = "purple"

# ----------------------------------
nn_params["threshold"] = 1.7#activation hreshold#"activation_threshold"#"kcorneighbor"#cluster#"all"
nn_params["r"] = 0.1#probabilidad de que uno este on y otro off
nn_params["color"] = "blue"

# ----------------------------------
bp_params["threshold"] = 0#no threshold
bp_params["binarize"] = False
bp_params["color"] = "green"

# ----------------------------------
cc_params["threshold"] = 0#no threshold
cc_params["binarize"] = False
cc_params["color"] = "orange"

# ------------------------
diseases_params["score_threshold"] = 0.2
diseases_params["num_genes_threshold"] = 5
diseases_params["selected_diseases"] = None

# -------------------------
priorization_params["evaluator"] = {"linkage100_AUC02": evaluators.AUROClinkage}
priorization_params["linkage_interval"] = 100  # how many genes in the linkage
priorization_params["max_fpr"] = 0.1  # maximum false positive rate

priorization_params["leave_p_out"] = 1  # leave k out, 1 2...
priorization_params["n_evaluations"] = np.inf #np.inf#np.inf#3*188  # maximum number of evaluations. Inf means all combinations are tested.
# with the current filters, 188 diseases remain so there is a test for each one.

priorization_params["lambda exponent"] = -0.5  # equivale a 0.5, o sea el laplaciano simetrico. Pero puede ser otro numero... por alguna razon anda mejor =1 o sea heat laplacian. y alphas chicos.
priorization_params["alpha"] = 0.8  # [0.1, 0.5, 0.8]
priorization_params["max_iter"] = 100  # maximum number of iterations to stop propagation.
priorization_params["auroc_normalize"] = True

# -------------------------
optimizer_params["max_evals"] = 200

########################################################################################################################
# --------directories-------------
wd = os.getcwd()

Wpath = "{}/{}".format(config.Wnetworks_dir, global_params["analysisname"])
if not os.path.exists(Wpath):
    os.mkdir(Wpath)

########################################################################################################################
#


# ---------Select genes 2 use in common---------------
gl = diseaseslib.GeneLinkage(gene_code="entrezid", gen_interval_len=priorization_params["linkage_interval"])

# --- load info for networks ---
# ppi
ppi = biolib.PPI(filemanager.Load.ppi())
bp = biolib.Ontology(filemanager.Load.ontology_similarity_data("BP"))
gtex_genes = biolib.EXP().gene_names

# --- finding genes in common ---
genes_in_data = diseaseslib.get_gene_universe(gene_linkage=gl,
                                              dict_of_networks={},
                                              list_of_list_of_genes=[ppi.gene_names, bp.gene_names, gtex_genes],
                                              mode="intersect")
print("Genes in common: {}".format(len(genes_in_data)))
del bp; del gtex_genes

# --- ppi ---
ppi.filter_gene_names(genes_in_data)
ppi.apply_threshold(ppi_params["threshold"])
if ppi_params["binarize"]:
    ppi.binarize()

ppi.to_np(genes_in_data)
genes_in_data = ppi.get_gigant_commponent()
#genes_in_data = list(set(genes_in_data).difference(set(ppi.get_isolated_nodes())))
ppi.filter_gene_names(genes_in_data)
#ppi.isolate_chaperones(ppi_params["degree threshold"])

# --- make dict of networks ---
dict_of_networks = dict()
dict_of_networks["PPI"] = networks.Adjacency(ppi.data, genes_in_data); del ppi

# --- filter diseases ---
bip_diseases = networks.Bipartite(filemanager.Load.DisGeNet(columns=["diseaseId", "geneId", "score"]))
bip_diseases.filter_nodes_by_intersection("geneId", genes_in_data)
bip_diseases.filter_edges_by_score(diseases_params["score_threshold"])
bip_diseases.filter_nodes_by_degree("diseaseId", degree_lower_threshold=diseases_params["num_genes_threshold"])
bip_diseases.filter_nodes_by_intersection("diseaseId", diseases_params["selected_diseases"])

# --- filter gene sequence ---
gl.filter_genes(genes_in_data)



###############################################################################
#                       BP+PPI CC+PPI
###############################################################################

# no PRINCE

prioritize = diseaseslib.Prioritization(bipartite_network=bip_diseases,
                            query_nodes_name="diseaseId",
                            leave_p_out=priorization_params["leave_p_out"],
                            max_evals=priorization_params["n_evaluations"])
l_seeds, l_seeds_weight, l_targets, l_true_targets = prioritize.generate_data_to_prioritize(bip_diseases, gl)


evaluator = evaluators.AUROClinkage(node_names=genes_in_data,
                                    l_seeds=l_seeds,
                                    l_targets=l_targets,
                                    l_true_targets=l_true_targets,
                                    l_seeds_weight=l_seeds_weight,
                                    alpha=priorization_params["alpha"],
                                    laplacian_exponent=priorization_params["lambda exponent"],
                                    tol=1e-08,
                                    max_iter=priorization_params["max_iter"],
                                    max_fpr=priorization_params["max_fpr"],
                                    auroc_normalized=priorization_params["auroc_normalize"])


############################################################
t0=time()
auc_ppi = evaluator.evaluate(dict_of_networks["PPI"])
print("Time: ", time()-t0)
print(auc_ppi)


############################################################
#               Diseases proyections study

# import networkx as nx
#
# def get_hist(data):
#     counts, xlims = np.histogram(data, bins=int(np.sqrt(len(data))))
#     centers = (xlims[:-1] + xlims[1:]) / 2
#     return centers, counts
#
# def make_proyection_insight_plots(proy, filename):
#     ##########################
#     # --- weight distribution ---
#     fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 16), sharex=True)
#     data = proy.values.ravel()
#     ax[0].loglog(*get_hist(data), '.')
#     # ax[0].set_xlabel("edge weight")
#     ax[0].set_ylabel("number of edges")
#     ax[0].set_title("Histogram of edges by weight")
#     ax[1].plot(np.sort(data), 1-np.cumsum(np.sort(data))/np.sum(data))
#     ax[1].set_xlabel("edge weight")
#     ax[1].set_ylabel("Percentage of remaining edges")
#     ax[1].set_title("Remaining edges when setting different thresholds")
#     plt.savefig(filename+"_weight_distribution.svg")
#
#     ##############################
#     # ---- degree distribution ---
#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
#     degree_out = (proy.values>0).sum(axis=0).ravel()  # assuming columns are out.
#     degree_in = (proy.values>0).sum(axis=1).ravel()
#     strength_out = proy.values.sum(axis=0).ravel()  # assuming columns are out.
#     strength_in = proy.values.sum(axis=1).ravel()
#     ax[0].loglog(*get_hist(degree_out), '.', c="black", label="degree out")
#     ax[0].plot(*get_hist(degree_in), '.', c="blue", label="degree in")
#     ax[0].plot(*get_hist(strength_out), '.', c="green", label="strength out")
#     ax[0].plot(*get_hist(strength_in), '.', c="red", label="strength in")
#     ax[0].legend()
#     # ax[0].set_xlabel("edge weight")
#     ax[0].set_ylabel("number of diseases")
#     ax[0].set_title("Distribution of degree & strength")
#     ax[0].set_xlabel("strength/degree")
#
#     thresholds = np.linspace(0, data.max(), 50)
#     num_neighbours_range = [0, 1]
#     for num_nei in num_neighbours_range:
#         for threshold in thresholds:
#             deg_out_counts = np.sum((np.sort(proy.values, axis=0) > threshold).sum(axis=0) <= num_nei)
#             deg_in_counts = np.sum((np.sort(proy.values, axis=0) > threshold).sum(axis=1) <= num_nei)
#             line_out, = ax[1].plot(threshold, deg_out_counts, ".", c=(num_nei/len(num_neighbours_range), 0, 0, 0.7))
#             line_in, = ax[1].plot(threshold, deg_in_counts, ".", c=(0, num_nei/len(num_neighbours_range), 0, 0.7))
#         ax[1].legend((line_out, line_in), ("degree out <= {}".format(num_nei), "degree_in <= {}".format(num_nei)))
#
#     ax[1].set_xlabel("edge weight (threshold)")
#     ax[1].set_ylabel("Number of diseases with few neighbours")
#     ax[1].set_title("Degradation of the network with thresholding")
#     plt.savefig(filename + "_degree_distribution.svg")
#
#     ################################
#     # --- network ---
#     g = nx.from_pandas_adjacency(proy)
#
#     # --- position ---
#     pos = nx.fruchterman_reingold_layout(g)
#     # pos = nx.kamada_kawai_layout(g)
#     # pos = nx.spectral_layout(g,scale=2, dim=2)
#     # --- nodes color and size ---
#     strength = np.array([s for _, s in g.degree(weight="weight")])
#     degree = np.array([d for _, d in g.degree()])
#     col = strength
#     col_max = np.max(strength)
#     col_min = np.min(strength)
#     # --- edge weights ---
#     weight = [edge_attr["weight"] for _, _, edge_attr in g.edges(data=True)]
#
#     plt.figure()
#     nx.draw_networkx_nodes(g, pos, node_size=degree, node_color=col,
#                            alpha=0.5, cmap="jet", vmax=col_max, vmin=col_min)
#     nx.draw_networkx_edges(g, pos, width=weight, edge_color="black", alpha=1)
#     plt.axis("off")
#     plt.savefig(filename + "_network.svg")
#     plt.close("all")
#
#
# modes = ["one_mode_proyection", "laplacian", "laplacian", "laplacian", "laplacian"]
# laplacian_exponents = [None, -1, -1, -0.5, 0]
# simetrizes = [False, True, False, True, False]
# to_undirecteds = [False, True, False, False, True]
#
# for mode, laplacian_exponent, simetrize, to_undirected in zip(modes, laplacian_exponents, simetrizes, to_undirecteds):
#     proy = bip_diseases.get_proyection(bipartite_proyection_column_name="diseaseId",
#                                        mode=mode,
#                                        laplacian_exponent=laplacian_exponent,
#                                        simetrize=simetrize,
#                                        to_undirected=to_undirected)
#     np.fill_diagonal(proy.values, 0)
#     filename = Wpath+"/proy_{}_exp{}_sim{}_und{}".format(mode, laplacian_exponent, simetrize, to_undirected)
#     make_proyection_insight_plots(proy, filename)


#
# space = {"threshold": hp.uniform("threshold", 0, 1),
#          "simetrize": hp.choice("simetrize", [True, False]),
#          "to_directed": hp.choice("to_directed", [True, False]),
#          # "mode": hp.choice("mode", ["laplacian", "one_mode_proyection"]),
#          'laplacian_exponent': hp.uniform('laplacian_exponent', -1.0, 0)
#          }


def eval_func(space, space_fixed):
    try:
        space = {**space, **space_fixed}
        prince = diseaseslib.PrioritizationPRINCE(bipartite_network=bip_diseases,
                                                  query_nodes_name="diseaseId",
                                                  leave_p_out=priorization_params["leave_p_out"],
                                                  max_evals=priorization_params["n_evaluations"],
                                                  similarity_lower_threshold=space["threshold"],
                                                  mode=space["mode"],
                                                  to_undirected=space["to_directed"])
        l_seeds, l_seeds_weight, l_target_genes, l_true_target_genes = prince.generate_data_to_prioritize(bip_diseases, gl)

        localeval = evaluators.AUROClinkage(node_names=genes_in_data,
                                            l_seeds=l_seeds,
                                            l_targets=l_targets,
                                            l_true_targets=l_true_targets,
                                            l_seeds_weight=l_seeds_weight,
                                            alpha=priorization_params["alpha"],
                                            laplacian_exponent=priorization_params["lambda exponent"],
                                            tol=1e-08,
                                            max_iter=priorization_params["max_iter"],
                                            max_fpr=priorization_params["max_fpr"],
                                            auroc_normalized=priorization_params["auroc_normalize"])
        val = localeval.evaluate(dict_of_networks["PPI"])
        print(val)
        return val
    except AssertionError:
        print(space)
        return -1 # mark that something went wrong


############################################################
space = {"threshold": hp.uniform("threshold", 0, 1),
         "to_directed": hp.choice("to_directed", [True, False])}

max_evals = optimizer_params["max_evals"]
for mode in ("laplacian", "one_mode_proyection"):
    test_name = "PRINCE_"+mode
    filename = Wpath+"/{}_PPI_results".format(test_name)
    space_fixed = dict()
    space_fixed["mode"] = mode


    optimizer = Optimizer(optimization_name=test_name,
                          path2files=Wpath,
                          space=space,
                          objective_function=lambda space: eval_func(space, space_fixed),
                          max_evals=max_evals,
                          maximize=True)
    tpe_results, best = optimizer.optimize()
    tpe_results = tpe_results.sort_values("threshold")
    tpe_results.to_csv(filename+".csv")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 7))
    for label, col, isdirected in zip(("directed", "original scores"), ("blue", "green"), (True, False)):
        try:
            visualizators.plot_optimization(x_variable="threshold",
                                            y_variable=evaluator.metric_name,
                                            optimizer=optimizer,
                                            tpe_results=tpe_results.loc[(tpe_results["to_directed"] == isdirected) &
                                                                        (tpe_results[optimizer.__name__] != -1), :],
                                            color=col,
                                            label=label,
                                            ax=ax)
        except:
            pass
    visualizators.plot_baseline(baseline={"value": auc_ppi,
                                          "color": ppi_params["color"],
                                          "label": "no PRINCE"},
                                ax=ax)
    plt.savefig(filename + ".svg")
    plt.close()


####################################################################
####################################################################
####################################################################
sadsvsv
prioritize = diseaseslib.Prioritization(bipartite_network=bip_diseases,
                            query_nodes_name="diseaseId",
                            leave_p_out=priorization_params["leave_p_out"],
                            max_evals=np.inf)
l_seeds, l_seeds_weight, l_targets, l_true_targets = prioritize.generate_data_to_prioritize(bip_diseases, gl)


prince = diseaseslib.PrioritizationPRINCE(bipartite_network=bip_diseases,
                                          query_nodes_name="diseaseId",
                                          leave_p_out=priorization_params["leave_p_out"],
                                          max_evals=np.inf,
                                          similarity_lower_threshold=0.01,
                                          mode="laplacian",  # space["mode"],
                                          laplacian_exponent=-1,
                                          simetrize=True,
                                          to_undirected=True)

for dis, similar_dis in prince.dict_similar_diseases.items():
    print(dis, end="\t")
    print(len(similar_dis), end="\n")
    print(prince.dict_similar_diseases[dis], end="\n")
    print(prince.get_nodes_weights(bip_diseases, dis), end="\n")
    print(prioritize.get_nodes_weights(bip_diseases, dis), end="\n\n")

np.random.seed(0)
seeds, seeds_weight, dropped, groups = prince.generate_evaluation_sets(bip_diseases, gl)
np.random.seed(0)
seeds2, seeds_weight2, dropped2, groups2 = prioritize.generate_evaluation_sets(bip_diseases, gl)
print(set([g[0] for g in dropped]).difference(set([g[0] for g in dropped2])))
print(set([g[0] for g in dropped2]).difference(set([g[0] for g in dropped])))

sum([len(set(s1).difference(set(s2))) != 0 for s1, s2 in zip(dropped, dropped2)])
sum([len(set(s1).difference(set(s2))) != 0 for s1, s2 in zip(seeds_weight, seeds_weight2)])
sum([len(set(s1).difference(set(s2))) != 0 for s1, s2 in zip(seeds, seeds2)])


np.random.seed(0)
l_seeds2, l_seeds_weight2, l_target_genes2, l_true_target_genes2 = prioritize.generate_data_to_prioritize(bip_diseases, gl)
np.random.seed(0)
l_seeds, l_seeds_weight, l_target_genes, l_true_target_genes = prince.generate_data_to_prioritize(bip_diseases, gl)
print(set([g[0] for g in l_true_target_genes2]).difference(set([g[0] for g in l_true_target_genes])))
print(set([g[0] for g in l_true_target_genes]).difference(set([g[0] for g in l_true_target_genes2])))
sum([len(set(s1).difference(set(s2))) != 0 for s1, s2 in zip(l_seeds, l_seeds2)])
sum([len(set(s1).difference(set(s2))) != 0 for s1, s2 in zip(l_seeds_weight, l_seeds_weight2)])
sum([len(set(s1).difference(set(s2))) != 0 for s1, s2 in zip(l_target_genes, l_target_genes2)])
sum([len(set(s1).difference(set(s2))) != 0 for s1, s2 in zip(l_true_target_genes, l_true_target_genes2)])

localeval = evaluators.AUROClinkage(node_names=genes_in_data,
                                    l_seeds=l_seeds,
                                    l_targets=l_targets,
                                    l_true_targets=l_true_targets,
                                    l_seeds_weight=l_seeds_weight,
                                    alpha=priorization_params["alpha"],
                                    laplacian_exponent=priorization_params["lambda exponent"],
                                    tol=1e-08,
                                    max_iter=priorization_params["max_iter"],
                                    max_fpr=priorization_params["max_fpr"],
                                    auroc_normalized=priorization_params["auroc_normalize"])

print("prince: {}".format(localeval.evaluate(dict_of_networks["PPI"])))

localeval = evaluators.AUROClinkage(node_names=genes_in_data,
                                    l_seeds=l_seeds2,
                                    l_targets=l_target_genes2,
                                    l_true_targets=l_true_target_genes2,
                                    l_seeds_weight=l_seeds_weight2,
                                    alpha=priorization_params["alpha"],
                                    laplacian_exponent=priorization_params["lambda exponent"],
                                    tol=1e-08,
                                    max_iter=priorization_params["max_iter"],
                                    max_fpr=priorization_params["max_fpr"],
                                    auroc_normalized=priorization_params["auroc_normalize"])

print("basic: {}".format(localeval.evaluate(dict_of_networks["PPI"])))


##############################################

prioritize = diseaseslib.Prioritization(bipartite_network=bip_diseases,
                            query_nodes_name="diseaseId",
                            leave_p_out=priorization_params["leave_p_out"],
                            max_evals=np.inf)

prince = diseaseslib.PrioritizationPRINCE(bipartite_network=bip_diseases,
                                          query_nodes_name="diseaseId",
                                          leave_p_out=priorization_params["leave_p_out"],
                                          max_evals=np.inf,
                                          similarity_lower_threshold=0.5,
                                          mode="laplacian",  # space["mode"],
                                          laplacian_exponent=-1,
                                          simetrize=True,
                                          to_undirected=True)


np.random.seed(0)
l_seeds2, l_seeds_weight2, l_target_genes2, l_true_target_genes2 = prioritize.generate_data_to_prioritize(bip_diseases, gl)
np.random.seed(0)
l_seeds, l_seeds_weight, l_target_genes, l_true_target_genes = prince.generate_data_to_prioritize(bip_diseases, gl)

localeval = evaluators.AUROClinkage(node_names=genes_in_data,
                                    l_seeds=l_seeds,
                                    l_targets=l_targets,
                                    l_true_targets=l_true_targets,
                                    l_seeds_weight=l_seeds_weight,
                                    alpha=priorization_params["alpha"],
                                    laplacian_exponent=priorization_params["lambda exponent"],
                                    tol=1e-08,
                                    max_iter=priorization_params["max_iter"],
                                    max_fpr=priorization_params["max_fpr"],
                                    auroc_normalized=priorization_params["auroc_normalize"])

print("prince: {}".format(localeval.evaluate(dict_of_networks["PPI"])))

localeval = evaluators.AUROClinkage(node_names=genes_in_data,
                                    l_seeds=l_seeds2,
                                    l_targets=l_target_genes2,
                                    l_true_targets=l_true_target_genes2,
                                    l_seeds_weight=l_seeds_weight2,
                                    alpha=priorization_params["alpha"],
                                    laplacian_exponent=priorization_params["lambda exponent"],
                                    tol=1e-08,
                                    max_iter=priorization_params["max_iter"],
                                    max_fpr=priorization_params["max_fpr"],
                                    auroc_normalized=priorization_params["auroc_normalize"])

print("basic: {}".format(localeval.evaluate(dict_of_networks["PPI"])))

