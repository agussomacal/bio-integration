"""
TODO: filter isolated genes??
"""


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
import evaluators
import integrators
from optimizers import Optimizer


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

#----------------------------------
global_params["analysisname"] = "Experiments_lambda_exponent"

# --- Define integrators ---
integrators_dict = {"simple": lambda dict_of_nets: integrators.SimpleAdditive(dict_of_nets),
                    "laplacian_0": lambda dict_of_nets: integrators.LaplacianAdditive(dict_of_nets, 0),
                    "laplacian_01": lambda dict_of_nets: integrators.LaplacianAdditive(dict_of_nets, -0.1),
                    "laplacian_03": lambda dict_of_nets: integrators.LaplacianAdditive(dict_of_nets, -0.3),
                    "laplacian_05": lambda dict_of_nets: integrators.LaplacianAdditive(dict_of_nets, -0.5),
                    "laplacian_07": lambda dict_of_nets: integrators.LaplacianAdditive(dict_of_nets, -0.7),
                    "laplacian_09": lambda dict_of_nets: integrators.LaplacianAdditive(dict_of_nets, -0.9),
                    "laplacian_1": lambda dict_of_nets: integrators.LaplacianAdditive(dict_of_nets, -1)}
global_params["combining_methods"] = list(integrators_dict.keys())


# global_params["tissues"] = biolib.all_tissues#["Lung", "Liver", 'Pancreas', 'Stomach']#, "Heart - Left Ventricle","Adipose - Subcutaneous"]#None
global_params["dataset2use"] = "reads.dgelist"#"prepareForCoexp"#puede ser prepareForCoexp o reads.dgelist, special es usar prepareForCoexp sino es reads.edgelist que tiene todo
global_params["networks_2_use"] = "all"#"intersection"#"union"
global_params["save_w"] = False
# global_params["tissue_specific_mode"] = "by_max_score"#"by_name"
# global_params["n_cores"] = 2
# global_params["recalculate"] = True

#-----------------------------------
preprocess_params["eps"] = 1         #epsiolon apra aplicar el logreads

#----------------------------------
ppi_params["threshold"] = 0.75#0.75#0.63
ppi_params["binarize"] = True
ppi_params["degree threshold"] = 200#200#1000
ppi_params["color"] = "red"

#----------------------------------
coexp_params["threshold"] = 0.6
coexp_params["binarize"] = False
coexp_params["similarity"] = "corr"#"gtom1"#"corr"
coexp_params["color"] = "purple"

#----------------------------------
nn_params["threshold"] = 1.7#activation hreshold#"activation_threshold"#"kcorneighbor"#cluster#"all"
nn_params["r"] = 0.1#probabilidad de que uno este on y otro off
nn_params["color"] = "blue"

#----------------------------------
bp_params["threshold"] = 0#no threshold
bp_params["binarize"] = False
bp_params["color"] = "green"

#----------------------------------
cc_params["threshold"] = 0.6#no threshold
cc_params["binarize"] = False
cc_params["color"] = "orange"

#------------------------
diseases_params["score_threshold"] = 0.2
diseases_params["num_genes_threshold"] = 5
diseases_params["selected_diseases"] = None

#-------------------------
priorization_params["evaluator"] = {"linkage100_AUC02": evaluators.AUROClinkage}
priorization_params["linkage_interval"] = 100  # how many genes in the linkage
priorization_params["max_fpr"] = 0.1  # maximum false positive rate

priorization_params["leave_p_out"] = 1  # leave k out, 1 2...
priorization_params["n_evaluations"] = np.inf#np.inf#3*188  # maximum number of evaluations. Inf means all combinations are tested.
# with the current filters, 188 diseases remain so there is a test for each one.

priorization_params["lambda exponent"] = -0.5  # equivale a 0.5, o sea el laplaciano simetrico. Pero puede ser otro numero... por alguna razon anda mejor =1 o sea heat laplacian. y alphas chicos.
priorization_params["alpha"] = 0.8  # [0.1, 0.5, 0.8]
priorization_params["max_iter"] = 100  # maximum number of iterations to stop propagation.
priorization_params["auroc_normalize"] = True

# -------------------------
optimizer_params["max_evals"] = 1

########################################################################################################################
#--------directories-------------
wd = os.getcwd()

Wpath = "{}/{}".format(config.Wnetworks_dir, global_params["analysisname"])
if not os.path.exists(Wpath):
    os.mkdir(Wpath)

# with open(Wpath+'/params.json') as f:
#     data = json.load(f)

########################################################################################################################
#


# ---------Select genes 2 use in common---------------
# --- using most restricting params

d = diseaseslib.DiGeNet(group_by="by_diseases")
gl = diseaseslib.GeneLinkage(gene_code="entrezid", gen_interval_len=priorization_params["linkage_interval"])

# --- load info for networks ---
# ppi, bp
ppi = biolib.PPI(filemanager.Load.ppi())
bp = biolib.Ontology(filemanager.Load.ontology_similarity_data("BP"))
gtex_genes = biolib.EXP().gene_names

# --- finding genes in common ---
genes_in_data = diseaseslib.get_gene_universe(gene_linkage=gl,
                                              dict_of_networks={},
                                              list_of_list_of_genes=[ppi.gene_names, bp.gene_names, gtex_genes],
                                              mode="intersect")
print("Genes in common: {}".format(len(genes_in_data)))

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

bp.filter_gene_names(genes_in_data)

# --- bp ---

bp.convert_nan_to_zero()
bp.apply_threshold(bp_params["threshold"])
if bp_params["binarize"]:
    bp.binarize()

bp.to_np(genes_in_data)

# --- make dict of networks ---
dict_of_networks = dict()
dict_of_networks["PPI"] = algorithms.Adjacency(ppi.data, genes_in_data); del ppi
dict_of_networks["BP"] = algorithms.Adjacency(bp.data, genes_in_data); del bp

# --- filter diseases ---
d.filter_genes(genes_in_data)
d.filter_columns()
d.filter_genes_byscore(diseases_params["score_threshold"])
d.filter_diseases_bynum(diseases_params["num_genes_threshold"])
d.filter_diseases(diseases_params["selected_diseases"])
d.make_disease_groups()
print(d)

# --- filter gene sequence ---
gl.filter_genes(genes_in_data)

prioritize = diseaseslib.PreparePrioritization(leave_p_out=priorization_params["leave_p_out"],
                                               max_evals=priorization_params["n_evaluations"])
l_seeds, l_seeds_weight, l_targets, l_true_targets = prioritize.generate_data_to_prioritize(d, gl)


###############################################################################
#                       BP+PPI CC+PPI
###############################################################################
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

t0 = time()
auc_bp = evaluator.evaluate(dict_of_networks["BP"])
print("Time: ", time() - t0)
print(auc_bp)



############################################################
for integrator_name, integrator in integrators_dict.items():
    integrator = integrator(dict_of_networks)
    if "Laplacian" in integrator.__name__:
        priorization_exponent = integrator.laplacian_exponent
    else:
        priorization_exponent = priorization_params["lambda exponent"]
    evaluator = evaluators.AUROClinkage(node_names=genes_in_data,
                                        l_seeds=l_seeds,
                                        l_targets=l_targets,
                                        l_true_targets=l_true_targets,
                                        l_seeds_weight=l_seeds_weight,
                                        alpha=priorization_params["alpha"],
                                        laplacian_exponent=priorization_exponent,
                                        tol=1e-08,
                                        max_iter=priorization_params["max_iter"],
                                        max_fpr=priorization_params["max_fpr"],
                                        auroc_normalized=priorization_params["auroc_normalize"])

    test_name = evaluator.metric_name + "_PropagatorExponent" + priorization_exponent + "_" + integrator_name
    optimizer = Optimizer(optimization_name=test_name,
                         path2files=Wpath,
                         space=Optimizer.get_integrator_space(integrator=integrator),
                         objective_function=lambda sp: Optimizer.gamma_objective_function(sp,
                                                                                         evaluator=evaluator,
                                                                                         integrator=integrator),
                         max_evals=optimizer_params["max_evals"],
                         maximize=True)
    tpe_results, best = optimizer.optimize()
    tpe_results = Optimizer.normalize_network_gamma_coeficients(tpe_results, dict_of_networks.keys())
    tpe_results = tpe_results.sort_values("PPI")

    tpe_results.to_csv(Wpath+"/{}_PPI_BP_results.csv".format(test_name))
    plt.figure(figsize=(12, 7))
    plt.plot(tpe_results["PPI"], tpe_results[optimizer.__name__], '.-')
    plt.xlabel("$\gamma_{}PPI$")
    plt.ylabel(evaluator.metric_name)
    plt.hlines((auc_ppi, auc_bp), colors=(ppi_params["color"], bp_params["color"]), xmin=0, xmax=1, linestyle="dashdot")
    plt.savefig(Wpath+"/{}_PPI_BP_results.svg".format(test_name))
    plt.close()


############################################################
test_name = "PPI_lambda_propagator"
filename = Wpath+"/{}_PPI_results".format(test_name)
max_evals = optimizer_params["max_evals"]


def eval_func(space):
    return evaluators.AUROClinkage(node_names=genes_in_data,
                            l_seeds=l_seeds,
                            l_targets=l_targets,
                            l_true_targets=l_true_targets,
                            l_seeds_weight=l_seeds_weight,
                            alpha=priorization_params["alpha"],
                            laplacian_exponent=space["lambda exponent"],  # here is the varaible
                            tol=1e-08,
                            max_iter=priorization_params["max_iter"],
                            max_fpr=priorization_params["max_fpr"],
                            auroc_normalized=priorization_params["auroc_normalize"]).evaluate(dict_of_networks["PPI"])



optimizer = Optimizer(optimization_name=test_name,
                     path2files=Wpath,
                     space={"lambda exponent": hp.uniform("lambda exponent", -1, 0)},
                     objective_function=eval_func,
                     max_evals=max_evals,
                     maximize=True)
tpe_results, best = optimizer.optimize()
tpe_results = tpe_results.sort_values("lambda exponent")

tpe_results.to_csv(filename+".csv")
plt.figure(figsize=(12, 7))
plt.plot(tpe_results["lambda exponent"], tpe_results[optimizer.__name__], '.-', c="orange", label="laplacian_exponent")
plt.hlines(auc_ppi, colors=ppi_params["color"], xmin=-1, xmax=0, linestyle="dashdot", label="baseline: exponent 0.5")
plt.xlabel("$\lambda$ exponent")
plt.ylabel(evaluator.metric_name)
plt.legend()
plt.savefig(filename+".svg")
plt.close()

dvsv
############################################################
test_name = "PPI_threshold_binary"
filename = Wpath+"/{}_PPI_results".format(test_name)
max_evals = optimizer_params["max_evals"]


def eval_func(space):
    # --- ppi ---
    ppi = biolib.PPI(filemanager.Load.ppi())
    ppi.filter_gene_names(genes_in_data)
    ppi.apply_threshold(space["threshold"])
    if space["binarize"]:
        ppi.binarize()
    ppi.to_np(genes_in_data)
    ppi = algorithms.Adjacency(ppi.data, genes_in_data)

    return evaluator.evaluate(ppi)


optimizer = Optimizer(optimization_name=test_name,
                     path2files=Wpath,
                     space={"threshold": hp.uniform("threshold", 0, 1),
                            "binarize": hp.choice("binarize", [True, False])},
                     objective_function=eval_func,
                     max_evals=max_evals,
                     maximize=True)
tpe_results, best = optimizer.optimize()
tpe_results = tpe_results.sort_values("threshold")

tpe_results.to_csv(filename+".csv")
plt.figure(figsize=(12, 7))
plt.plot(tpe_results.loc[tpe_results["binarize"] == 1, "threshold"],
         tpe_results.loc[tpe_results["binarize"] == 1, optimizer.__name__],
         '.-', c="blue", label="binary")
plt.plot(tpe_results.loc[tpe_results["binarize"] == 0, "threshold"],
         tpe_results.loc[tpe_results["binarize"] == 0, optimizer.__name__],
         '*-', c="purple", label="not binary")
plt.xlabel("PPI threshold")
plt.ylabel(evaluator.metric_name)
plt.hlines(auc_ppi, colors=ppi_params["color"], xmin=0, xmax=1, linestyle="dashdot", label="baseline: ppi binary 0.75")
plt.legend()
plt.savefig(filename+".svg")
plt.close()


############################################################
test_name = "BP_threshold_binary"
filename = Wpath+"/{}_BP_results".format(test_name)
max_evals = 2*optimizer_params["max_evals"]


def eval_func(space):
    # --- bp ---
    bp = biolib.Ontology(filemanager.Load.ontology_similarity_data("BP"))
    bp.filter_gene_names(genes_in_data)
    bp.convert_nan_to_zero()
    bp.apply_threshold(space["threshold"])
    if space["binarize"]:
        bp.binarize()
    bp.to_np(genes_in_data)
    bp = algorithms.Adjacency(bp.data, genes_in_data)

    return evaluator.evaluate(bp)


optimizer = Optimizer(optimization_name=test_name,
                     path2files=Wpath,
                     space={"threshold": hp.uniform("threshold", 0, 1),
                            "binarize": hp.choice("binarize", [True, False])},
                     objective_function=eval_func,
                     max_evals=max_evals,
                     maximize=True)
tpe_results, best = optimizer.optimize()
tpe_results = tpe_results.sort_values("threshold")

tpe_results.to_csv(filename+".csv")
plt.figure(figsize=(12, 7))
plt.plot(tpe_results.loc[tpe_results["binarize"] == 1, "threshold"],
         tpe_results.loc[tpe_results["binarize"] == 1, optimizer.__name__],
         '.-', c="blue", label="binary")
plt.plot(tpe_results.loc[tpe_results["binarize"] == 0, "threshold"],
         tpe_results.loc[tpe_results["binarize"] == 0, optimizer.__name__],
         '*-', c="purple", label="not binary")
plt.xlabel("BP threshold")
plt.ylabel(evaluator.metric_name)
plt.hlines(auc_bp, colors=bp_params["color"], xmin=0, xmax=1, linestyle="dashdot", label="baseline: bp not binary 0")
plt.legend()
plt.savefig(filename+".svg")
plt.close()


############################################################
test_name = "PPI_alpha"
filename = Wpath+"/{}_PPI_results".format(test_name)
max_evals = optimizer_params["max_evals"]


def eval_func(space):
    return evaluators.AUROClinkage(node_names=genes_in_data,
                            l_seeds=l_seeds,
                            l_targets=l_targets,
                            l_true_targets=l_true_targets,
                            l_seeds_weight=l_seeds_weight,
                            alpha=space["alpha"],
                            laplacian_exponent=priorization_params["lambda exponent"],  # here is the varaible
                            tol=1e-08,
                            max_iter=priorization_params["max_iter"],
                            max_fpr=priorization_params["max_fpr"],
                            auroc_normalized=priorization_params["auroc_normalize"]).evaluate(dict_of_networks["PPI"])



optimizer = Optimizer(optimization_name=test_name,
                     path2files=Wpath,
                     space={"alpha": hp.uniform("alpha", 0, 1)},
                     objective_function=eval_func,
                     max_evals=max_evals,
                     maximize=True)
tpe_results, best = optimizer.optimize()
tpe_results = tpe_results.sort_values("alpha")

tpe_results.to_csv(filename+".csv")
plt.figure(figsize=(12, 7))
plt.plot(tpe_results["alpha"], tpe_results[optimizer.__name__], '.-', c="orange", label="alpha")
plt.hlines(auc_ppi, colors=ppi_params["color"], xmin=0, xmax=1, linestyle="dashdot", label="baseline: alpha 0.8")
plt.xlabel("alpha propagation intensity")
plt.ylabel(evaluator.metric_name)
plt.legend()
plt.savefig(filename+".svg")
plt.close()




# auc = list()
#
# assert ((a == dict_of_networks["PPI"].matrix).all())
#
# assert (a == dict_of_networks["PPI"].matrix).all()
# auc.append(evaluator.evaluate(1*dict_of_networks["BP"]+0*dict_of_networks["PPI"]))
# assert (a == dict_of_networks["PPI"].matrix).all()
# auc.append(evaluator.evaluate(0.9*dict_of_networks["BP"]+0.1*dict_of_networks["PPI"]))
# assert (a == dict_of_networks["PPI"].matrix).all()
# auc.append(evaluator.evaluate(0.7*dict_of_networks["BP"]+0.3*dict_of_networks["PPI"]))
# auc.append(evaluator.evaluate(0.5*dict_of_networks["BP"]+0.5*dict_of_networks["PPI"]))
# auc.append(evaluator.evaluate(0.3*dict_of_networks["BP"]+0.7*dict_of_networks["PPI"]))
# auc.append(evaluator.evaluate(0.1*dict_of_networks["BP"]+0.9*dict_of_networks["PPI"]))
# assert (a == dict_of_networks["PPI"].matrix).all()
# auc.append(evaluator.evaluate(0*dict_of_networks["BP"]+1*dict_of_networks["PPI"]))
#
# plt.figure(figsize=(12, 7))
# plt.plot([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1], auc, "*-")
# plt.hlines((auc_ppi, auc_bp), colors=(ppi_params["color"], bp_params["color"]), xmin=0, xmax=1, linestyle="dashdot")
# plt.savefig(Wpath+"/{}_PPI_BP_results.svg".format("simpletry"))
#
# trials
# tpe_results2 = {param_name: param_value for param_name, param_value in trials.idxs_vals[1].items()}
# tpe_results2["a"] = [sign * x['loss'] for x in trials.results]
# tpe_results2 = pd.DataFrame(tpe_results)
