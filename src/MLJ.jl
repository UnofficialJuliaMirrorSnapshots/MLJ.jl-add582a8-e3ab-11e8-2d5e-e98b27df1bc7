module MLJ

## EXPORTS

export MLJ_VERSION

# defined in include files:
export @curve, @pcurve, pretty,                   # utilities.jl
    coerce, supervised, unsupervised,             # tasks.jl
    report,                                       # machines.jl
    Holdout, CV, evaluate!, Resampler,            # resampling.jl
    Params, params, set_params!,                  # parameters.jl
    strange, iterator,                            # parameters.jl
    Grid, TunedModel, learning_curve!,            # tuning.jl
    EnsembleModel,                                # ensembles.jl
    rebind!,                                      # networks.jl
    machines, sources, anonymize!,                # composites.jl
    @from_network,                                # composites.jl
    fitresults,                                   # composites.jl
    @pipeline,                                    # pipelines.jl
    matching                                      # matching.jl

# defined in include files "machines.jl and "networks.jl":
export Machine, NodalMachine, machine, AbstractNode,
        source, node, fit!, freeze!, thaw!, Node, sources, origins

# re-export from Random, Statistics, Distributions, CategoricalArrays:
export pdf, mode, median, mean, shuffle!, categorical, shuffle, levels, levels!
export std, support

# re-export from MLJBase and ScientificTypes:
export nrows, nfeatures, color_off, color_on,
    selectrows, selectcols,
    SupervisedTask, UnsupervisedTask, MLJTask,
    Deterministic, Probabilistic, Unsupervised, Supervised,
    DeterministicNetwork, ProbabilisticNetwork,
    GrayImage, ColorImage, Image,
    Found, Continuous, Finite, Infinite,
    OrderedFactor, Unknown,
    Count, Multiclass, Binary, Scientific,
    scitype, scitype_union, schema, scitypes,
    target_scitype, input_scitype, output_scitype,
    predict, predict_mean, predict_median, predict_mode,
    transform, inverse_transform, se, evaluate, fitted_params,
    @constant, @more, HANDLE_GIVEN_ID, UnivariateFinite,
    classes,
    partition, unpack,
    mav, mae, rms, rmsl, rmslp1, rmsp, l1, l2,
    misclassification_rate, cross_entropy,
    default_measure,
    @load_boston, @load_ames, @load_iris, @load_reduced_ames,
    @load_crabs

# re-export from MLJModels:
export models, localmodels, @load, load, info,
    ConstantRegressor, ConstantClassifier,     # builtins/Constant.jl
    StaticTransformer, FeatureSelector,        # builtins/Transformers.jl
    UnivariateStandardizer, Standardizer,
    UnivariateBoxCoxTransformer,
    OneHotEncoder, UnivariateDiscretizer,
    FillImputer


## IMPORTS

using MLJBase
using MLJModels

# these are defined in MLJBase
export load_boston, load_ames, load_iris
export load_reduced_ames
export load_crabs

# to be extended:
import MLJBase: fit, update, clean!,
    predict, predict_mean, predict_median, predict_mode,
    transform, inverse_transform, se, evaluate, fitted_params,
    show_as_constructed, params
import MLJModels: models

import Pkg, Pkg.TOML
using Tables, OrderedCollections

using  CategoricalArrays
import Distributions
import Distributions: pdf, mode
import Statistics, StatsBase, LinearAlgebra, Random
import Random: AbstractRNG, MersenneTwister

using ProgressMeter

import PrettyTables
using ScientificTypes

using ComputationalResources
using ComputationalResources: CPUProcesses

const DEFAULT_RESOURCE = Ref{AbstractResource}(CPU1())

# convenience packages
using DocStringExtensions: SIGNATURES, TYPEDEF

# to be extended:
import Base: ==, getindex, setindex!
import StatsBase.fit!

# from Standard Library:
import Distributed: @distributed, nworkers, pmap
using RecipesBase # for plotting

const srcdir = dirname(@__FILE__) # the directory containing this file:
const CategoricalElement = Union{CategoricalString,CategoricalValue}

# FIXME replace with either Pkg.installed()["MLJ"] or
# uuid = Pkg.project().dependencies["MLJ"]
# version = Pkg.dependencies()[uuid].version
# ---
# this is currently messy because it's been enacted then reverted
# see https://github.com/JuliaLang/julia/pull/33410
# and https://github.com/JuliaLang/Pkg.jl/pull/1086/commits/996c6b9b69ef0c058e0105427983622b7cc8cb1d
toml = Pkg.TOML.parsefile(joinpath(dirname(dirname(pathof(MLJ))), "Project.toml"))
const MLJ_VERSION = toml["version"]

## INCLUDES

include("utilities.jl")     # general purpose utilities
include("machines.jl")
include("networks.jl")      # for building learning networks
include("composites.jl")    # composite models & exporting learning networks
include("pipelines.jl")     # pipelines (exported linear learning networks)
include("operations.jl")    # syntactic sugar for operations (predict, etc)

if VERSION ≥ v"1.3.0-"
    include("arrows.jl")
end

include("resampling.jl")    # resampling strategies and model evaluation
include("parameters.jl")    # hyperparameter ranges and grid generation
include("tuning.jl")
include("ensembles.jl")     # homogeneous ensembles
include("model_matching.jl")# inferring model search criterion from data
include("tasks.jl")         # enhancements to MLJBase task interface
include("scitypes.jl")      # extensions to ScientificTypes.sictype
include("plotrecipes.jl")

end # module
