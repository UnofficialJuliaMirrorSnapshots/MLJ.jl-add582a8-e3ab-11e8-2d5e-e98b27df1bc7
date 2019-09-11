module MLJ

## EXPORTS

# defined in include files:
export @curve, @pcurve, pretty,                       # utilities.jl
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
        @pipeline                                      # pipelines.jl

# defined in include files "machines.jl and "networks.jl":
export Machine, NodalMachine, machine, AbstractNode,
        source, node, fit!, freeze!, thaw!, Node, sources, origins

# defined in include file "builtins/Transformers.jl":
#-#

# rexport from Random, Statistics, Distributions, CategoricalArrays:
export pdf, mode, median, mean, shuffle!, categorical, shuffle, levels, levels!
export std

# re-export from MLJBase and ScientificTypes:
export nrows, nfeatures, 
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
    partition, unpack,
    mav, mae, rms, rmsl, rmslp1, rmsp, l1, l2,  
    misclassification_rate, cross_entropy,      
    default_measure,                            
    @load_boston, @load_ames, @load_iris, @load_reduced_ames,
    @load_crabs

# re-export from MLJModels:
export models, localmodels, @load, load, info,
    ConstantRegressor, ConstantClassifier,     # builtins/Constant.jl
    KNNRegressor,                              # builtins/KNN.jl  
    StaticTransformer, FeatureSelector,        # builtins/Transformers.jl
    UnivariateStandardizer, Standardizer,
    UnivariateBoxCoxTransformer,
    OneHotEncoder


## IMPORTS

using MLJBase
using MLJModels

# to be extended:
import MLJBase: fit, update, clean!,
    predict, predict_mean, predict_median, predict_mode,
    transform, inverse_transform, se, evaluate, fitted_params,
    show_as_constructed, params
import MLJModels: models

using Requires
import Pkg.TOML
using OrderedCollections
using  CategoricalArrays
import Distributions: pdf, mode
import Distributions
import StatsBase
using ProgressMeter
import Tables
import PrettyTables
import Random
using ScientificTypes
import ScientificTypes

# convenience packages
using DocStringExtensions: SIGNATURES, TYPEDEF

# to be extended:
import Base.==
import StatsBase.fit!

# from Standard Library:
using Statistics
using LinearAlgebra
using Random
import Distributed: @distributed, nworkers, pmap
using RecipesBase # for plotting

const srcdir = dirname(@__FILE__) # the directory containing this file:
const CategoricalElement = Union{CategoricalString,CategoricalValue}


## INCLUDES

include("utilities.jl")     # general purpose utilities
include("machines.jl")    
include("networks.jl")      # for building learning networks
include("composites.jl")    # composite models & exporting learning networks
include("pipelines.jl")     # pipelines (exported linear learning networks)
include("operations.jl")    # syntactic sugar for operations (predict, etc)
include("resampling.jl")    # resampling strategies and model evaluation
include("parameters.jl")    # hyperparameter ranges and grid generation
include("tuning.jl")
include("ensembles.jl")     # homogeneous ensembles
include("tasks.jl")         # enhancements to MLJBase task interface 
include("scitypes.jl")      # extensions to ScientificTypes.sictype
include("plotrecipes.jl")


## INCLUDES FOR OPTIONAL DEPENDENCIES

function __init__()
    @require(CSV="336ed68f-0bac-5ca0-87d4-7b16caf5d00b",
             include("datasets_requires.jl"))
end


end # module
