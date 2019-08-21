## TRAITS FOR SCORES AND LOSS FUNCTIONS

target_scitype(measure) = Unknown
prediction_type(measure) = :unkown # other options are :probabilistic, :deterministic, or :interval
orientation(measure) = :loss  # other options are :score, :other
reports_each_observation(measure) = false
is_feature_dependent(measure) = false
supports_weights(measure) = false


# Note: target_scitype and supports_weights already defined in
# MLJBase, but only for <:Model


## EVALUATION 

# yhat - predictions (point or probabilisitic)
# X - features
# y - target observations
# w - per-observation weights

# Note that the following methods are unspecialized to any particular
# measure. They can be overloaded; see eg. src/loss_functions_extension.jl

value(measure, yhat, X, y, w) = value(measure, yhat, X, y, w,
                                      Val(is_feature_dependent(measure)),
                                      Val(supports_weights(measure)))

#  is feature independent, weights not supported:
value(measure, yhat, X, y, w, ::Val{false}, ::Val{false}) = measure(yhat, y)

#  is feature dependent:, weights not supported:
value(measure, yhat, X, y, w, ::Val{true}, ::Val{false}) = measure(yhat, X, y)


#  is feature independent, weights supported:
value(measure, yhat, X, y, w, ::Val{false}, ::Val{true}) = measure(yhat, y, w)
value(measure, yhat, X, y, ::Nothing, ::Val{false}, ::Val{true}) = measure(yhat, y)

#  is feature dependent, weights supported:
value(measure, yhat, X, y, w, ::Val{true}, ::Val{true}) = measure(yhat, X, y, w)
value(measure, yhat, X, y, ::Nothing, ::Val{true}, ::Val{true}) = measure(yhat, X, y)


## HELPERS

"""
$SIGNATURES

Check that two vectors have compatible dimensions
"""
function check_dimensions(ŷ::AbstractVector, y::AbstractVector)
    length(y) == length(ŷ) ||
        throw(DimensionMismatch("Differing numbers of observations and "*
                                "predictions. "))
    return nothing
end

function check_pools(ŷ, y)
              y[1].pool.index == levels(ŷ[1])[1].pool.index ||
              error("Conflicting categorical pools found "*
                    "in observations and predictions. ")
    return nothing
end

              
## FOR BUILT-IN MEASURES

abstract type Measure end

Base.show(stream::IO, ::MIME"text/plain", m::Measure) = print(stream, "$(measurename(m)) (callable Measure)")
Base.show(stream::IO, m::Measure) = print(stream, measurename(m))

"""
    info(measure::Measure)

Return a named tuple summarizing the traits defined for `measure`. 

"""
MLJBase.info(measure::Measure) = (target_scitype=target_scitype(measure),
                            prediction_type=prediction_type(measure),
                            orientation=orientation(measure),
                            reports_each_observation=reports_each_observation(measure),
                            is_feature_dependent=is_feature_dependent(measure),
                            supports_weights=supports_weights(measure))


## REGRESSOR METRICS (FOR DETERMINISTIC PREDICTIONS)

mutable struct MAV<: Measure end
"""
    mav(ŷ, y)
    mav(ŷ, y, w)

Mean absolute error (also known as MAE).

``\\text{MAV} =  n^{-1}∑ᵢ|yᵢ-ŷᵢ|`` or ``\\text{MAV} =  ∑ᵢwᵢ|yᵢ-ŷᵢ|/∑ᵢwᵢ``

"""
mav = MAV()
measurename(::MAV) = "mav"

target_scitype(::MAV) = Continuous
prediction_type(::MAV) = :deterministic
orientation(::MAV) = :loss
reports_each_observation(::MAV) = false
is_feature_dependent(::MAV) = false
supports_weights(::MAV) = true

function (::MAV)(ŷ, y)
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = y[i] - ŷ[i]
        ret += abs(dev)
    end
    return ret / length(y)
end

function (::MAV)(ŷ, y, w)
    check_dimensions(ŷ, y)
    check_dimensions(y, w)
    ret = 0.0
    for i in eachindex(y)
        dev = w[i]*(y[i] - ŷ[i])
        ret += abs(dev)
    end
    return ret / sum(w)
end

# synonym
"""
mae(ŷ, y)

See also [`mav`](@ref).
"""
const mae = mav


struct RMS <: Measure end
"""
    rms(ŷ, y)
    rms(ŷ, y, w)

Root mean squared error:

``\\text{RMS} = \\sqrt{n^{-1}∑ᵢ|yᵢ-ŷᵢ|^2}`` or ``\\text{RMS} = \\sqrt{\\frac{∑ᵢwᵢ|yᵢ-ŷᵢ|^2}{∑ᵢwᵢ}}``
"""
rms = RMS()
measurename(::RMS) = "rms"

target_scitype(::RMS) = Continuous #AbstractVector{Continuous}
prediction_type(::RMS) = :deterministic
orientation(::RMS) = :loss
reports_each_observation(::RMS) = false
is_feature_dependent(::RMS) = false
supports_weights(::RMS) = true

function (::RMS)(ŷ, y)
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = y[i] - ŷ[i]
        ret += dev * dev
    end
    return sqrt(ret / length(y))
end

function (::RMS)(ŷ, y, w)
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = y[i] - ŷ[i]
        ret += w[i]*dev*dev
    end
    return sqrt(ret / sum(w))
end

struct L2 <: Measure end
"""
    l2(ŷ, y)
    l2(ŷ, y, w)

L2 per-observation loss.

"""
l2 = L2()
measurename(::L2) = "l2"

target_scitype(::L2) = Continuous # AbstractVector{Continuous}
prediction_type(::L2) = :deterministic
orientation(::L2) = :loss
reports_each_observation(::L2) = true
is_feature_dependent(::L2) = false
supports_weights(::L2) = true

(::L2)(ŷ, y) = (check_dimensions(ŷ, y); (y - ŷ).^2)

function (::L2)(ŷ, y, w)
    check_dimensions(ŷ, y)
    check_dimensions(w, y)
    return (y - ŷ).^2 .* w ./ (sum(w)/length(y)) 
end

struct L1 <: Measure end
"""
    l1(ŷ, y)
    l1(ŷ, y, w)

L1 per-observation loss.

"""
l1 = L1()
measurename(::L1) = "l1"

target_scitype(::L1) = Continuous # AbstractVector{Continuous}
prediction_type(::L1) = :deterministic
orientation(::L1) = :loss
reports_each_observation(::L1) = true
is_feature_dependent(::L1) = false
supports_weights(::L1) = true

(::L1)(ŷ, y) = (check_dimensions(ŷ, y); abs.(y - ŷ))

function (::L1)(ŷ, y, w)
    check_dimensions(ŷ, y)
    check_dimensions(w, y)
    return abs.(y - ŷ) .* w ./ (sum(w)/length(y))
end

struct RMSL <: Measure end
"""
    rmsl(ŷ, y)

Root mean squared logarithmic error:

``\\text{RMSL} = n^{-1}∑ᵢ\\log\\left({yᵢ \\over ŷᵢ}\\right)``

See also [`rmslp1`](@ref).
"""
rmsl = RMSL()
measurename(::RMSL) = "rmsl"

target_scitype(::RMSL) = Continuous # AbstractVector{Continuous}
prediction_type(::RMSL) = :deterministic
orientation(::RMSL) = :loss
reports_each_observation(::RMSL) = false
is_feature_dependent(::RMSL) = false
supports_weights(::RMSL) = false

function (::RMSL)(ŷ, y)
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = log(y[i]) - log(ŷ[i])
        ret += dev * dev
    end
    return sqrt(ret / length(y))
end


struct RMSLP1 <: Measure end
"""
    rmslp1(ŷ, y)

Root mean squared logarithmic error with an offset of 1:

``\\text{RMSLP1} = n^{-1}∑ᵢ\\log\\left({yᵢ + 1 \\over ŷᵢ + 1}\\right)``

See also [`rmsl`](@ref).
"""
rmslp1 = RMSLP1()
measurename(::RMSLP1) = "rmslp1"

target_scitype(::RMSLP1) = Continuous # AbstractVector{Continuous}
prediction_type(::RMSLP1) = :deterministic
orientation(::RMSLP1) = :loss
reports_each_observation(::RMSLP1) = false
is_feature_dependent(::RMSLP1) = false
supports_weights(::RMSLP1) = false

function (::RMSLP1)(ŷ, y)
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = log(y[i] + 1) - log(ŷ[i] + 1)
        ret += dev * dev
    end
    return sqrt(ret / length(y))
end

struct RMSP <: Measure end
"""
    rmsp(ŷ, y)

Root mean squared percentage loss:

``\\text{RMSP} = m^{-1}∑ᵢ \\left({yᵢ-ŷᵢ \\over yᵢ}\\right)^2``

where the sum is over indices such that `yᵢ≂̸0` and `m` is the number of such indices.
"""
rmsp = RMSP()
measurename(::RMSP) = "rmsp"

target_scitype(::RMSP) = Continuous #AbstractVector{Continuous}
prediction_type(::RMSP) = :deterministic
orientation(::RMSP) = :loss
reports_each_observation(::RMSP) = false
is_feature_dependent(::RMSP) = false
supports_weights(::RMSP) = false

function (::RMSP)(ŷ, y)
    check_dimensions(ŷ, y)
    ret = 0.0
    count = 0
    for i in eachindex(y)
        if y[i] != 0.0
            dev = (y[i] - ŷ[i])/y[i]
            ret += dev * dev
            count += 1
        end
    end
    return sqrt(ret/count)
end


## CLASSIFICATION METRICS (FOR DETERMINISTIC PREDICTIONS)

struct MisclassificationRate <: Measure end
misclassification_rate = MisclassificationRate()
measurename(::MisclassificationRate) = "misclassification_rate"

target_scitype(::MisclassificationRate) = Finite # AbstractVector{<:Finite}
prediction_type(::MisclassificationRate) = :deterministic
orientation(::MisclassificationRate) = :loss
reports_each_observation(::MisclassificationRate) = false
is_feature_dependent(::MisclassificationRate) = false
supports_weights(::MisclassificationRate) = true

(::MisclassificationRate)(ŷ, y) = mean(y .!= ŷ)
(::MisclassificationRate)(ŷ, y, w) = sum((y .!= ŷ) .*w) / sum(w)


## CLASSIFICATION METRICS (FOR PROBABILISTIC PREDICTIONS)

struct CrossEntropy <: Measure end
cross_entropy = CrossEntropy()
measurename(::CrossEntropy) = "cross_entropy"

target_scitype(::CrossEntropy) = Finite # AbstractVector{<:Finite}
prediction_type(::CrossEntropy) = :probabilistic
orientation(::CrossEntropy) = :loss
reports_each_observation(::CrossEntropy) = true
is_feature_dependent(::CrossEntropy) = false
supports_weights(::CrossEntropy) = false

# for single pattern:
_cross_entropy(d::UnivariateFinite, y) = -log(d.prob_given_level[y])

function (::CrossEntropy)(ŷ, y)
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)          
    return broadcast(_cross_entropy, Any[ŷ...], y)
end


## DEFAULT MEASURES

default_measure(model::M) where M<:Supervised =
    default_measure(model, target_scitype_union(M))
default_measure(model, ::Any) = nothing
default_measure(model::Deterministic, ::Type{<:Union{Continuous,Count}}) = rms
# default_measure(model::Probabilistic, ::Type{<:Union{Continuous,Count}}) =
default_measure(model::Deterministic, ::Type{<:Finite}) =
    misclassification_rate
default_measure(model::Probabilistic, ::Type{<:Finite}) = cross_entropy


