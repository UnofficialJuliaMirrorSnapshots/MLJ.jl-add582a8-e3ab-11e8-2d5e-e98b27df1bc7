# We wish to extend operations to identically named methods dispatched
# on `Machine`s and `NodalMachine`s. For example, we have from the model API
#
# `predict(model::M, fitresult, X) where M<:Supervised`
#
# but want also want
#
# `predict(machine::Machine, X)` where `X` is data
#
# and "networks.jl" requires us to define
#
# `predict(machine::NodalMachine, X)` where `X` is data
#
# and we would like the syntactic sugar (for `X` a node):
#
# `predict(machine::NodalMachine, X::Node)=node(predict, machine, X)`
#
# (If an operation has zero arguments, we cannot achieve the last
# desire because of ambiguity with the preceding one.)
#
# The following macros are for this purpose.

## TODO: need to add checks on the arguments of
## predict(::AbstractMachine, ) and transform(::AbstractMachine, )
## Refactor using simpler non-macro code generation

macro extend_to_machines(operation)
    quote

        # with arguments specified:
        function $(esc(operation))(machine::AbstractMachine, args...)
            if isdefined(machine, :fitresult)
                return $(esc(operation))(machine.model,
                                         machine.fitresult,
                                         args...)
            else
                throw(error("$machine has not been trained."))
            end
        end
        $(esc(operation))(machine::Machine; rows=:) =
            $(esc(operation))(machine, selectrows(machine.args[1], rows))
        $(esc(operation))(machine::Machine, task::MLJTask) =
            $(esc(operation))(machine, task.X)

    end
end

macro sugar(operation)
    quote
        $(esc(operation))(machine::NodalMachine, args::AbstractNode...) =
            node($(esc(operation)), machine, args...)
    end
end

@extend_to_machines predict
@extend_to_machines predict_mode
@extend_to_machines predict_mean
@extend_to_machines predict_median
@extend_to_machines transform
@extend_to_machines inverse_transform
@extend_to_machines se

@sugar predict
@sugar predict_mode
@sugar predict_mean
@sugar predict_median
@sugar transform
@sugar inverse_transform
@sugar se


# the zero argument special cases:
function evaluate(machine::AbstractMachine)
    if isdefined(machine, :fitresult)
        return evaluate(machine.model, machine.fitresult)
    else
        throw(error("$machine has not been trained."))
    end
end
function fitted_params(machine::AbstractMachine)
    if isdefined(machine, :fitresult)
        return fitted_params(machine.model, machine.fitresult)
    else
        throw(error("$machine has not been trained."))
    end
end

getindex(n::Node{<:NodalMachine{<:Model}}, s::Symbol) = getproperty(n.machine.model, s)
setindex!(n::Node{<:NodalMachine{<:Model}}, v, s::Symbol) = setproperty!(n.machine.model, s, v)
