## ABSTRACT NODES AND SOURCE NODES

abstract type AbstractNode <: MLJType end

"""
$TYPEDEF

A Source node wraps training data in a format such as a table or a categorical vector.
"""
mutable struct Source <: AbstractNode
    data  # training data
end

is_stale(s::Source) = false

# make source nodes callable:
function (s::Source)(; rows=:)
    rows == (:) && return s.data
    return selectrows(s.data, rows)
end
(s::Source)(Xnew) = Xnew

"""
$SIGNATURES

Attach new data `X` to an existing source node `s`.
"""
function rebind!(s::Source, X)
    s.data = X
    return s
end


"""
$SIGNATURES

Return a list of all origins of a node `N` accessed by a call `N()`.
These are the source nodes of the acyclic directed graph (DAG)
associated with the learning network terminating at `N`, if edges
corresponding to training arguments are excluded. A `Node` object
cannot be called on new data unless it has a unique origin.

Not to be confused with `sources(N)` which refers to the same graph
but without the training edge deletions.

See also: [`node`](@ref), [`source`](@ref).
"""
origins(s::Source) = [s,]


"""
$SIGNATURES

Incrementally append to `nodes1` all elements in `nodes2`, excluding any elements
previously added (or any element of `nodes1` in its initial state).
"""
function _merge!(nodes1, nodes2)
    for x in nodes2
        if !(x in nodes1)
            push!(nodes1, x)
        end
    end
    return nodes1
end

# Note that `fit!` has already been defined for any  AbstractMachine in machines.jl

"""
$TYPEDEF

A NodalMachine wraps a model as part of a learning network.
"""
mutable struct NodalMachine{M<:Model} <: AbstractMachine{M}

    model::M
    previous_model::M
    fitresult
    cache
    args::Tuple{Vararg{AbstractNode}}
    report
    frozen::Bool
    rows            # for remembering the rows used in last call to `fit!`
    state::Int      # number of times fit! has been called on machine
    upstream_state  # for remembering the upstream state in last call to `fit!`

    function NodalMachine{M}(model::M, args::AbstractNode...) where M<:Model

        # check number of arguments for model subtypes:
        !(M <: Supervised) || length(args) > 1 ||
            throw(error("Wrong number of arguments. " *
                        "You must provide target(s) for supervised models."))

        !(M <: Unsupervised) || length(args) == 1 ||
            throw(error("Wrong number of arguments. " *
                        "Use NodalMachine(model, X) for an unsupervised model."))

        machine = new{M}(model)
        machine.frozen = false
        machine.state = 0
        machine.args = args

        machine.upstream_state = Tuple([state(arg) for arg in args])

        return machine
    end
end

# automatically detect type parameter:
NodalMachine(model::M, args...) where M<:Model = NodalMachine{M}(model, args...)

"""
$SIGNATURES

Freeze the machine `machine` so that it will never be retrained (unless it is thawed).
See also [`thaw!`](@ref).
"""
function freeze!(machine::NodalMachine)
    machine.frozen = true
end

"""
$SIGNATURES

Unfreeze the machine `machine` so that it can be retrained.
See also [`freeze!`](@ref).
"""
function thaw!(machine::NodalMachine)
    machine.frozen = false
end

"""
$SIGNATURES

Check if a machine is stale.
"""
function is_stale(machine::NodalMachine)
    !isdefined(machine, :fitresult) ||
        machine.model != machine.previous_model ||
        reduce(|,[is_stale(arg) for arg in machine.args])
end

"""
$SIGNATURES

Return the state of a machine.
"""
state(machine::NodalMachine) = machine.state


## NODES

struct Node{T<:Union{NodalMachine, Nothing}} <: AbstractNode

    operation   # that can be dispatched on a fit-result (eg, `predict`) or a static operation
    machine::T  # is `nothing` for static operations
    args::Tuple{Vararg{AbstractNode}}  # nodes where `operation` looks for its arguments
    origins::Vector{Source}
    nodes::Vector{AbstractNode}  # all upstream nodes, order consistent with DAG order (the node "tape")

    function Node{T}(operation,
                     machine::T,
                     args::AbstractNode...) where {M<:Model, T<:Union{NodalMachine{M},Nothing}}

        # check the number of arguments:
        if machine === nothing && isempty(args)
            throw(error("`args` in `Node(::Function, args...)` must be non-empty. "))
        end

        origins_ = unique(vcat([origins(arg) for arg in args]...))
        length(origins_) == 1 ||
            @warn "A node referencing multiple origins when called " *
                  "has been defined:\n$(origins_). "

        # initialize the list of upstream nodes:
        nodes_ = AbstractNode[]

        # merge the lists from arguments:
        for arg in args
            _merge!(nodes_, nodes(arg))
        end

        # merge the lists from training arguments:
        if machine !== nothing
            for arg in machine.args
                _merge!(nodes_, nodes(arg))
            end
        end

        return new{T}(operation, machine, args, origins_, nodes_)
    end
end

"""
$SIGNATURES

Access the origins (source nodes) of a given node.
"""
origins(X::Node) = X.origins

"""
$SIGNATURES

Return all nodes upstream of a node, including the node, in order consistent with the DAG.
"""
nodes(X::Node) = AbstractNode[X.nodes..., X]
nodes(S::Source) = AbstractNode[S, ]

"""
$SIGNATURES

Check if a node is stale.
"""
function is_stale(X::Node)
    (X.machine !== nothing && is_stale(X.machine)) ||
        reduce(|, [is_stale(arg) for arg in X.args])
end

state(s::MLJ.Source) = (state = 0, )

"""
$SIGNATURES

Return the state of a node.
"""
function state(W::MLJ.Node)
    mach = W.machine
    state_ = W.machine === nothing ? 0 : state(W.machine)
    if mach === nothing
        trainkeys   = []
        trainvalues = []
    else
        trainkeys   = (Symbol("train_arg", i) for i in eachindex(mach.args))
        trainvalues = (state(arg) for arg in mach.args)
    end
    keys = tuple(:state,
                 (Symbol("arg", i) for i in eachindex(W.args))...,
                 trainkeys...)
    values = tuple(state_,
                   (state(arg) for arg in W.args)...,
                   trainvalues...)
    return NamedTuple{keys}(values)
end

# autodetect type parameter:
Node(operation, machine::M, args...) where M <: Union{NodalMachine,Nothing} =
    Node{M}(operation, machine, args...)

# constructor for static operations:
Node(operation, args::AbstractNode...) = Node(operation, nothing, args...)

# make nodes callable:
(y::Node)(; rows=:) = (y.operation)(y.machine, [arg(rows=rows) for arg in y.args]...)
function (y::Node)(Xnew)
    length(y.origins) == 1 ||
        error("Nodes with multiple origins are not callable on new data. "*
              "Use origins(node) to inspect. ")
    return (y.operation)(y.machine, [arg(Xnew) for arg in y.args]...)
end

"""
$SIGNATURES

Allow nodes to share the `selectrows(X, r)` syntax of concrete tabular data (needed for
`fit(::AbstractMachine, ...)` in machines.jl).
"""
MLJBase.selectrows(X::AbstractNode, r) = X(rows=r)

# and for the special case of static operations:
(y::Node{Nothing})(; rows=:) = (y.operation)([arg(rows=rows) for arg in y.args]...)
(y::Node{Nothing})(Xnew) = (y.operation)([arg(Xnew) for arg in y.args]...)

"""
$SIGNATURES

Train the machines of all dynamic nodes in the learning network terminating at
`N` in an appropriate order.
"""
function fit!(y::Node; rows=nothing, verbosity::Int=1, force::Bool=false)
    if rows === nothing
        rows = (:)
    end

    # get non-source nodes:
    nodes_ = filter(nodes(y)) do n
        n isa Node
    end

    # get machines to fit:
    machines = map(n -> n.machine, nodes_)
    machines = filter(unique(machines)) do mach
        mach !== nothing
    end

    for mach in machines
        fit!(mach; rows=rows, verbosity=verbosity, force=force)
    end

    return y
end

# allow arguments of `Nodes` and `NodalMachine`s to appear
# at REPL:
istoobig(d::Tuple{AbstractNode}) = length(d) > 10

# overload show method

function _recursive_show(stream::IO, X::AbstractNode)
    if X isa Source
        printstyled(IOContext(stream, :color=>MLJBase.SHOW_COLOR), MLJBase.handle(X), color=:blue)
    else
        detail = (X.machine === nothing ? "(" : "($(MLJBase.handle(X.machine)), ")
        operation_name = typeof(X.operation).name.mt.name
        print(stream, operation_name, "(")
        if X.machine !== nothing
            color = (X.machine.frozen ? :red : :green)
            printstyled(IOContext(stream, :color=>MLJBase.SHOW_COLOR), MLJBase.handle(X.machine),
                        bold=MLJBase.SHOW_COLOR)
            print(stream, ", ")
        end
        n_args = length(X.args)
        counter = 1
        for arg in X.args
            _recursive_show(stream, arg)
            counter >= n_args || print(stream, ", ")
            counter += 1
        end
        print(stream, ")")
    end
end

function Base.show(stream::IO, ::MIME"text/plain", X::AbstractNode)
    id = objectid(X)
    description = string(typeof(X).name.name)
    str = "$description @ $(MLJBase.handle(X))"
    printstyled(IOContext(stream, :color=>MLJBase.SHOW_COLOR), str, color=:blue)
    if !(X isa Source)
        print(stream, " = ")
        _recursive_show(stream, X)
    end
end

function Base.show(stream::IO, ::MIME"text/plain", machine::NodalMachine)
    id = objectid(machine)
    description = string(typeof(machine).name.name)
    str = "$description @ $(MLJBase.handle(machine))"
    printstyled(IOContext(stream, :color=>MLJBase.SHOW_COLOR), str, bold=MLJBase.SHOW_COLOR)
    print(stream, " = ")
    print(stream, "machine($(machine.model), ")
    n_args = length(machine.args)
    counter = 1
    for arg in machine.args
        printstyled(IOContext(stream, :color=>MLJBase.SHOW_COLOR), MLJBase.handle(arg), bold=MLJBase.SHOW_COLOR)
        counter >= n_args || print(stream, ", ")
        counter += 1
    end
    print(stream, ")")
end

## SYNTACTIC SUGAR FOR LEARNING NETWORKS

"""
    Xs = source(X)

Defines a `Source` object out of data `X`. The data can be a vector,
categorical vector, or table. The calling behaviour of a source node is this:

    Xs() = X
    Xs(rows=r) = selectrows(X, r)  # eg, X[r,:] for a DataFrame
    Xs(Xnew) = Xnew

See also: [`origins`](@ref), [`node`](@ref).

"""
source(X) = Source(X) # here `X` is data
source(X::Source) = X

"""
    N = node(f::Function, args...)

Defines a `Node` object `N` wrapping a static operation `f` and arguments
`args`. Each of the `n` elements of `args` must be a `Node` or `Source`
object. The node `N` has the following calling behaviour:

    N() = f(args[1](), args[2](), ..., args[n]())
    N(rows=r) = f(args[1](rows=r), args[2](rows=r), ..., args[n](rows=r))
    N(X) = f(args[1](X), args[2](X), ..., args[n](X))

    J = node(f, mach::NodalMachine, args...)

Defines a dynamic `Node` object `J` wrapping a dynamic operation `f`
(`predict`, `predict_mean`, `transform`, etc), a nodal machine `mach` and
arguments `args`. Its calling behaviour, which depends on the outcome of
training `mach` (and, implicitly, on training outcomes affecting its
arguments) is this:

    J() = f(mach, args[1](), args[2](), ..., args[n]())
    J(rows=r) = f(mach, args[1](rows=r), args[2](rows=r), ..., args[n](rows=r))
    J(X) = f(mach, args[1](X), args[2](X), ..., args[n](X))

Generally `n=1` or `n=2` in this latter case.

    predict(mach, X::AbsractNode, y::AbstractNode)
    predict_mean(mach, X::AbstractNode, y::AbstractNode)
    predict_median(mach, X::AbstractNode, y::AbstractNode)
    predict_mode(mach, X::AbstractNode, y::AbstractNode)
    transform(mach, X::AbstractNode)
    inverse_transform(mach, X::AbstractNode)

Shortcuts for `J = node(predict, mach, X, y)`, etc.

Calling a node is a recursive operation which terminates in the call
to a source node (or nodes). Calling nodes on *new* data `X` fails unless the
number of such nodes is one.

See also: [`source`](@ref), [`origins`](@ref).

"""
node = Node

# unless no arguments are `AbstractNode`s, `machine` creates a
# NodalTrainableModel, rather than a `Machine`:
machine(model::Model, args::AbstractNode...) = NodalMachine(model, args...)
machine(model::Model, X, y::AbstractNode) = NodalMachine(model, source(X), y)
machine(model::Model, X::AbstractNode, y) = NodalMachine(model, X, source(y))

MLJBase.matrix(X::AbstractNode) = node(MLJBase.matrix, X)

Base.log(X::AbstractNode) = node(v->log.(v), X)
Base.exp(X::AbstractNode) = node(v->exp.(v), X)

import Base.+
+(y1::AbstractNode, y2::AbstractNode) = node(+, y1, y2)
+(y1, y2::AbstractNode) = node(+, y1, y2)
+(y1::AbstractNode, y2) = node(+, y1, y2)

import Base.*
*(lambda::Real, y::AbstractNode) = node(y->lambda*y, y)
