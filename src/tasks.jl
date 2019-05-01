function Base.getindex(task::SupervisedTask, r)
    X = selectrows(task.X, r)
    y = task.y[r]
    is_probabilistic = task.is_probabilistic
    input_scitypes = scitypes(X)
    input_scitype_union = Union{input_scitypes...}
    target_scitype_union = scitype_union(y)
    input_is_multivariate = task.input_is_multivariate
    return SupervisedTask(X,
                          y,
                          is_probabilistic,
                          input_scitypes,
                          input_scitype_union,
                          target_scitype_union,
                          input_is_multivariate)
end

function Random.shuffle!(rng::AbstractRNG, task::SupervisedTask)
    rows = shuffle!(rng, Vector(1:nrows(task)))
    task.X = selectrows(task.X, rows)                    
    task.y = selectrows(task.y, rows)                    
    return task
end

function Random.shuffle!(task::SupervisedTask)
    rows = shuffle!(Vector(1:nrows(task)))
    task.X = selectrows(task.X, rows)                    
    task.y = selectrows(task.y, rows)                    
    return task
end

Random.shuffle(rng::AbstractRNG, task::SupervisedTask) = task[shuffle!(rng, Vector(1:nrows(task)))]
Random.shuffle(task::SupervisedTask) = task[shuffle!(Vector(1:nrows(task)))]



    



    
