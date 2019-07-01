module TestUtilities

# using Revise
using Test
using MLJ


d = Dict(
    :x => Set([1, 2]),
    :y => Set([2, 3, 5]),
    :z => Set([4, 7]),
    :a => Set([8, 1]),
    :b => Set([4,]),
    :w => Set([3, 1, 2]),
    :t => Set([0,]))

dinv = Dict(
    0 => Set([:t,]),
    1 => Set([:x, :a, :w]),
    2 => Set([:x, :y, :w]),
    3 => Set([:y, :w]),
    4 => Set([:z, :b]),
    5 => Set([:y,]),
    7 => Set([:z,]),
    8 => Set([:a,]))

@test MLJ.inverse(d) == dinv

d = Dict()
d[:test] = Tuple{Union{Continuous,Missing}, Finite}
d["junk"] = Dict{Any,Any}("H" => Missing, :cross => "lemon", :t => :w, "r" => "r")
d["a"] = "b"
d[:f] = true
d["j"] = :post

@test MLJ.decode_dic(MLJ.encode_dic(d)) == d

true

end # module
true
