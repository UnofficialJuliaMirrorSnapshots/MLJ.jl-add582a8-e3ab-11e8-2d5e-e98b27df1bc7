module Registry

# this module 

import Pkg.TOML
using InteractiveUtils

# for testings decoding of metadata:
import MLJBase: Found, Continuous, Finite, Infinite
import MLJBase: OrderedFactor, Count, Multiclass, Binary

const srcdir = dirname(@__FILE__) # the directory containing this file
const environment_path = joinpath(srcdir, "..")


## METHODS TO GENERATE METADATA AND WRITE TO ARCHIVE

function finaltypes(T::Type)
    s = InteractiveUtils.subtypes(T)
    if isempty(s)
        return [T, ]
    else
        return reduce(vcat, [finaltypes(S) for S in s])
    end
end

const project_toml = joinpath(srcdir, "../Project.toml")
const packages = map(Symbol, keys(TOML.parsefile(project_toml)["deps"])|>collect)
filter!(packages) do pkg
    !(pkg in [:MLJ, :MLJBase, :MLJModels, :InteractiveUtils, :Pkg])
end

const package_import_commands =  [:(import $pkg) for pkg in packages]

macro update()
    mod = __module__
    _update(mod, false)
end

macro update(ex)
    mod = __module__
    test_env_only = eval(ex)
    test_env_only isa Bool || "b in @update(b) must be Bool. "
    _update(mod, test_env_only)
end

function _update(mod, test_env_only)

    test_env_only && @info "Testing registry environment only. "
    
    program1 = quote
        @info "Packages to be searched for model implementations:"
        for pkg in $packages
            println(pkg)
        end
        using Pkg
        Pkg.activate($environment_path)
        @info "resolving registry environment..."
        Pkg.resolve()
    end

    program2 = quote

        @info "Instantiating registry environment..."
        Pkg.instantiate()

        @info "Loading registered packages..."
        import MLJBase
        import MLJModels
        using Pkg.TOML
        
        # import the packages
        $(Registry.package_import_commands...)

        @info "Generating model metadata..."

        modeltypes = MLJ.Registry.finaltypes(MLJBase.Model)
        
        # generate and write to file the model metadata:
        packages = string.(MLJ.Registry.packages)
        meta_given_package = Dict()
        for pkg in packages
            meta_given_package[pkg] = Dict()
        end
        for M in modeltypes
            _MLJBase.info = MLJBase.info(M)
            pkg = _MLJBase.info[:package_name]
            if !(pkg in ["unknown", "MLJ"]) 
                modelname = _MLJBase.info[:name]
                meta_given_package[pkg][modelname] = _MLJBase.info
            end
        end
        open(joinpath(MLJ.Registry.srcdir, "../Metadata.toml"), "w") do file
            TOML.print(file, MLJ.encode_dic(meta_given_package))
        end
        
        # generate and write to file list of models for each package:
        models_given_pkg = Dict()
        for pkg in packages
            models_given_pkg[pkg] = collect(keys(meta_given_package[pkg]))
        end
        open(joinpath(MLJ.Registry.srcdir, "../Models.toml"), "w") do file
            TOML.print(file, models_given_pkg)
        end

        :(println("Local Metadata.toml updated."))

    end

    mod.eval(program1)
    test_env_only || mod.eval(program2)

    println("If you have called @update from the REPL then your namespace "* 
            "is now polluted. Restart your REPL. ")

    true
end

end # module
