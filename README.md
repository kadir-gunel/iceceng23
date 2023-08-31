# iceceng23
Code samples in the Appendix part of *Model Agnostic Knowledge Transfer Methods for Sentence Embedding Models* paper by *K. Gunel* and *M. Fatih Amasyali*.

- The code given below is written in [Julia](https://julialang.org/downloads/). 
- You need to install [BSON.jl](https://github.com/JuliaIO/BSON.jl) for loading provided data.

## Note 
- Please note that the code samples below are for demonstration purposes only.
- The shared code can work between different dimensional data :
  - provided data (see WMT Vectors folder)
  - on your data (as long as it is in matrix form)


## Toy Data
For data you can either: 
1. download the provided embedding space
2. create your own random data 

```julia
# if BSON not install uncomment the following 2 lines
# using Pkg
# Pkg.add("BSON")

using LinearAlgebra
using BSON: @load
using Random

# uncomment one of the following: option 1/2 for loading data

# option 1
# @load "/path/to/FT_first_30k_sentences.bson" x
# FT = x |> copy 
# @load "/path/to/sBert_first_30k_sentences.bson" x
# sBert = x |> copy


# option 2
# FT = rand(300, 30_000)
# sBert = rand(768, 30_000)

```


## Appendix 1: Orthogonal Matrix Properties
### Property 1

The property of the W rotation matrix is that, since it is orthogonal, its transpose (W^T) can rotate sBert sentence embeddings towards FastText space. This operation can be used as a supervised dimensionality reduction operation.


```julia
function mapOrthogonal(From, To)
  F = svd(From * To') 
  W = (F.V * F.U')
  return W
end

W = mapOrthogonal(FastText, sBert)
W_ft = W * Fasttext
W_sbert = W' * FastText

```


### Property 2
The property of the rotation matrix W comes from, again, its orthogonality. Summing any number of orthogonal matrices equals again another orthogonal matrix (WGlobal). Our experiments for rotation uses this property, since it gives better rotation results compared to single rotation matrix.

```julia
function orthogonalProperty2(FastText, sBert)
  Ws = [] # hold every rotation matrix W
  G = 100
  # each group has 100 sentences -> 3_000 orthogonal models will be created
  for i in 1:G:300_000
    rng = i:i+G
    W, _ = mapOrthogonal(FastText[:, rng], sBert[:, rng])
    push!(Ws, W) # adds W to rotation matrix list Ws 
   end
  W = sum(Ws) 
  return W
end
```

## Appendix 2: Alignment Procedure
Since embedding dimensions of both models are different from each other, it is decided to use an alignment search function which can work on sample space.
```topk_mean``` function is similar to Pytorch's ```torch.topk``` method.

```julia
function topk_mean(sim, k; inplace=false)
    n = size(sim, 1)
    ans_ = (zeros(eltype(sim), n, 1)) |> typeof(sim)
    if k <= 0
        return ans_
    end
    if !inplace
        sim = deepcopy(sim)
    end
    min_ = findmin(sim)[1]
    for i in 1:k
        vals_idx = findmax(sim, dims=2);
        ans_ += vals_idx[1]
        sim[vals_idx[2]] .= min_
    end
    return ans_ / k
end
```

```csls``` function is a special version of k-nearest neighbor which was developed for overcoming the hubness problem of knn  [[1]](#1). 

```julia
function csls(sim; k::Int64=10)
    knn_sim_fwd = topk_mean(sim, k);
    knn_sim_bwd = topk_mean(permutedims(sim), k);
    sim -= ones(eltype(sim), size(sim)) .* (knn_sim_fwd / 2) + ones(eltype(sim), size(sim)) .* ((knn_sim_bwd / 2));
end
```



```julia
function findAlignments(X, Y)
  # X and Y have equal number of samples, their dimensions can be different
  samples = size(X, 2)
  xsim = X' * X # n by n matrix 
  ysim = Y' * Y
  sort!(ysim, dims=1)
  sort!(xsim, dims=1)
  sim = xsim' * ysim; 
  sim = csls(sim, k=10) # this is special version of k-nearest neigbor for overcoming hubness problem
  # find the most appropriate neighbor by averaging top 10 samples
  src_idx = vcat(collect(1:samples), permutedims(getindex.(argmax(sim, dims=1), 1)))
  trg_idx = vcat((getindex.(argmax(sim, dims=2),2)), collect(1:samples))
  # list of highest index for source/target
  return vec(src_idx), vec(trg_idx)
end
```
### Additional Functions
You may also want to apply different types of normalization to your data such as unit, centering, and whitening. Below you can find functions for them:

```julia
function whitening(M::Matrix{Float32})
    F = svd(M)
    F.V * diagm(1 ./ F.S) * (F.Vt)
end
```

```julia
center(matrix::T) where {T} = matrix .- (vec(mean(matrix, dims=2)))

function unit(matrix::T) where {T}
    norms = p_norm(matrix)
    norms[findall(isequal(0), norms)] .= 1
    return matrix ./ norms
end
```
In order to apply different combinations you can use piping : 

```julia
x_normalized = x |> unit |> center |> unit
```

## References

<a id="1">[1]</a> 
Alexis Conneau and Guillaume Lample and Marc'Aurelio Ranzato and Ludovic Denoyer and Herve Jegou: 
Word Translation Without Parallel Data, 
CoRR,
doi:[/1710.04087](https://doi.org/10.48550/arXiv.1710.04087),
2017.
