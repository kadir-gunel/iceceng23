# iceceng23
Code samples in the Appendix part of *Model Agnostic Knowledge Transfer Methods for Sentence Embedding Models* paper

For data you can either download the provided embedding spaces or create your own random data : 
1. download the provided embedding space
2. create your own random data 

```julia
using BSON: @load
using Random

# @load "/path/to/FT-sentence-embeddings/" FT
# @load "/path/to/sBert-sentence-embeddings/" sBert

# FT = rand(300, 30_000)
# sBert = rand(768, 30_000)

```


## Appendix 1: Orthogonal Matrix Properties
### Property 1

```julia
function mapOrthogonal(From, To)
  F = svd(From * To’) 
  W = (F.V * F.U’)
  return W
end

W = mapOrthogonal(FastText, sBert)
W_ft = W * Fasttext
W_sbert = W' * FastText

```


### Property 2
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
```julia
function findAlignments(X, Y)
  # X and Y have equal number of samples, their dimensions can be different 
  xsim = X’ * X # n by n matrix 
  ysim = Y’ * Y
  sort!(ysim, dims=1) sort!(xsim, dims=1)
  sim = xsim’ * ysim; 
  sim = knn(sim, k=10)
  # find the most appropriate neighbor by averaging top 10 samples
  src_idx = getindex(argmax(sim, dims=1))
  trg_idx = getindex(argmax(sim, dims=2))
  # list of highest index for source/target
  return src_idx, trg_idx 
end
```
