# iceceng23
Code samples in the Appendix part of *Model Agnostic Knowledge Transfer Methods for Sentence Embedding Models* paper

For data you can either download the provided embedding spaces or create your own random data : 
1. download the provided embedding space
2. create your own random data 

```julia
using BSON: @load
using Random

# uncomment one of the following of option 1/2 for loading data

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
  F = svd(From * To’) 
  W = (F.V * F.U’)
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
Since embedding dimensions of both models are different from each other, it is de-cided to use an alignment search function which can work on sample space.


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
