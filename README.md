# iceceng23
Code samples in the Appendix part of *Model Agnostic Knowledge Transfer Methods for Sentence Embedding Models* paper


## Appendix 1: Orthogonal Matrix Properties
### Property 1

```julia
function mapOrthogonal(From, To)
  U, S, Vt = svd(From * To’) 
  W = (V * U’)
  return W
end

W = mapOrthogonal(FastText, sBert)
W_ft = W * Fasttext
W_sbert = W' * FastText

```



