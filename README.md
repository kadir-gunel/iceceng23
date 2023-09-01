# iceceng23
Repository for *"Model Agnostic Knowledge Transfer Methods for Sentence Embedding Models*" paper by *Gunel* and *Amasyali*.

## How to repeat the experiments 

1. Download [Julia](julia.com)
  - All experiments are done with Julia version 1.8
2. Install dependent packages
  - Knet
  - BSON
  - JLD2
  - CUDA
  - ... 
3. Download data
4. Execute 

## Data Loading for FastText Embeddings
Loading pre-trained FastText models (especially English - 2M by 300 vectors!!) takes too much time and space. Since there is no native way of loading them in Julia, I found a 2-step way: First you load the data by using ```readEmbeds``` function. This function returns a vocabulary and a matrix (word vectors). The second step is to write it in binary format and use ```Mmap``` utility for later usages. This is a one time thing but it saves tons of time and space. 

```readEmbeds``` function requires to have vocabulary and dimension size of the intended FastText model. 
To do so: 
1. Open ft-model.txt with an text editor (use nano since it gives you the number of lines(vocabulary) automatically)
2. Insert ```voc_size dim_size``` at top line
3. Save and exit


```julia
function readEmbeds(file; threshold=0, vocabulary=Nothing, dtype=Float32)
    @warn "This method reads the word embedding matrix in column major format"
    count, dims = parse.(Int64, split(readline(file), ' '))
    words  = String[]; # words in vocabulary
    matrix = isequal(vocabulary, Nothing) ?  Array{dtype}(undef, count, dims) : Array{Float32}[];

    # p = Progress(dims, 1, "Reading embeddig file: $file") # this is for progressbar
    for (i, line) in enumerate(drop(eachline(file), 1))
        mixed = split(chomp(line), " ")
        if vocabulary == Nothing
            push!(words, mixed[1])
            matrix[i, :] .= parse.(dtype, mixed[2:end])
        elseif in(mixed[1], vocabulary)
            push!(words, mixed[1])
            push!(matrix, parse.(dtype, mixed[2:end]))
        end
        # next!(p)
    end
    words, Matrix(permutedims(matrix))
end
```

Use ```writeBinaryEmbeddings``` to write it to the disk.

```julia
using Mmap

function writeBinaryEmbeddings(file::String, WE::Matrix, V::Array{String})
    d, w = size(WE);
    @info "Creating .bin for Word Embeddings"
    if d > w
        @warn "Permuting the Embedding matrix to raw major form"
        WE = Matrix(permutedims(WE))
    end

    s = open(file * "_WE.bin", "w+")
    write(s, d)
    write(s, w)
    write(s, WE)
    close(s)

    @info "Creating .txt for Vocabulary"
    s = open(file * "_voc.txt", "w+")
    for word in V
        write(s, word*"\n")
    end
    close(s)
    @info "Files are written by adding '_WE.bin' to the given file name $file "
end
```

Use ```readBinaryEmbeddings``` to load the saved vocabulary file and word vectors.
```julia
using Mmap

function readBinaryEmbeddings(file::String; atype=Float32)
    @info "Reading Word Embedding file"
    s = open(file * "_WE.bin")   # default is read-only
    m = read(s, Int)
    n = read(s, Int)
    WE = Mmap.mmap(s, Matrix{atype}, (m,n))
    close(s)

    @info "Reading vocabulary file"
    V = readlines(file * "_voc.txt")
    return V, Matrix(permutedims(WE))
end
```






