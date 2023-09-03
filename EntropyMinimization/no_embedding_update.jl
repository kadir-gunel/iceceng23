using Base.Iterators
using Base.GC

using SparseArrays
using LinearAlgebra
using Statistics
using Random
using Printf

using CUDA
using Knet
using Knet: update!
using StatsBase

using BSON: @load

Random.seed!(1234)


global atype = Knet.gpu(0) == -1 ? Array : KnetArray

"""
This reading method is sequential and consumes too much memory.
"""
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



# helpers for creating batches
function crp2int(corpus::Array{String}, s2i::Dict)
    intcrp = Array{Int64,1}[]
    for line in corpus
        aux = Int64[]
        for word in split(line)
            haskey(s2i, word) ? push!(aux, s2i[word]) : push!(aux, s2i["<UNK>"])
        end
        push!(intcrp, aux)
    end
    return intcrp
end

function chop!(corpora::Array; maxSeqLen::Int64=70)
    longer_idx = findall(i -> length(i) > maxSeqLen, corpora)
    for id in longer_idx
        keepat!(corpora[id], collect(1:maxSeqLen))
    end
end

function padding!(corpus::Array, pad_tag_id::Int64; maxSeqLen::Int64=70)
    maxlen = maximum(length.(corpus))
    maxSeqLen = maxlen < maxSeqLen ? maxlen : maxSeqLen
    idx_pad = findall(i -> length(i) < maxSeqLen , corpus)
    if !isempty(idx_pad)
        for sent in idx_pad
            diff = maxlen - length(corpus[sent])
            for i in 1:diff
                insert!(corpus[sent], 1, pad_tag_id)
            end
        end
    end
end

function dynamicBatch!(corpus64::Array, pad_tag_id::Int64; maxSeqLen::Int64=70, bsize=32)
    sort!(corpus64, by=length)
    chop!(corpus64, maxSeqLen=maxSeqLen)
    batches = collect(Iterators.partition(corpus64, bsize))
    for batch in batches
        padding!(collect(batch), pad_tag_id)
    end
    return batches
end

pickSentences(tot_lines::Int64; k::Int64=Int(64e3)) = (collect(1:tot_lines) |> shuffle)[1:k]

#============== Model Functions =================#

init(d...) = atype(xavier(Float32, d...))
initstate(bsize, hidden) = atype(zeros(Float32, bsize, hidden))

function initModel(hidden, emb_size)
    model = Dict{Symbol, Any}()
    params = (:hidden, :z, :r)
    for param in params
        model[(Symbol(:W_, param))] = init(hidden + emb_size, hidden)
        model[(Symbol(:b_, param))] = init(1, hidden)
    end
    return model # later add ft emebddings as embedding layer
end

function outputModel(hidden, vocsize)
    model = Dict{Symbol, Any}();
    model[:W] = init(hidden, vocsize)
    model[:b] = init(1, vocsize)
    return model
end

function gru(model, hidden, input)
    x = hcat(hidden, input) # (bsize, 300) + (bsize, 768) = (bsize, 1068)
    z = sigm.(x * model[:W_z] .+ model[:b_z]) # (bsize, 768)
    r = sigm.(x * model[:W_r] .+ model[:b_r]) # (bsize, 1068) * (1068, 768) = (bsize, 768)
    x = hcat((r .* hidden), input) # (bsize, 768) + (bsize, 300) = (bsize, 1068)
    h = tanh.(x * model[:W_hidden] .+ model[:b_hidden]) # (bsize, 1068) * (1068, 768) = (bsize, 768)
    hidden =  ((1 .- z) .* hidden) .+ (z .* h) # (bsize, 768)
    return hidden
end


function applyMask(hidden, xt) # x @ time t
    mask = .!(xt .== vocsize) # vocsize length is the padding number in dictionary
    M = atype{Float32,2}(fill(1.0, size(hidden))) # (bsize, 768)
    M = M .* atype{Float32}(mask) # (bsize, 768)
    return hidden .* M # hence we clear all paddings states
end


function decode(model, x, E)
    sumloss= 0.
    embeds = E
    decoder= model[:decoder]
    output = model[:output]
    tsteps = size(x, 1)
    bsize  = size(x, 2)
    hState = initstate(bsize, hidden) # resets the initial state to 0(or random)
    for t in 1:(tsteps - 1) # this is a decoder, start from 1 till n-1
        input= embeds[x[t, :],:]
        # mask = .!(x[t, :] .== vocsize) # vocsize length is the padding number in dictionary
        hState = gru(decoder, hState, input)
        hState = applyMask(hState, x[t, :])
        x_next = predictWords(output, hState)
        sumloss += sum(x_next .* getGolds(x[(t + 1), :])) # should debug here
    end
    return -sumloss
end


predictWords(output, hidden) = logsoftmax(hidden * output[:W] .+
    output[:b], dims=2)


function getGolds(indeces::Array)
    Is, Js, Vs = indeces, collect(1:length(indeces)), Array{Int64}(ones(length(indeces)))
    ygold = permutedims(atype{Float32, 2}(sparse(Is, Js, Vs, vocsize, length(indeces))))
    return ygold
end


gradients = gradloss(decode);

function train(model, data_trn, embeddings)
    examples = 0; totloss = 0;
    opts = optimizers(model, optim, lr=lrate)
    for x in data_trn
        x = reduce(hcat, x)
        grads, loss = gradients(model, x, embeddings)
        update!(model, grads, opts)
        totloss += loss
        examples += length(x)
    end
    totloss/examples
end

function validate_loss(model, data, embeddings)
    examples = 0.; totloss = 0.; euc_loss = 0.
    for x in data
        x = reduce(hcat, x)
        examples += length(x)
        totloss  += decode(model, x, embeddings)
    end
    return totloss / examples
end


@info "Reading Embedding File"
words, X =
    readEmbeds("/home/phd/Documents/Conference/FT_word_embeddings.txt");
@info "Reading WMT dataset"
lines = readlines("/home/phd/Documents/europarl-en.lower.txt");
line_lens = lines .|> split .|> length; # 1965734
linesGTWords = findall(x -> x > 10, line_lens); # returns indices of longer than 10
lines = lines[linesGTWords];
idx = pickSentences(length(lines), k=300_000);

corpus = lines[idx]
corp_words = lines[idx] .|> split;
corp_vocab = reduce(vcat, corp_words) .|> String |> unique;


ft_s2i = Dict(term => i for (i, term) in enumerate(words))
ft_s2i["<PAD>"] = get(ft_s2i, "<PAD>", 0) + length(ft_s2i) + 1 # adding PAD to the dictionary
ft_s2i["<UNK>"] = get(ft_s2i, "<UNK>", 0) + length(ft_s2i) + 1 # adding PAD to the dictionary
ft_i2s = Dict(i => term for (i, term) in enumerate(words));



word_idx = Int64[] # fast text indices of corpora words
for word in corp_vocab
    word in words ? push!(word_idx, ft_s2i[word]) : nothing
end
push!(word_idx, ft_s2i["<PAD>"])
push!(word_idx, ft_s2i["<UNK>"]);


s2i = Dict(j => i for (i, j) in enumerate(word_idx)); # ft to 1 index

corpus64 = crp2int(corpus, ft_s2i);


corpus64_2 = []
for (i, line) in enumerate(corpus64)
    line = [s2i[id] for id in line]
    push!(corpus64_2, line)
end


global optim  = Adam
global lrate  = 5e-3
global bsize= 128

global hidden = 768
global vocsize = length(word_idx)

@info "Creating Batches"
pad_tag_id = s2i[word_idx[end]] # padding id
@time batches = dynamicBatch!(corpus64_2, pad_tag_id, bsize=bsize);


rng2 = MersenneTwister(1234);
ids = shuffle(rng2, 1:length(batches));

trn = Int(floor(length(ids) * .95)) # take first n elements as train set
tst = Int(floor(length(ids) * .05))

train_loader = batches[ids[1:trn]]
valid_loader = batches[ids[1+trn:end]];


@info "Creating Model parameters"

model = Dict{Symbol, Any}();
model[:decoder] = initModel(hidden, 300)
model[:output]  = outputModel(hidden, vocsize);
E = KnetArray{Float32}(vcat(X[word_idx[1:end-2], :], rand(2, 300))); # last 2 words are for PAD and UNK;


global epochs = 5

function my_train!(model, train_loader, valid_loader)
    for epoch in 1:epochs
        trn_xloss = train(model, train_loader, E)
        val_xloss = validate_loss(model, valid_loader, E)

        @info "Epoch: $epoch"
        @printf "Train Loss: %.4f , Valid Loss: %.4f \t" trn_xloss val_xloss
        @printf "Train PP: %.4f, Valid PP: %.4f \n" exp(trn_xloss) exp(val_xloss)

        # if you want to save model use JLD2
        # JLD2.@save "$(save_path)/$(epoch).jld2" model"
    end
end


@info "Training Started ..."
my_train!(model, train_loader, valid_loader)
