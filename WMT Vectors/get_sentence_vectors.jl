using Random
using Printf
using Statistics
using LinearAlgebra
using MKL

using XLEs

using BSON: @save, @load
using PyCall

@pyimport torch
@pyimport fasttext as ft
@pyimport sentence_transformers.SentenceTransformer as st



Random.seed!(1234)

function loadsBert(sentences; model::String="/media/phd/PhD/DATA/all-mpnet-base-v2")
    @info model
    bert_model = st.SentenceTransformer(model, device="cuda:0")
    bert_embeddings = bert_model.encode(sentences, device="cuda:0", batch_size=1500)
    return bert_embeddings |> permutedims
end

function loadFT(sentences, model::String="/media/phd/PhD/DATA/FT/cc.en.300.bin")
    ft_embeddings = []
    ft_model = ft.load_model(model)
    for (i, sentence) in collect(enumerate(sentences))
        push!(ft_embeddings, ft_model.get_sentence_vector(sentence))
    end
    ft_embeddings = reduce(hcat, ft_embeddings)
    return ft_embeddings
end



lines = readlines("/home/phd/Documents/europarl-en.lower.txt");
# lines =

line_lens = lines .|> split .|> length; # 1965734
# if you watn to eliminate lines smaller than 10
# longer_lines = findall(x -> x > 10, line_lens)
# lines = lines[longer_lines];

# quick (for nearly 2M sentences takes 15-20 mins)
ftV = loadFT(lines);
writeBinaryEmbeddings("../models/WMT/lc-tok/FT_WMT_ALL", ftV, String[])

# leave your computer alone for while depending on your gpu
torch.cuda.empty_cache()
sBertV = loadsBert(lines);
writeBinaryEmbeddings("../models/WMT/lc-tok/sBERT_768_WMT_ALL", sBertV, String[])
