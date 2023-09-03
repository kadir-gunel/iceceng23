
using Random
Random.seed!(1234);

using Base.Iterators
using LinearAlgebra
using Statistics
using Random
using Printf


using Optimisers
using Flux
using Flux.Losses: mse
using Flux: params, onehotbatch
using Flux.Data: DataLoader

using BSON: @save, @load
using NPZ
using ONNXNaiveNASflux



function mapOrthogonalSentences(X::T, Y::T) where {T}
    F = svd(X * Y')
    W = permutedims(F.U * F.Vt) # * cuinv((X * X') + λ .* CuMatrix{Float32}(I, 300, 300)))
    return W
end

pickSentences(tot_lines::Int64; k::Int64=Int(64e3)) = (collect(1:tot_lines) |> shuffle)[1:k]


@info "Reading Sentence Embeddings of FT and sBERT"
_, X = readBinaryEmbeddings("../../models/WMT/lc-tok/FT_WMT_ALL")
_, Y = readBinaryEmbeddings("../../models/WMT/lc-tok/sBERT_768_WMT_ALL");


X, Y = map(permutedims, [X, Y]);

lines = readlines("/home/phd/Documents/europarl-en.lower.txt");
line_lens = lines .|> split .|> length; # 1965734
linesGTWords = findall(x -> x > 10, line_lens); # returns indices of longer than 10
lines = lines[linesGTWords];
idx = pickSentences(length(lines), k=300_000);


trn = Int(floor(length(idx) * .95)) # take first n elements as train set
tst = Int(floor(length(idx) * .05))

Xtrn = X[:, idx[1:trn]]
Ytrn = Y[:, idx[1:trn]]

Xval = X[:, idx[trn+1:end]]
Yval = Y[:, idx[trn+1:end]];

@info "Calculating single rotation matrix: W"
W = mapOrthogonalSentences(Xtrn, Ytrn)


@info "Iteratively calculating the rotation matrix : W_Global "
G = 100
iter = div(trn, G)

W_Global = zeros(size(Ytrn, 1), size(Xtrn, 1))
for i in 1:G:(size(Xtrn, 2))
    rng = i:i+G-1
    Wi, _ = mapOrthogonalSentences(Xtrn[:, rng], Ytrn[:, rng])
    W_Global = W_Global + Wi
end



@info "Fine-tuning W_Global with Neural Network"

# loss functions for NN: euclidean, cosine and mse
p_norm(M::T; dim=2) where {T} = sqrt.(sum(real(M .* conj(M)), dims=dim))
cosine(X::T, Y::T) where {T} = diag((X ./ p_norm(X)) * (Y ./
    p_norm(Y))')

euc_loss(model)= (x, y) -> sqrt(mse(model(x), y, agg=sum))
mse_loss(model)= (x, y) -> (mse(model(x), y, agg=mean))
cosine_loss(model) = (x, y) -> 1 .- mean(abs.(cosine(model(x) |>
    permutedims, y |> permutedims)))

@info "Creating Batches"
train_loader = DataLoader((Xtrn, Float32.(W' * Ytrn) |> unit), batchsize=50000, shuffle=true)
valid_loader = DataLoader((Xval, Float32.(W' * Yval) |> unit), batchsize=15000, shuffle=true);


@info "Model Creation"
atype = gpu;
model = Chain(Dense(300, 300, elu)) |> atype;

λ = 5e-3
rule = Optimisers.OptimiserChain(Optimisers.ADAM(λ),
                                Optimisers.WeightDecay(1f-8),
                                Optimisers.ClipGrad(1e2));
opt_state = Optimisers.setup(rule, model);


for epoch in 1:epochs
    names = [:euc, :cos, :mse]
    trn_losses = Dict(name => Float32[] for name in names);
    tst_losses = Dict(name => Float32[] for name in names);

    Flux.trainmode!(model);
    for (x, y) in train_loader

        loss, ∇model = Flux.withgradient(model, x, y) do m, x, y
            euc_loss(m)(x |> atype, y |> atype)
        end

        opt_state, model = Optimisers.update!(opt_state, model, ∇model[1])
        # Flux.train!(euc_loss(model), Flux.params(model), [(x|> atype, y |> atype)], opt_state)
        push!(trn_losses[:euc], euc_loss(model)(x|> atype, y |> atype))
        # push!(trn_losses[:cos], cosine_loss(model)(x|> atype, y |> atype))
        # push!(trn_losses[:mse], mse_loss(model)(x|> atype, y |> atype))
    end

    Flux.testmode!(model)
    for (x, y) in valid_loader
        push!(tst_losses[:euc], euc_loss(model)(x|> atype, y |> atype))
        push!(tst_losses[:cos], cosine_loss(model)(x|> atype, y |> atype))
        push!(tst_losses[:mse], mse_loss(model)(x|> atype, y |> atype))
    end

    @info "Epoch $epoch"
    @printf "Train Loss Euclidean: %.4f \n" mean(trn_losses[:euc])
    @printf "Validation Loss Euclidean: %.4f \t Cosine: %.4f \t
    MSE:%4f \n" mean(tst_losses[:euc]) mean(tst_losses[:cos]) mean(tst_losses[:mse])
end
