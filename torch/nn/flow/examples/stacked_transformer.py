import torch
import torch.nn.functional as F
from torch.nn.flow.chain import Chain
from torch.nn.flow.examples.utils.mnist_trainer import train_and_evaluate

# ========================= micro blocks (readable building bricks) =========================

def ToHeads(tc: Chain, *, d_model: int, n_heads: int):
    Dh = d_model // n_heads
    return (
        tc
        .tensor.unflatten(dim=2, sizes=(n_heads, Dh))   # [B,T,D] -> [B,T,H,Dh]
        .tensor.transpose(1, 2)                         # [B,H,T,Dh]
    )

# ========================= embeddings / positions =========================

def TokenEmbedWithPos(tc: Chain, *, vocab_size: int, d_model: int, max_seq: int, pos_dropout: float = 0.0):
    # idx [B,T] -> tok [B,T,D] ; then add learned positions
    return (
        tc
        .nn.Embedding(vocab_size, d_model)
        .fork_tensor(
            tok     = lambda x: x,
            pos_idx = lambda x: torch.arange(x.shape[1], device=x.device, dtype=torch.long)[None, :].expand(x.shape[0], -1),
        )
        .fork(
            tok = lambda b: b.select("tok"),
            pos = lambda b: b.select("pos_idx").nn.Embedding(max_seq, d_model).nn.Dropout(p=pos_dropout),
        )
        .fuse(lambda r: r["tok"] + r["pos"])
    )

def AddLearnedPos(tc: Chain, *, d_model: int, max_seq: int, pos_dropout: float = 0.0):
    # input stream already [B,T,D]
    return (
        tc
        .fork_tensor(
            tok     = lambda x: x,
            pos_idx = lambda x: torch.arange(x.shape[1], device=x.device, dtype=torch.long)[None, :].expand(x.shape[0], -1),
        )
        .fork(
            tok = lambda b: b.select("tok"),
            pos = lambda b: b.select("pos_idx").nn.Embedding(max_seq, d_model).nn.Dropout(p=pos_dropout),
        )
        .fuse(lambda r: r["tok"] + r["pos"])
    )

# ========================= transformer blocks =========================

def MHSA(tc: Chain, *, d_model: int, n_heads: int, dropout_p: float = 0.0):
    # [B,T,D] → proj → split → heads → SDPA → merge → out proj
    return (
        tc
        .nn.Linear(d_model, 3 * d_model, bias=False)                 # [B,T,3D]
        .fork_tensor(                                                # split q/k/v
            q=lambda x, D=d_model: x[..., :D],
            k=lambda x, D=d_model: x[..., D:2*D],
            v=lambda x, D=d_model: x[..., 2*D:],
        )
        .fork(                                                        # to heads
            q=lambda b: b.select("q").pipe(ToHeads, d_model=d_model, n_heads=n_heads),
            k=lambda b: b.select("k").pipe(ToHeads, d_model=d_model, n_heads=n_heads),
            v=lambda b: b.select("v").pipe(ToHeads, d_model=d_model, n_heads=n_heads),
        )
        .call_module(
            F.scaled_dot_product_attention,
            'q',
            'k',
            'v',
            is_causal=True,
            dropout_p=dropout_p
        )
        .tensor.transpose(1, 2)                         # [B,T,H,Dh]
        .tensor.contiguous()
        .tensor.flatten(start_dim=2)                                                # [B,T,D]
        .nn.Linear(d_model, d_model, bias=False)                     # out proj
    )

def MLP(tc: Chain, *, d_model: int, mlp_mult: int = 4, dropout_p: float = 0.0):
    return (
        tc
        .nn.Linear(d_model, mlp_mult * d_model, bias=False)
        .functional.gelu()
        .nn.Dropout(p=dropout_p)
        .nn.Linear(mlp_mult * d_model, d_model, bias=False)
    )

def TransformerBlock(tc: Chain, *, d_model: int, n_heads: int, mlp_mult: int = 4, attn_dropout: float = 0.0, mlp_dropout: float = 0.0):
    return (
        tc
        .residual(lambda b: (
            b.nn.LayerNorm(d_model)
             .pipe(MHSA, d_model=d_model, n_heads=n_heads, dropout_p=attn_dropout)
        ))
        .residual(lambda b: (
            b.nn.LayerNorm(d_model)
             .pipe(MLP, d_model=d_model, mlp_mult=mlp_mult, dropout_p=mlp_dropout)
        ))
    )

# ========================= image frontend and head =========================

def ImagePatchEmbed(tc: Chain, *, d_model: int):
    # For patch_size=1, treat each pixel as a token: [B,1,28,28] -> [B,784,D]
    return (
        tc
        .tensor.flatten(start_dim=1)    # [B, 784]
        .tensor.unsqueeze(-1)           # [B, 784, 1]
        .nn.Linear(1, d_model)          # [B, 784, D]
    )

def ClassifierHead(tc: Chain, *, d_model: int, n_classes: int = 10):
    return (
        tc
        .nn.LayerNorm(d_model)
        .tensor.mean(dim=1)             # pool tokens -> [B,D]
        .nn.Linear(d_model, n_classes)  # logits
    )

# ========================= full model =========================

def mnist_transformer(
    *,
    d_model: int = 128,
    n_layers: int = 2,
    n_heads: int = 4,
    mlp_mult: int = 2,
    attn_dropout: float = 0.1,
    mlp_dropout: float = 0.1,
    pos_dropout: float = 0.0,
):
    T = 28 * 28
    return (
        Chain()
        .pipe(ImagePatchEmbed, d_model=d_model)     # [B,784,D]
        .pipe(AddLearnedPos, d_model=d_model, max_seq=T, pos_dropout=pos_dropout)
        .repeat(n_layers, TransformerBlock, d_model=d_model, n_heads=n_heads,
                mlp_mult=mlp_mult, attn_dropout=attn_dropout, mlp_dropout=mlp_dropout)
        .pipe(ClassifierHead, d_model=d_model, n_classes=10)
        .freeze()
    )

# ========================= main (uses your MNIST trainer) =========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model = mnist_transformer(
        d_model=128,
        n_layers=2,
        n_heads=4,
        mlp_mult=2,
        attn_dropout=0.0,
        mlp_dropout=0.1,
        pos_dropout=0.0,
    ).to(device)

    model = torch.compile(model)  # optional

    # the trainer you provided earlier expects (model, epochs, lr, batch_size, device)
    train_and_evaluate(model, epochs=5, lr=1e-3, batch_size=128)

if __name__ == "__main__":
    main()
