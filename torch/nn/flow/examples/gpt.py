from torch.nn.flow.chain import Chain
import torch

# ---- token + learned positional embeddings and sum ----
def EmbedWithPos(tc: Chain, *, vocab_size: int, d_model: int, max_seq: int):
    return (
        tc
        .embedding(vocab_size, d_model)
        .bundle(
            tok=lambda b: b.apply(lambda x: x),
            pos=lambda b: (
                b
                .apply(lambda x: (x.shape[0], x.shape[1]))
                .apply(lambda bt: (bt[0], torch.arange(bt[1])))
                .apply(lambda bt: bt[1][None, :].expand(bt[0], -1))
                .apply(lambda pos: pos.to(dtype=torch.long))
                .embedding(max_seq, d_model)
            ),
        )
        .join(lambda tok, pe: tok + pe)
    )

# ---- split project → q,k,v; heads reshape; SDPA; merge; out proj ----
def split_qkv(tc: Chain, d_model: int):
    return (
        tc
        .apply(lambda x, D=d_model: x.split(D, dim=-1))
        .apply(lambda parts: (parts[0], parts[1], parts[2]))
    )

def to_heads(tc: Chain, d_model: int, n_heads: int):
    return (
        tc
        .bundle(
            q=lambda b: (
                b
                .apply(lambda t: t[0])
                .apply(lambda q, D=d_model, H=n_heads: q.view(q.shape[0], q.shape[1], H, D // H))
                .apply(lambda q: q.transpose(1, 2))
            ),
            k=lambda b: (
                b
                .apply(lambda t: t[1])
                .apply(lambda k, D=d_model, H=n_heads: k.view(k.shape[0], k.shape[1], H, D // H))
                .apply(lambda k: k.transpose(1, 2))
            ),
            v=lambda b: (
                b
                .apply(lambda t: t[2])
                .apply(lambda v, D=d_model, H=n_heads: v.view(v.shape[0], v.shape[1], H, D // H))
                .apply(lambda v: v.transpose(1, 2))
            ),
        )
        .join(lambda q, k, v: (q, k, v))
    )

def from_heads(tc: Chain, d_model: int):
    return (
        tc
        .apply(lambda y: y.transpose(1, 2))
        .apply(lambda y, D=d_model: y.contiguous().view(y.shape[0], y.shape[1], D))
    )

def MHSA(tc: Chain, *, d_model: int, n_heads: int, dropout_p: float = 0.0):
    return (
        tc
        .linear(d_model, 3 * d_model, bias=False)
        .pipe(split_qkv, d_model=d_model)
        .pipe(to_heads, d_model=d_model, n_heads=n_heads)
        .apply(lambda t: (t[0], t[1], t[2]))
        .apply(lambda t, p=dropout_p: F.scaled_dot_product_attention(
            t[0],
            t[1],
            t[2],
            is_causal=True,
            dropout_p=p,
        ))
        .pipe(from_heads, d_model=d_model)
        .linear(d_model, d_model, bias=False)
    )

# ---- simple GELU MLP: d -> 4d -> GELU -> d ----
def MLP(tc: Chain, *, d_model: int, mlp_mult: int = 4):
    return (
        tc
        .linear(d_model, mlp_mult * d_model, bias=False)
        .gelu()
        .linear(mlp_mult * d_model, d_model, bias=False)
    )

# ---- one Transformer block (pre-LN GPT style) ----
def TransformerBlock(tc: Chain, *, d_model: int, n_heads: int, mlp_mult: int = 4, dropout_p: float = 0.0):
    return (
        tc
        .residual(lambda b: (
            b
            .layernorm(d_model)
            .pipe(MHSA, d_model=d_model, n_heads=n_heads, dropout_p=dropout_p)
        ))
        .residual(lambda b: (
            b
            .layernorm(d_model)
            .pipe(MLP, d_model=d_model, mlp_mult=mlp_mult)
        ))
    )

# ---- final head ----
def FinalHead(tc: Chain, *, d_model: int, vocab_size: int):
    return (
        tc
        .layernorm(d_model)
        .linear(d_model, vocab_size, bias=False)
    )

# ========================= Build minimal GPT (model IS a Chain) =========================

def gpt_chain(
    *,
    vocab_size: int,
    d_model: int = 512,
    n_layers: int = 6,
    n_heads: int = 8,
    mlp_mult: int = 4,
    max_seq: int = 2048,
    dropout_p: float = 0.0,
):
    return (
        Chain()
        .pipe(EmbedWithPos, vocab_size=vocab_size, d_model=d_model, max_seq=max_seq)
        .repeat(n_layers, TransformerBlock, d_model=d_model, n_heads=n_heads, mlp_mult=mlp_mult, dropout_p=dropout_p)
        .pipe(FinalHead, d_model=d_model, vocab_size=vocab_size)
        .freeze()
    )

# ========================= quick smoke test =========================
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    V = 32000
    D = 384
    H = 6
    L = 4
    MAX_SEQ = 256

    model = gpt_chain(
        vocab_size=V,
        d_model=D,
        n_layers=L,
        n_heads=H,
        mlp_mult=4,
        max_seq=MAX_SEQ,
        dropout_p=0.0,
    )

    compiled = torch.compile(model)

    B = 2
    T = 128
    x = torch.randint(0, V, (B, T))
    logits = compiled(x)
    print("logits:", logits.shape)
