import torch
import torch.nn as nn
import torch.nn.functional as F

class _Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x)

class _TensorProxy:
    def __init__(self, chain): self._c = chain
    def __getattr__(self, name):
        def wrapper(*a, **kw):
            return self._c.apply(lambda x, n=name, a=a, kw=kw: getattr(x, n)(*a, **kw))
        return wrapper

class _NNProxy:
    def __init__(self, chain): self._c = chain
    def __getattr__(self, name):
        cls = getattr(nn, name)  # e.g., nn.Linear
        def ctor(*a, **kw):
            mod = cls(*a, **kw)
            return self._c._add(mod)
        return ctor

class _FunctionalProxy:
    def __init__(self, chain, namespace):
        self._c = chain
        self._ns = namespace  # e.g., torch.nn.functional
    def __getattr__(self, name):
        fn = getattr(self._ns, name)  # e.g., F.gelu
        def call(*a, **kw):
            return self._c.apply(lambda x, f=fn, a=a, kw=kw: f(x, *a, **kw))
        return call


class _FieldsModule(nn.Module):
    """
    Wrap any callable(*tensors, **kw) as an nn.Module that:
      - pulls named fields out of a dict stream,
      - calls fn(*fields, **kw),
      - returns the result (or writes into dict when out is set).
    """
    def __init__(self, fn, names, *, out=None, **kwargs):
        super().__init__()
        self.fn = fn
        self.names = tuple(names)
        self.out = out
        self.kw = dict(kwargs)

    def forward(self, r):
        # Expect a dict stream with the fields
        args = [r[n] for n in self.names]
        y = self.fn(*args, **self.kw)
        if self.out is None:
            return y
        rr = dict(r)
        rr[self.out] = y
        return rr

class Chain(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self._frozen = False
        self.tensor = _TensorProxy(self)
        self.nn = _NNProxy(self)                  # x.nn.Linear(...).nn.ReLU()...
        self.functional = _FunctionalProxy(self, F)  # x.functional.gelu(), x.functional.relu()


    # ---- core composition ----
    def _add(self, mod: nn.Module):
        if self._frozen:
            raise RuntimeError("Chain is frozen")
        self.layers.append(mod)
        return self

    def apply(self, fn):
        return self._add(_Lambda(fn))

    def pipe(self, fn, *a, **kw):
        out = fn(self, *a, **kw)
        return out if isinstance(out, Chain) else self

    def repeat(self, n, fn, *a, **kw):
        for _ in range(n):
            self.pipe(fn, *a, **kw)
        return self

    def residual(self, fn, *a, **kw):
        block = Chain()
        fn(block, *a, **kw)
        block.freeze()
        return self.apply(lambda x, b=block: x + b(x))

    def fork(self, **branches):
        names = tuple(branches.keys())
        built = {}
        for n in names:
            b = Chain()
            branches[n](b)        # the user-supplied lambda uses chaining on 'b'
            b.freeze()
            built[n] = b
        return self._add(_Lambda(lambda x, names=names, built=built: {n: built[n](x) for n in names}))

    def call_module(self, fn, *names, out=None, **kwargs):
        """
        Register _FieldsModule(fn, names, out=..., **kwargs) as an nn.Module step.
        No lambdas; stays fully module-based.
        """
        return self._add(_FieldsModule(fn, names, out=out, **kwargs))

    # ---- add to Chain ----
    def fork_tensor(self, **branches):
        """
        Split the current *Tensor* stream into a dict of named tensors by applying
        pure Tensor->Tensor functions (no Chain plumbing) per branch.
        """
        fns = {k: v for k, v in branches.items()}
        return self.apply(lambda x, fns=fns: {name: fn(x) for name, fn in fns.items()})

    def select(self, name: str):
        """Collapse a dict stream to a single field tensor."""
        return self.apply(lambda r, n=name: r[n] if isinstance(r, dict) else r)

    def fuse(self, fn, *, out=None):
        def _fuse(r, f=fn, out=out):
            y = f(r)
            if out is None:
                return y
            rr = dict(r) if isinstance(r, dict) else {"x": r}
            rr[out] = y
            return rr
        return self._add(_Lambda(_fuse))

     # convenience when you only want one field back
    def pick(self, name: str):
        return self._add(_Lambda(lambda r, n=name: r[n] if isinstance(r, dict) else r))

    # concat/stack selected fields (store as new field if 'out' given, else replace stream)
    def concat(self, names, dim=-1, out=None):
        def _cat(r, ns=tuple(names), d=dim, out=out):
            y = torch.cat([r[n] for n in ns], dim=d)
            if out is None: return y
            rr = dict(r); rr[out] = y; return rr
        return self._add(_Lambda(_cat))

    def stack(self, names, dim=0, out=None):
        def _stk(r, ns=tuple(names), d=dim, out=out):
            y = torch.stack([r[n] for n in ns], dim=d)
            if out is None: return y
            rr = dict(r); rr[out] = y; return rr
        return self._add(_Lambda(_stk))

    def freeze(self):
        self._frozen = True
        return self

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x

