"""ERNIE-RNA <-> transformers compatibility shim.

This multimolecule build's ERNIE-RNA modeling code calls transformers' masking utils with the
keyword `input_embeds=` (a typo), but transformers 5.x expects `inputs_embeds=`. RNA-FM uses the
correct spelling, which is why only ERNIE-RNA fails. Importing this module idempotently wraps the
two mask builders inside the ernierna module namespace to accept the misspelled kwarg.

Also exposes load_ernierna() which handles the second quirk: ErnieRnaModel's __init__ vocab check
(default tokenizer has 26 tokens vs config's 28), resolved by passing ERNIE-RNA's own tokenizer.

  from Functions.RNAPhaseek.ernierna_compat import patch_ernierna, load_ernierna
"""
_PATCHED = False


def patch_ernierna():
    """Rename input_embeds -> inputs_embeds for create_bidirectional_mask / create_causal_mask
    as referenced inside the ernierna modeling module. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    import multimolecule.models.ernierna.modeling_ernierna as _ern
    from transformers.masking_utils import create_bidirectional_mask as _cbm, create_causal_mask as _ccm

    def _fix(fn):
        def wrapped(*a, **k):
            if "input_embeds" in k:
                k["inputs_embeds"] = k.pop("input_embeds")
            return fn(*a, **k)
        return wrapped

    _ern.create_bidirectional_mask = _fix(_cbm)
    _ern.create_causal_mask = _fix(_ccm)

    # MPS fix: ERNIE's get_pairwise_bias computes bias[b,i,j] = pairwise_bias_map[ids[b,i], ids[b,j]]
    # via .expand() + 2-D advanced indexing. MPS mishandles both the stride-0 expand AND broadcast
    # index arithmetic, returning out-of-bounds garbage indices nondeterministically (crashes like
    # "index 78 out of bounds for size 28" / "index 2884 for size 784"). Reformulate with NO advanced
    # indexing at all: one-hot + matmul. With O = onehot(ids) (B,L,V),  O @ M @ Oᵀ  equals
    # M[ids_i, ids_j] exactly (V=28, cheap), and matmul is fully MPS-reliable.
    import torch.nn.functional as _Fnn
    from multimolecule import ErnieRnaModel

    def _safe_get_pairwise_bias(self, input_ids, attention_mask=None):
        if hasattr(input_ids, "tensor"):
            input_ids = input_ids.tensor
        bias_map = self.pairwise_bias_map
        if bias_map.device != input_ids.device:
            bias_map = bias_map.to(input_ids.device)
            self.pairwise_bias_map = bias_map
        V = bias_map.shape[0]
        oh = _Fnn.one_hot(input_ids.long(), V).to(bias_map.dtype)   # (B, L, V)
        return (oh @ bias_map) @ oh.transpose(1, 2)                 # (B, L, L) == M[ids_i, ids_j]

    ErnieRnaModel.get_pairwise_bias = _safe_get_pairwise_bias
    _PATCHED = True


def load_ernierna(model_id="multimolecule/ernierna", attn_implementation="eager"):
    """Load an ErnieRnaModel with the compat patch applied and its own (28-token) tokenizer,
    sidestepping the vocab-size check. attn_implementation='eager' is the reference path."""
    patch_ernierna()
    from multimolecule import RnaTokenizer, ErnieRnaModel
    tok = RnaTokenizer.from_pretrained(model_id)
    model = ErnieRnaModel.from_pretrained(model_id, tokenizer=tok, attn_implementation=attn_implementation)
    return model
