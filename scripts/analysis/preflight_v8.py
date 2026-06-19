"""Pre-flight for run_v8_cv: recompute the 7 extended features (task 22) and validate the
45-dim plumbing (pool -> bio 45 -> model bio_dim=45 -> one forward). Real file so mp spawn works."""
import os, sys
sys.path.insert(0, os.getcwd())


def main():
    import paths, numpy as np, torch, multimolecule  # noqa
    import Functions.RNAPhaseek.RNAPhaseek_hybrid_eval as E
    from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
    from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import HybridRNADataset, make_collate_fn
    from Functions.RNAPhaseek.RNAPhaseek_utils import setup_device
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from run_v5_final import build_pool_v5
    from run_v8_cv import ext7_parallel

    print("building pool + recomputing ext7 features (task 22)...", flush=True)
    build_pool_v5()
    new7 = ext7_parallel(E.G["seqs"], "Data/splits/biophys_v5_ext7.npy")
    E.G["bio"] = np.hstack([E.G["bio"], new7]).astype(np.float32)
    print("pool bio shape:", E.G["bio"].shape, "(want N x 45)")
    print("new7 mean:", np.round(new7.mean(0), 3), "| any NaN:", bool(np.isnan(new7).any()))

    dev = setup_device()
    args = HybridTrainArgs(bio_dim=45, use_species_embed=False, unfreeze_last_n=2, freeze_backbone=False)
    tok = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    model = E.init_model(args, dev)
    ds = HybridRNADataset(E.G["seqs"][:4], E.G["paths"][:4], E.G["y"][:4], E.G["bio"][:4], args.max_nucleotides)
    ld = DataLoader(ds, batch_size=4, collate_fn=make_collate_fn(tok, topk_m=args.topk_m))
    for tk, at, Lh, bi, yb in ld:
        out = model(tk.to(dev), at.to(dev), labels=yb.to(dev), Lhat_stack=Lh.to(dev), bio_features=bi.to(dev))
        print("forward OK -> logits", tuple(out[0].shape), "| loss", float(out[1]))
        break
    print("PREFLIGHT PASSED: 45-dim plumbing works end-to-end")


if __name__ == "__main__":
    main()
