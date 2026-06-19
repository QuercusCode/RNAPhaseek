"""
Characterize the diversity probe: re-score every de novo design config with the
full v3 pipeline, compare base composition / repeat content across lengths +
reward methods, and benchmark against the REAL training positives.
Produces a comparison table + Figure 12.

  python characterize_diversity.py
"""
import os, sys, glob, json, tempfile
import numpy as np, torch
import multimolecule  # noqa
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pathlib import Path
from collections import Counter
from Functions.RNAPhaseek.RNAPhaseek_hybrid        import RNAFMHybridClassifier
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data   import read_fasta, HybridRNADataset, make_collate_fn
from Functions.RNAPhaseek.RNAPhaseek_utils         import list_npz_sorted, setup_device, set_seed
from Functions.RNA_biophysical                     import RNABiophysicalExtractor
from Functions.precompute_fegs                     import process_fasta

FINAL = "model/strict_eval_v3aug/final_model.pt"
CONFIGS = [("seqprop_L80","outputs/designs/designed_div_seqprop_L80.fasta"),
           ("seqprop_L200","outputs/designs/designed_v3_seqprop.fasta"),
           ("seqprop_L400","outputs/designs/designed_div_seqprop_L400.fasta"),
           ("cond_L200","outputs/designs/designed_div_cond_L200.fasta"),
           ("struct_L200","outputs/designs/designed_div_struct_L200.fasta")]

def recover_norm():
    pos=read_fasta("Data/raw/multispecies/strict_pool_v3_positives.fasta")
    neg=read_fasta("Data/raw/multispecies/strict_pool_v3_negatives_all.fasta")
    y=np.concatenate([np.ones(len(pos)),np.zeros(len(neg))]).astype(int)
    bio=np.vstack([np.load("Data/splits/biophys_strict_v3_pos.npy"),np.load("Data/splits/biophys_strict_v3_neg.npy")]).astype(np.float32)
    dev,_=train_test_split(np.arange(len(y)),test_size=0.15,random_state=999,stratify=y)
    f_tr,_=train_test_split(dev,test_size=0.15,random_state=7,stratify=y[dev])
    btr=np.vstack([bio[f_tr],np.load("Data/splits/biophys_synth_train.npy").astype(np.float32)])
    return btr.mean(0),btr.std(0).clip(min=1e-8)

def comp(seqs):
    """mean A/C/G/U fraction (%) over a list of sequences."""
    tot=Counter()
    n=0
    for s in seqs:
        for c in s: tot[c]+=1; n+=1
    return {b:100*tot.get(b,0)/n for b in "ACGU"} if n else {b:0 for b in "ACGU"}

def main():
    set_seed(42); device=setup_device()
    args=HybridTrainArgs(bio_dim=33,use_species_embed=False,unfreeze_last_n=2,freeze_backbone=False)
    model=RNAFMHybridClassifier(args).to(device).eval()
    model.load_state_dict(torch.load(FINAL,map_location=device,weights_only=True))
    tok=AutoTokenizer.from_pretrained(args.backbone,trust_remote_code=True)
    m,sd=recover_norm(); ext=RNABiophysicalExtractor(normalize=False)

    def score(seqs):
        d=Path(tempfile.mkdtemp(prefix="fegs_div_"))
        tmpfa=d/"s.fasta"
        with open(tmpfa,"w") as f:
            for i,s in enumerate(seqs): f.write(f">s{i}\n{s}\n")
        process_fasta(tmpfa,d,topk=10,seq_len=1024,overwrite=True,workers=2)
        paths=list_npz_sorted(str(d))
        bio=np.stack([ext._compute_one(s) for s in seqs]).astype(np.float32)
        ds=HybridRNADataset(seqs,paths,np.zeros(len(seqs),int),(bio-m)/sd,args.max_nucleotides)
        ld=DataLoader(ds,batch_size=4,shuffle=False,collate_fn=make_collate_fn(tok,topk_m=10))
        out=[]
        with torch.no_grad():
            for tk,at,Lh,bi,_ in ld:
                tk=tk.to(device);at=at.to(device);Lh=Lh.to(device);bi=bi.to(device) if bi is not None else None
                lg,_=model(tk,at,labels=None,Lhat_stack=Lh,bio_features=bi)
                fin=torch.isfinite(lg).all(-1,keepdim=True);lg=torch.where(fin,lg,torch.zeros_like(lg))
                out.append(torch.softmax(lg,-1)[:,1].cpu().numpy())
        return np.concatenate(out), bio

    rows=[]
    print(f"\n{'config':<14} {'n':>3} {'fullP':>6} {'A%':>4} {'C%':>4} {'G%':>4} {'U%':>4} {'GC%':>4} {'tri':>4}")
    print("-"*64)
    for label,fa in CONFIGS:
        if not os.path.exists(fa): print(f"{label:<14} (missing {fa})"); continue
        seqs=[s for _,s in read_fasta(fa)]
        if not seqs: continue
        pr,bio=score(seqs); cm=comp(seqs)
        tri=float(np.mean(bio[:,31]))
        rows.append((label,len(seqs),pr.mean(),cm["A"],cm["C"],cm["G"],cm["U"],cm["G"]+cm["C"],tri))
        print(f"{label:<14} {len(seqs):>3} {pr.mean():>6.3f} {cm['A']:>4.0f} {cm['C']:>4.0f} {cm['G']:>4.0f} {cm['U']:>4.0f} {cm['G']+cm['C']:>4.0f} {tri:>4.1f}")

    # Reference: real training positives
    posseqs=[s for _,s in read_fasta("Data/raw/multispecies/strict_pool_v3_positives.fasta")]
    cmp=comp(posseqs)
    rows.append(("REAL positives",len(posseqs),float("nan"),cmp["A"],cmp["C"],cmp["G"],cmp["U"],cmp["G"]+cmp["C"],float("nan")))
    print(f"{'REAL positives':<14} {len(posseqs):>3} {'--':>6} {cmp['A']:>4.0f} {cmp['C']:>4.0f} {cmp['G']:>4.0f} {cmp['U']:>4.0f} {cmp['G']+cmp['C']:>4.0f}   --")

    # Figure 12: base composition per config vs real positives
    labels=[r[0] for r in rows]
    A=[r[3] for r in rows];C=[r[4] for r in rows];G=[r[5] for r in rows];U=[r[6] for r in rows]
    x=np.arange(len(labels))
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar(x,A,label="A",color="#e74c3c")
    ax.bar(x,C,bottom=A,label="C",color="#3498db")
    ax.bar(x,G,bottom=np.array(A)+np.array(C),label="G",color="#2ecc71")
    ax.bar(x,U,bottom=np.array(A)+np.array(C)+np.array(G),label="U",color="#f39c12")
    ax.axhline(25,color="grey",ls=":",lw=1); ax.text(len(labels)-0.5,26,"uniform 25%",fontsize=7,color="grey")
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,ha="right")
    ax.set_ylabel("Base composition (%)"); ax.set_ylim(0,100)
    ax.set_title("Figure 12 — De novo design composition across configs vs. real LLPS positives",fontweight="bold",fontsize=11)
    ax.legend(ncol=4,loc="upper center",frameon=False,bbox_to_anchor=(0.5,1.0))
    fig.savefig("report_assets/fig12_design_diversity.png",dpi=140,bbox_inches="tight"); plt.close()
    print("\nSaved -> report_assets/fig12_design_diversity.png")
    json.dump([{"config":r[0],"n":r[1],"fullP":r[2],"A":r[3],"C":r[4],"G":r[5],"U":r[6],"GC":r[7],"tri":r[8]} for r in rows],
              open("model/strict_eval_v3aug/diversity_summary.json","w"),indent=2)

if __name__=="__main__":
    main()
