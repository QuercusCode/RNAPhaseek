"""External novelty check: BLASTn the de novo designs against NCBI `nt` via the
remote BLAST URL API. Reports, per design, the best hit (accession, % identity,
length, E-value) or 'no significant similarity' — the standard 'never seen before' test.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python scripts/analysis/blast_novelty.py
"""
import sys, time, re, urllib.parse, urllib.request

BASE = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"


def load(f):
    out = []; h = None; s = ""
    for ln in open(f):
        ln = ln.rstrip()
        if ln.startswith(">"):
            if h: out.append((h, s))
            h = ln[1:].split()[0]; s = ""
        elif ln: s += ln
    if h: out.append((h, s))
    return out


def http(params, data=None):
    url = BASE + "?" + urllib.parse.urlencode(params) if not data else BASE
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers={"User-Agent": "RNAPhaseek-novelty/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8", "replace")


def main():
    designs = load("outputs/designs/designed_ga_v6.fasta") + load("outputs/designs/designed_den_v6.fasta")
    query = "\n".join(f">{h}\n{s}" for h, s in designs)
    print(f"submitting {len(designs)} designs to NCBI blastn vs nt ...", flush=True)

    put = http({}, data={"CMD": "Put", "PROGRAM": "blastn", "DATABASE": "nt",
                         "QUERY": query, "HITLIST_SIZE": "5"})
    rid = re.search(r"RID = (\S+)", put); rtoe = re.search(r"RTOE = (\d+)", put)
    if not rid:
        sys.exit("failed to get RID; NCBI may be busy. Response head:\n" + put[:500])
    rid = rid.group(1); est = int(rtoe.group(1)) if rtoe else 60
    print(f"RID={rid}  estimated {est}s; polling ...", flush=True)

    deadline = time.time() + 1800
    time.sleep(min(est, 60))
    while time.time() < deadline:
        st = http({"CMD": "Get", "FORMAT_OBJECT": "SearchInfo", "RID": rid})
        if "Status=READY" in st:
            print("READY — fetching results", flush=True); break
        if "Status=UNKNOWN" in st or "Status=FAILED" in st:
            sys.exit(f"BLAST job {rid} FAILED/UNKNOWN")
        print("  ...waiting", flush=True); time.sleep(45)
    else:
        sys.exit("timed out waiting for BLAST")

    txt = http({"CMD": "Get", "FORMAT_TYPE": "Text", "RID": rid})
    # NCBI Text report delimits queries with "Query= <id>"; each may carry weak local
    # hits. Parse each design's BEST (first/lowest-E) alignment and judge significance.
    print("\n================ NCBI nt BLAST RESULTS ================")
    print(f"{'design':<26}{'bestE':>8} {'%id':>5} {'alnLen':>7} {'cov%':>6}")
    blocks = re.split(r"\nQuery= ", txt)[1:]
    sig = 0
    for blk in blocks:
        qid = blk.split("\n", 1)[0].strip()
        if "Sequences producing significant alignments" not in blk or "ALIGNMENTS" not in blk:
            print(f"{qid:<26}{'--':>8} {'--':>5} {'0nt':>7} {'0%':>6}  no hits")
            continue
        aln = blk.split("ALIGNMENTS", 1)[1]
        ev = re.search(r"Expect = (\S+)", aln)
        idn = re.search(r"Identities = (\d+)/(\d+) \((\d+)%\)", aln)
        e = float(ev.group(1).rstrip(",")) if ev else float("inf")
        alnlen = int(idn.group(2)) if idn else 0
        pid = int(idn.group(3)) if idn else 0
        flag = "  *** SIGNIFICANT" if e < 0.01 else ""
        if e < 0.01:
            sig += 1
        print(f"{qid:<26}{e:>8.2g} {pid:>4}% {alnlen:>5}nt {100*alnlen/200:>5.0f}%{flag}")
    print(f"\nSummary: {sig}/{len(blocks)} designs have a SIGNIFICANT (E<0.01) nt hit.")
    print("  (0 significant + only short, scattered, E>1 local matches = novel vs nt)")
    print(f"Full report: https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Get&RID={rid}")


if __name__ == "__main__":
    main()
