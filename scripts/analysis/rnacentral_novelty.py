"""RNA-specific novelty check: search the de novo design panel against RNAcentral
(all known RNA, ~99 member databases) via its EBI sequence-search REST API. A design
with no significant RNAcentral hit does not match any known RNA sequence.

  python scripts/analysis/rnacentral_novelty.py
"""
import json, time, urllib.parse, urllib.request, urllib.error

SUB = "https://sequence-search.rnacentral.org/api/submit-job"
STATUS = "https://sequence-search.rnacentral.org/api/job-status/"
RESULTS = "https://sequence-search.rnacentral.org/api/job-results/"
# wet-lab candidate panel (high-confidence, structure-validated picks)
PANEL = ["ga_v6_design_0", "den_design_130", "den_design_27", "den_design_2", "den_design_62"]


def load(f):
    out = {}; h = None; s = ""
    for ln in open(f):
        ln = ln.rstrip()
        if ln.startswith(">"):
            if h: out[h] = s
            h = ln[1:].split("_P")[0]; s = ""
        elif ln: s += ln
    if h: out[h] = s
    return out


def post(seq):
    body = json.dumps({"sequence": seq, "databases": []}).encode()
    req = urllib.request.Request(SUB, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["job_id"]


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (202, 502, 503):
            return None
        raise


def main():
    seqs = {}
    seqs.update(load("outputs/designs/designed_ga_v6.fasta"))
    seqs.update(load("outputs/designs/designed_den_v6.fasta"))
    panel = [(name, seqs[name]) for name in PANEL if name in seqs]
    print(f"submitting {len(panel)} panel designs to RNAcentral ...", flush=True)
    jobs = []
    for name, s in panel:
        jid = post(s); jobs.append((name, jid))
        print(f"  {name}: {jid}", flush=True)
        time.sleep(7)  # rate limit: 10 submits/min

    print("polling (each searches ~99 databases; can be slow) ...", flush=True)
    results = {}
    deadline = time.time() + 2400
    pending = list(jobs)
    while pending and time.time() < deadline:
        time.sleep(25)
        still = []
        for name, jid in pending:
            st = get(STATUS + jid)
            if st is None:
                still.append((name, jid)); continue
            s = st.get("status", "")
            if s in ("finished", "success", "done"):
                results[name] = jid
            elif s in ("error", "not_found", "failed"):
                results[name] = None
                print(f"  {name}: job {s}", flush=True)
            else:
                still.append((name, jid))
        pending = still
        print(f"  {len(results)}/{len(jobs)} done"
              + (f"; waiting on {[n for n, _ in pending]}" if pending else ""), flush=True)

    print("\n================ RNAcentral RESULTS ================")
    for name, jid in jobs:
        rj = results.get(name)
        if rj is None:
            print(f"  {name:<18} (no result / timed out / error)"); continue
        data = get(RESULTS + rj + "?page=1&page_size=5")
        hits = (data or {}).get("results") or (data or {}).get("hits") or []
        if not hits:
            print(f"  {name:<18} NO RNAcentral hit  (novel)")
            continue
        h = hits[0]
        rid = h.get("rnacentral_id") or h.get("id") or "?"
        idn = h.get("identity") or h.get("percent_identity") or h.get("alignment_identity")
        ev = h.get("e_value", h.get("evalue", "?"))
        qc = h.get("query_coverage", h.get("query_cov", "?"))
        desc = (h.get("description") or "")[:40]
        print(f"  {name:<18} {len(hits)} hit(s); best id={idn} E={ev} qcov={qc} [{rid} {desc}]")
    print("\n(keys seen on a sample hit, for reference):")
    for name, jid in jobs:
        rj = results.get(name)
        if rj:
            d = get(RESULTS + rj + "?page=1&page_size=1")
            hs = (d or {}).get("results") or (d or {}).get("hits") or []
            print("  top-level keys:", list((d or {}).keys()))
            if hs:
                print("  hit keys:", list(hs[0].keys()))
            break


if __name__ == "__main__":
    main()
