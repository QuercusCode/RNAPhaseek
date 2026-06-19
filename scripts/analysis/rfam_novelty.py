"""RNA-family novelty check: scan each de novo design against Rfam (curated RNA
families / covariance models) via the Rfam REST API. A design that matches NO
Rfam family does not recapitulate any known structured RNA family.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python scripts/analysis/rfam_novelty.py
"""
import json, time, urllib.parse, urllib.request, urllib.error

SUBMIT = "https://rfam.org/search/sequence"


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


def submit(seq):
    body = urllib.parse.urlencode({"seq": seq}).encode()
    req = urllib.request.Request(SUBMIT, data=body,
                                 headers={"Accept": "application/json", "Expect": "",
                                          "User-Agent": "RNAPhaseek-novelty/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["resultURL"]


def poll(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "RNAPhaseek-novelty/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status == 202:
                return None
            data = json.loads(r.read().decode())
            return data if ("hits" in data or "closed" in data) else None
    except urllib.error.HTTPError as e:
        if e.code in (202, 502, 503):
            return None
        raise


def main():
    designs = load("outputs/designs/designed_ga_v6.fasta") + load("outputs/designs/designed_den_v6.fasta")
    print(f"submitting {len(designs)} designs to Rfam ...", flush=True)
    jobs = []
    for h, s in designs:
        try:
            jobs.append((h, submit(s)))
        except Exception as e:
            jobs.append((h, None)); print(f"  submit failed {h}: {e}", flush=True)
        time.sleep(1.0)
    print("polling ...", flush=True)

    results = {}
    deadline = time.time() + 900
    pending = [j for j in jobs if j[1]]
    while pending and time.time() < deadline:
        time.sleep(15)
        still = []
        for h, url in pending:
            r = poll(url)
            if r is None:
                still.append((h, url))
            else:
                results[h] = r
        pending = still
        print(f"  {len(results)}/{len(jobs)} done", flush=True)

    print("\n================ Rfam RNA-FAMILY RESULTS ================")
    matched = 0
    for h, _ in jobs:
        r = results.get(h)
        if r is None:
            print(f"  {h:<26} (no result / timed out)"); continue
        hits = r.get("hits") or {}
        # hits is a dict keyed by family accession; keep significant ones
        fams = [(acc, hl) for acc, hl in hits.items() if hl]
        if not fams:
            print(f"  {h:<26} NO known RNA family  (novel)")
        else:
            matched += 1
            desc = ", ".join(f"{acc}" for acc, _ in fams[:3])
            print(f"  {h:<26} MATCH: {desc}")
    done = sum(1 for h, _ in jobs if h in results)
    print(f"\nSummary: {done - matched}/{done} resolved designs match NO known Rfam family.")


if __name__ == "__main__":
    main()
