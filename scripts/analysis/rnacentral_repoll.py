"""Re-poll the saved RNAcentral novelty jobs (no resubmit). Jobs persist ~7 days.
  python scripts/analysis/rnacentral_repoll.py
"""
import json, urllib.request, urllib.error

ST = "https://sequence-search.rnacentral.org/api/job-status/"
RES = "https://sequence-search.rnacentral.org/api/job-results/"


def get(url, t=30):
    try:
        with urllib.request.urlopen(url, timeout=t) as r:
            return json.loads(r.read().decode())
    except (urllib.error.HTTPError, Exception):
        return None


def main():
    jobs = json.load(open("outputs/rnacentral_jobs.json"))["jobs"]
    for name, jid in jobs.items():
        st = get(ST + jid, 20) or {}
        status = st.get("status", "no-response")
        if status not in ("finished", "success", "done"):
            print(f"  {name:<16} {status} (progress {st.get('progress')}, "
                  f"{st.get('databases_finished')}/{st.get('databases_total')} DBs)")
            continue
        d = get(RES + jid + "?page=1&page_size=5", 40) or {}
        hits = d.get("results") or d.get("hits") or []
        if not hits:
            print(f"  {name:<16} DONE — NO RNAcentral hit (novel)")
        else:
            h = hits[0]
            print(f"  {name:<16} DONE — {len(hits)} hit(s); top="
                  f"{h.get('rnacentral_id', h.get('id', '?'))} "
                  f"id={h.get('identity', '?')} E={h.get('e_value', h.get('evalue', '?'))}")


if __name__ == "__main__":
    main()
