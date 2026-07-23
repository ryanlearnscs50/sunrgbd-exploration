#!/usr/bin/env python3
"""Collect mAP results from all eval_*.log files and compare to the released
manifests. Writes a human-readable report to stdout and a machine-readable
results.json next to it."""
import json
import re
import sys
from pathlib import Path

ROOT = Path("/data3/ryan/ldmr_exploration")
LOGS = ROOT / "logs"
CKPTS = ROOT / "checkpoints"

PROTOCOLS = ["3stage", "5stage", "10stage"]


def parse_log(path):
    """Return dict with mAP_0.25, mAP_0.50, per-class AP@0.25, or None."""
    try:
        text = path.read_text(errors="replace").replace("\r", "\n")
    except OSError:
        return None
    # The final "Detailed Results" dict is the authoritative, full-precision source.
    m = re.search(r"'mAP_0\.25':\s*([0-9.eE+-]+)", text)
    if not m:
        return None
    out = {"mAP_0.25": float(m.group(1))}
    m50 = re.search(r"'mAP_0\.50':\s*([0-9.eE+-]+)", text)
    if m50:
        out["mAP_0.50"] = float(m50.group(1))
    mar = re.search(r"'mAR_0\.25':\s*([0-9.eE+-]+)", text)
    if mar:
        out["mAR_0.25"] = float(mar.group(1))
    per_class = {}
    for name, val in re.findall(r"'([a-z_]+)_AP_0\.25':\s*([0-9.eE+-]+)", text):
        per_class[name] = float(val)
    out["per_class_AP_0.25"] = per_class
    out["num_classes_evaluated"] = len(per_class)
    return out


def load_manifest(proto):
    p = CKPTS / f"sunrgbd_{proto}" / "manifest.json"
    if not p.exists():
        return {}
    man = json.loads(p.read_text())
    return {int(s["stage"]): s.get("mAP@0.25") for s in man.get("stages", [])}


def main():
    results = {}
    lines = []
    lines.append("LDMR — SUN RGB-D checkpoint evaluation (40-class metadata)")
    lines.append("Reproduced on 2x RTX 3090, val split = 5050 scenes")
    lines.append("=" * 78)

    for proto in PROTOCOLS:
        reported = load_manifest(proto)
        n_stages = max(reported) if reported else 0
        lines.append("")
        lines.append(f"## sunrgbd_{proto}")
        lines.append("")
        lines.append(f"{'stage':>6} | {'ours mAP@.25':>12} | {'reported':>9} | "
                     f"{'delta':>8} | {'ours mAP@.50':>12} | {'#cls':>4}")
        lines.append("-" * 78)
        results[proto] = {}
        for stage in range(1, n_stages + 1):
            log = LOGS / f"eval_{proto}_s{stage:02d}.log"
            r = parse_log(log)
            rep = reported.get(stage)
            if r is None:
                status = "no result" if log.exists() else "not run"
                lines.append(f"{stage:>6} | {status:>12} | "
                             f"{(f'{rep:.4f}' if rep is not None else '-'):>9} | "
                             f"{'-':>8} | {'-':>12} | {'-':>4}")
                continue
            delta = (r["mAP_0.25"] - rep) if rep is not None else None
            results[proto][stage] = {**r, "reported_mAP_0.25": rep,
                                     "delta": delta}
            lines.append(
                f"{stage:>6} | {r['mAP_0.25']:>12.4f} | "
                f"{(f'{rep:.4f}' if rep is not None else '-'):>9} | "
                f"{(f'{delta:+.4f}' if delta is not None else '-'):>8} | "
                f"{r.get('mAP_0.50', float('nan')):>12.4f} | "
                f"{r['num_classes_evaluated']:>4}")

    # Reproduction verdict on the final stage of each protocol
    lines.append("")
    lines.append("=" * 78)
    lines.append("FINAL-STAGE REPRODUCTION SUMMARY")
    lines.append("=" * 78)
    for proto in PROTOCOLS:
        rep_all = load_manifest(proto)
        if not rep_all:
            continue
        final = max(rep_all)
        got = results.get(proto, {}).get(final)
        if not got:
            lines.append(f"  sunrgbd_{proto:<8} stage {final:>2}: NOT COMPLETED")
            continue
        d = got["delta"]
        verdict = "MATCH" if d is not None and abs(d) < 0.005 else "CHECK"
        lines.append(f"  sunrgbd_{proto:<8} stage {final:>2}: "
                     f"ours {got['mAP_0.25']:.4f} vs reported "
                     f"{got['reported_mAP_0.25']:.4f}  ({d:+.4f})  [{verdict}]")

    # Forgetting curve: per-class AP of the first stage's classes over time
    lines.append("")
    lines.append("=" * 78)
    lines.append("FORGETTING CURVE (mAP@0.25 over all seen classes, per stage)")
    lines.append("=" * 78)
    for proto in PROTOCOLS:
        seq = results.get(proto, {})
        if not seq:
            continue
        pts = " -> ".join(f"s{k}:{v['mAP_0.25']:.4f}" for k, v in sorted(seq.items()))
        lines.append(f"  {proto:<8} {pts}")
    lines.append("")
    lines.append("NOTE: mAP at each stage is computed over the classes seen SO FAR,")
    lines.append("so the sequence is not a like-for-like comparison across stages —")
    lines.append("later stages average over a larger, harder class set.")

    report = "\n".join(lines)
    print(report)
    (LOGS / "results.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
