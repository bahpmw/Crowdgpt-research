"""Read-only views over the record for /v2/status, /metrics and the dashboard.

Every value carries a state: ``live`` (current round, not yet recorded),
``provisional`` (recorded, below quorum), ``final`` (quorum co-signed) or
``stale`` (contributor not seen for 2+ rounds).
"""

from __future__ import annotations

from .record import Record


def contributors(record: Record, live: dict[str, dict], current_round: int,
                 strikes: dict[str, int], banned: set[str]) -> list[dict]:
    rows: dict[str, dict] = {}
    for e in record.entries:
        if e["type"] == "register":
            rows.setdefault(e["body"]["key"], _row(e["body"]["key"]))
        if e["type"] != "round":
            continue
        fin = record.is_final(e)
        for c in e["body"]["cells"]:
            row = rows.setdefault(c["key"], _row(c["key"]))
            ok = c["status"] == "accepted"
            row["accepted" if ok else "rejected"] += 1
            row["accepted_final"] += int(ok and fin)
            row.update(last_round=e["round"], module=c["module"], last_status=c["status"],
                       record_state="final" if fin else "provisional")
    out = []
    for kid, row in sorted(rows.items()):
        seen = live.get(kid, {})
        row.update(strikes=strikes.get(kid, 0), banned=kid in banned,
                   self_reported=seen.get("claimed", {}), last_seen=seen.get("t"),
                   live_module=seen.get("module"))
        if row["banned"]:
            row["state"] = "banned"
        elif seen.get("round") == current_round:
            row["state"] = "live"
        elif row["last_round"] and current_round - row["last_round"] >= 2:
            row["state"] = "stale"
        else:
            row["state"] = row["record_state"] or "registered"
        out.append(row)
    return out


def _row(kid: str) -> dict:
    return {"key": kid, "accepted": 0, "rejected": 0, "accepted_final": 0, "last_round": 0,
            "module": None, "last_status": None, "record_state": None}


def loss_curve(record: Record) -> list[dict]:
    return [{"round": e["round"], "loss": e["body"]["eval_loss"],
             "rolled_back": e["body"].get("rolled_back", False),
             "state": "final" if record.is_final(e) else "provisional"}
            for e in record.entries if e["type"] in ("genesis", "round")]


def tail(record: Record, n: int = 15) -> list[dict]:
    return [{"n": e["n"], "type": e["type"], "round": e["round"], "hash": e["hash"][:16],
             "sigs": len(e["sigs"]), "state": "final" if record.is_final(e) else "provisional",
             "summary": _summary(e)} for e in record.entries[-n:]]


def _summary(e: dict) -> str:
    b = e["body"]
    if e["type"] == "round":
        acc = sum(1 for c in b["cells"] if c["status"] == "accepted")
        return f"leader {b['leader']} accepted {acc}/{len(b['cells'])} loss {b['eval_loss']}"
    if e["type"] == "verdict":
        return f"{b['key']} {b['reason']} strikes {b['strikes']}{' BANNED' if b['banned'] else ''}"
    if e["type"] == "register":
        return f"client {b['key']}"
    if e["type"] in ("orchestrator_down", "orchestrator_up"):
        return b["orchestrator"]
    if e["type"] == "rollback":
        return f"rejected loss {b['rejected_loss']} kept {b['kept_loss']}"
    return f"preset {b.get('preset')} K {b.get('K')}"


def finality(record: Record) -> dict:
    fin = sum(1 for e in record.entries if record.is_final(e))
    return {"entries": len(record.entries), "final": fin, "provisional": len(record.entries) - fin}
