import json

from tools import time_to_target


def _curve(path, points):
    with path.open("w", encoding="utf-8") as handle:
        for tokens, seconds, loss in points:
            handle.write(json.dumps({"tokens": tokens, "seconds": seconds, "evalLoss": loss}) + "\n")
            handle.write(json.dumps({"tokens": tokens + 1, "seconds": seconds, "trainLoss": 1.0}) + "\n")


def test_time_to_target_reports_first_crossing_and_ratios(tmp_path, capsys):
    _curve(tmp_path / "adam.jsonl", [(100, 1.0, 8.0), (200, 2.0, 6.0), (300, 3.0, 5.0)])
    _curve(tmp_path / "dust.jsonl", [(100, 10.0, 8.0), (400, 40.0, 7.0), (800, 80.0, 6.0)])
    out = tmp_path / "ttt.json"
    assert time_to_target.main([f"adam={tmp_path / 'adam.jsonl'}", f"dust={tmp_path / 'dust.jsonl'}",
                                "--targets", "6.0", "5.0", "--baseline", "adam", "--out", str(out)]) == 0
    rows = {(r["target"], r["run"]): r for r in json.loads(out.read_text())["rows"]}
    assert rows[(6.0, "adam")]["tokens"] == 200
    assert rows[(6.0, "dust")] == {"target": 6.0, "run": "dust", "tokens": 800, "seconds": 80.0,
                                   "tokenRatio": 4.0, "timeRatio": 40.0}
    assert rows[(5.0, "dust")]["tokens"] is None
    assert "never" in capsys.readouterr().out


def test_default_targets_are_reached_by_every_run(tmp_path):
    curves = {"a": [{"tokens": 1, "seconds": 1, "evalLoss": 8.0}, {"tokens": 2, "seconds": 2, "evalLoss": 5.0}],
              "b": [{"tokens": 1, "seconds": 1, "evalLoss": 7.9}, {"tokens": 2, "seconds": 2, "evalLoss": 7.0}]}
    targets = time_to_target.default_targets(curves, 2)
    assert targets == [7.5, 7.0]
    assert all(time_to_target.first_reach(points, t) for points in curves.values() for t in targets)
