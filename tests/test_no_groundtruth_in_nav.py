"""Navigation modules must not import the ground-truth API (vlnav.groundtruth)."""
import ast
import os

ALLOWED = {"groundtruth.py", "evaluate.py", "survey.py", "logger.py",     # evaluator / logger / survey
           "mapping_flight.py", "map_eval.py", "loc_eval.py"}                              # Phase-A data collection (GT-steered) + eval
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vlnav")


def _imports(path):
    for n in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
        if isinstance(n, ast.Import):
            yield from (a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            yield ("." * n.level) + (n.module or "")
            yield from (("." * n.level) + (n.module or "") + "." + a.name for a in n.names)


def test_nav_modules_do_not_import_groundtruth():
    bad = []
    for f in sorted(os.listdir(ROOT)):
        if f.endswith(".py") and f not in ALLOWED:
            for imp in _imports(os.path.join(ROOT, f)):
                if "groundtruth" in imp:
                    bad.append((f, imp))
    assert not bad, bad


def test_checker_detects_violation(tmp_path):
    p = tmp_path / "x.py"; p.write_text("from .groundtruth import GroundTruth\n")
    assert any("groundtruth" in i for i in _imports(str(p)))
