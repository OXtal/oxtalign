"""The CLI verbs, the bare two-file legacy form, and the overlay writers."""
import csv
import json
import shutil

from oxtalign.cli import main


def test_bare_two_files_is_compare(exp, capsys):
    assert main([str(exp / "HOLSUX" / "HOLSUX.cif"), str(exp / "HOLSUX" / "HOLSUX01.cif")]) == 0
    out = capsys.readouterr().out
    assert "RMSD_15 = 0.09" in out and "matched 15/15" in out


def test_compare_json_and_profile(exp, capsys):
    a, b = str(exp / "HOLSUX" / "HOLSUX.cif"), str(exp / "HOLSUX" / "HOLSUX01.cif")
    main(["compare", a, b, "--json"])
    d = json.loads(capsys.readouterr().out)
    assert d["n_matched"] == 15 and d["status"] == "full" and abs(d["rmsd_n"] - 0.0925) < 1e-3
    main(["compare", a, b, "--profile"])
    assert capsys.readouterr().out.count("matched 15/15") == 3


def test_matrix_legacy_flag_and_csv(exp, tmp_path, capsys):
    files = [str(exp / "HOLSUX" / "HOLSUX.cif"), str(exp / "HOLSUX" / "HOLSUX01.cif"),
             str(exp / "CAPRYL.cif")]
    out = tmp_path / "m.csv"
    assert main([*files, "--matrix", "--csv", str(out)]) == 0
    rows = list(csv.reader(open(out)))
    assert rows[0] == ["file", *files]
    assert float(rows[1][2]) < 0.1 and rows[1][3] == "inf" and rows[3][3] == "0.000000"


def test_dedup_groups_matching_packings(exp, tmp_path, capsys):
    files = [str(exp / "HOLSUX" / "HOLSUX.cif"), str(exp / "HOLSUX" / "HOLSUX01.cif"),
             str(exp / "CAPRYL.cif"), str(exp / "ROY" / "QAXMEH24.cif"), str(exp / "ROY" / "ROY_R05.cif")]
    out = tmp_path / "clusters.csv"
    assert main(["dedup", *files, "--csv", str(out)]) == 0
    text = capsys.readouterr().out
    assert "4 distinct packings among 5 structures" in text
    rows = list(csv.DictReader(open(out)))
    by_file = {r["file"]: r for r in rows}
    assert by_file[files[0]]["cluster"] == by_file[files[1]]["cluster"]
    assert by_file[files[3]]["cluster"] != by_file[files[4]]["cluster"]
    assert by_file[files[1]]["representative"] == files[0]


def test_score_predictions_against_truth_directory(exp, tmp_path, capsys):
    preds = tmp_path / "preds"
    preds.mkdir()
    shutil.copy(exp / "HOLSUX" / "HOLSUX01.cif", preds / "HOLSUX_pred1.cif")
    shutil.copy(exp / "CAPRYL.cif", preds / "HOLSUX_pred2.cif")            # wrong compound -> fails
    out = tmp_path / "scores.csv"
    assert main(["score", str(preds), "--truth", str(exp / "HOLSUX"), "--csv", str(out)]) == 0
    text = capsys.readouterr().out
    assert "1/2 predictions match" in text
    rows = {r["file"]: r for r in csv.DictReader(open(out))}
    assert rows["HOLSUX_pred1.cif"]["passed"] == "True" and rows["HOLSUX_pred2.cif"]["passed"] == "False"


def test_overlay_writes_html_and_pdb(exp, tmp_path, capsys):
    a, b = str(exp / "HOLSUX" / "HOLSUX.cif"), str(exp / "HOLSUX" / "HOLSUX01.cif")
    html, pdb = tmp_path / "o.html", tmp_path / "o.pdb"
    assert main(["overlay", a, b, "-o", str(html), "--pdb", str(pdb)]) == 0
    assert "matched 15/15" in capsys.readouterr().out
    page = html.read_text()
    assert page.startswith("<!DOCTYPE html>") and '"role": "B"' in page
    lines = pdb.read_text().splitlines()
    atoms = [ln for ln in lines if ln.startswith("HETATM")]
    chains = {ln[21] for ln in atoms}
    per_chain = {c: sum(ln[21] == c for ln in atoms) for c in chains}
    assert chains == {"A", "B"} and per_chain["A"] == per_chain["B"] > 0   # same atoms on both sides
    assert all(len(ln) == 78 for ln in atoms)                              # fixed-column PDB records
    assert any(ln.startswith("CONECT") for ln in lines) and lines[-1] == "END"
    assert max(float(ln[60:66]) for ln in atoms if ln[21] == "B") < 1.0    # B-factor = per-molecule RMSD


def test_version_flag(capsys):
    import pytest

    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out.startswith("oxtalign 0.")
