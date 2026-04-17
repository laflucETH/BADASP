from pathlib import Path

from src.tree_builder import build_fasttree


def test_build_fasttree_prefers_fasttreemp_when_available(monkeypatch, tmp_path: Path) -> None:
    trimmed = tmp_path / "trimmed.aln"
    tree_out = tmp_path / "tree.nwk"
    trimmed.write_text(">s1\nAAAA\n>s2\nAAAT\n", encoding="utf-8")

    def _mock_which(cmd):
        if cmd == "FastTreeMP":
            return "/usr/local/bin/FastTreeMP"
        return None

    def _mock_run(cmd, check, stdout=None, env=None):
        assert check is True
        assert cmd[0] == "FastTreeMP"
        assert str(trimmed) in cmd
        assert env is not None
        assert int(env["OMP_NUM_THREADS"]) >= 1
        assert env["OMP_DYNAMIC"] == "FALSE"
        assert env["OMP_PROC_BIND"] == "TRUE"
        stdout.write("(s1:0.1,s2:0.2);\n")

    monkeypatch.setattr("src.tree_builder.shutil.which", _mock_which)
    monkeypatch.setattr("subprocess.run", _mock_run)

    output = build_fasttree(trimmed_alignment=trimmed, tree_output=tree_out)

    assert output == tree_out
    assert tree_out.read_text(encoding="utf-8").strip() == "(s1:0.1,s2:0.2);"


def test_build_fasttree_falls_back_to_fasttree_with_warning(monkeypatch, tmp_path: Path, capsys) -> None:
    trimmed = tmp_path / "trimmed.aln"
    tree_out = tmp_path / "tree.nwk"
    trimmed.write_text(">s1\nAAAA\n>s2\nAAAT\n", encoding="utf-8")

    def _mock_which(cmd):
        if cmd == "FastTreeMP":
            return None
        if cmd == "FastTree":
            return "/usr/local/bin/FastTree"
        return None

    def _mock_run(cmd, check, stdout=None, env=None):
        assert check is True
        assert cmd[0] == "FastTree"
        assert str(trimmed) in cmd
        stdout.write("(s1:0.1,s2:0.2);\n")

    monkeypatch.setattr("src.tree_builder.shutil.which", _mock_which)
    monkeypatch.setattr("subprocess.run", _mock_run)

    output = build_fasttree(trimmed_alignment=trimmed, tree_output=tree_out)

    captured = capsys.readouterr()
    assert "FastTreeMP not found" in captured.err
    assert output == tree_out
    assert tree_out.read_text(encoding="utf-8").strip() == "(s1:0.1,s2:0.2);"
