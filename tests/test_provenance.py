"""
tests/test_provenance.py — Testes de exemplo para src/provenance.py.

Verifica:
- MD5 de arquivo com conteúdo conhecido.
- Formato correto do JSON de proveniência.
- Campos obrigatórios presentes (card_version, access_date, files).
"""

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from src.provenance import compute_md5, extract_card_version, generate_provenance


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_known_file(tmp_path: Path) -> Path:
    """Cria um arquivo temporário com conteúdo ASCII conhecido."""
    f = tmp_path / "known.txt"
    f.write_bytes(b"hello world")
    return f


@pytest.fixture
def tmp_readme(tmp_path: Path) -> Path:
    """README simulado com versão semântica do CARD."""
    readme = tmp_path / "CARD-Download-README.txt"
    readme.write_text(
        "CARD Download README\nCARD version 3.2.6 released on 2024-01-01\n",
        encoding="utf-8",
    )
    return readme


@pytest.fixture
def tmp_card_json(tmp_path: Path) -> Path:
    """card.json simulado com campo _version."""
    cj = tmp_path / "card.json"
    cj.write_text(json.dumps({"_version": "3.2.6", "data": {}}), encoding="utf-8")
    return cj


@pytest.fixture
def tmp_card_json_no_version(tmp_path: Path) -> Path:
    """card.json simulado sem campo de versão."""
    cj = tmp_path / "card.json"
    cj.write_text(json.dumps({"data": {}}), encoding="utf-8")
    return cj


# ── Testes de compute_md5 ─────────────────────────────────────────────────────

class TestComputeMd5:
    def test_known_content_matches_expected_md5(self, tmp_known_file: Path):
        """MD5 de 'hello world' deve corresponder ao valor calculado independentemente."""
        expected = hashlib.md5(b"hello world").hexdigest()
        assert compute_md5(tmp_known_file) == expected

    def test_returns_32_char_hex_string(self, tmp_known_file: Path):
        """O MD5 deve ser uma string hexadecimal de 32 caracteres."""
        result = compute_md5(tmp_known_file)
        assert isinstance(result, str)
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_file_has_known_md5(self, tmp_path: Path):
        """MD5 de arquivo vazio é d41d8cd98f00b204e9800998ecf8427e."""
        empty = tmp_path / "empty.bin"
        empty.write_bytes(b"")
        assert compute_md5(empty) == "d41d8cd98f00b204e9800998ecf8427e"

    def test_different_content_different_md5(self, tmp_path: Path):
        """Dois arquivos com conteúdo distinto devem ter hashes distintos."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"foo")
        f2.write_bytes(b"bar")
        assert compute_md5(f1) != compute_md5(f2)

    def test_missing_file_raises(self, tmp_path: Path):
        """Deve levantar FileNotFoundError para arquivo inexistente."""
        with pytest.raises(FileNotFoundError):
            compute_md5(tmp_path / "nonexistent.txt")


# ── Testes de extract_card_version ────────────────────────────────────────────

class TestExtractCardVersion:
    def test_version_from_readme(self, tmp_readme: Path, tmp_card_json: Path):
        """Deve extrair a versão do README quando presente."""
        version = extract_card_version(tmp_readme, tmp_card_json)
        assert version == "3.2.6"

    def test_version_from_card_json_fallback(self, tmp_path: Path, tmp_card_json: Path):
        """Deve usar card.json como fallback quando o README não tem versão."""
        readme = tmp_path / "README.txt"
        readme.write_text("No version info here.\n", encoding="utf-8")
        version = extract_card_version(readme, tmp_card_json)
        assert version == "3.2.6"

    def test_returns_unknown_when_neither_source_has_version(
        self, tmp_path: Path, tmp_card_json_no_version: Path
    ):
        """Deve retornar 'unknown' quando nenhuma fonte contém versão."""
        readme = tmp_path / "README.txt"
        readme.write_text("No version here either.\n", encoding="utf-8")
        version = extract_card_version(readme, tmp_card_json_no_version)
        assert version == "unknown"

    def test_missing_readme_falls_back_to_json(
        self, tmp_path: Path, tmp_card_json: Path
    ):
        """Deve usar card.json quando o arquivo README não existe."""
        missing_readme = tmp_path / "missing_readme.txt"
        version = extract_card_version(missing_readme, tmp_card_json)
        assert version == "3.2.6"

    def test_missing_both_files_returns_unknown(self, tmp_path: Path):
        """Deve retornar 'unknown' quando ambos os arquivos estão ausentes."""
        version = extract_card_version(
            tmp_path / "no_readme.txt",
            tmp_path / "no_card.json",
        )
        assert version == "unknown"


# ── Testes de generate_provenance ─────────────────────────────────────────────

class TestGenerateProvenance:
    def _make_data_files(self, tmp_path: Path) -> dict[str, Path]:
        """Cria arquivos temporários com conteúdo conhecido para os testes."""
        files = {}
        for name, content in [("aro_index", b"aro data"), ("card_json", b"card data")]:
            p = tmp_path / f"{name}.tsv"
            p.write_bytes(content)
            files[name] = p
        return files

    def test_required_fields_present(self, tmp_path: Path):
        """O JSON gerado deve conter card_version, access_date e files."""
        data_files = self._make_data_files(tmp_path)
        output = tmp_path / "provenance.json"
        result = generate_provenance(data_files, "3.2.6", output)

        assert "card_version" in result
        assert "access_date" in result
        assert "files" in result

    def test_card_version_stored_correctly(self, tmp_path: Path):
        """card_version deve corresponder ao valor passado."""
        data_files = self._make_data_files(tmp_path)
        result = generate_provenance(data_files, "4.0.1", tmp_path / "p.json")
        assert result["card_version"] == "4.0.1"

    def test_access_date_is_today_iso_format(self, tmp_path: Path):
        """access_date deve ser a data de hoje no formato ISO 8601 (YYYY-MM-DD)."""
        data_files = self._make_data_files(tmp_path)
        result = generate_provenance(data_files, "3.2.6", tmp_path / "p.json")
        today = date.today().isoformat()
        assert result["access_date"] == today

    def test_existing_access_date_is_kept(self, tmp_path: Path):
        """Rodar de novo não troca a data de acesso original (TCC2, ponto 13)."""
        data_files = self._make_data_files(tmp_path)
        out = tmp_path / "p.json"
        out.write_text(json.dumps({"access_date": "2026-06-14"}), encoding="utf-8")
        assert generate_provenance(data_files, "4.0.1", out)["access_date"] == "2026-06-14"
        assert generate_provenance(data_files, "4.0.1", out, access_date="2026-01-01")["access_date"] == "2026-01-01"

    def test_sha256_and_relative_paths(self, tmp_path: Path):
        """SHA-256 correto e caminho relativo à raiz informada."""
        data_files = self._make_data_files(tmp_path)
        result = generate_provenance(data_files, "4.0.1", tmp_path / "p.json", base_dir=tmp_path)
        for name, file_path in data_files.items():
            assert result["files"][name]["sha256"] == hashlib.sha256(file_path.read_bytes()).hexdigest()
            assert result["files"][name]["path"] == file_path.name

    def test_files_dict_has_path_and_md5_per_entry(self, tmp_path: Path):
        """Cada entrada em 'files' deve ter as chaves 'path' e 'md5'."""
        data_files = self._make_data_files(tmp_path)
        result = generate_provenance(data_files, "3.2.6", tmp_path / "p.json")
        for logical_name, entry in result["files"].items():
            assert "path" in entry, f"Entrada '{logical_name}' sem 'path'"
            assert "md5" in entry, f"Entrada '{logical_name}' sem 'md5'"

    def test_md5_values_are_correct(self, tmp_path: Path):
        """Os MD5 gravados devem corresponder aos hashes reais dos arquivos."""
        data_files = self._make_data_files(tmp_path)
        result = generate_provenance(data_files, "3.2.6", tmp_path / "p.json")
        for logical_name, file_path in data_files.items():
            expected_md5 = hashlib.md5(file_path.read_bytes()).hexdigest()
            assert result["files"][logical_name]["md5"] == expected_md5

    def test_all_logical_names_present_in_files(self, tmp_path: Path):
        """Todos os nomes lógicos passados devem aparecer em 'files'."""
        data_files = self._make_data_files(tmp_path)
        result = generate_provenance(data_files, "3.2.6", tmp_path / "p.json")
        for name in data_files:
            assert name in result["files"]

    def test_json_file_is_created_and_valid(self, tmp_path: Path):
        """O arquivo JSON deve ser criado e conter JSON válido."""
        data_files = self._make_data_files(tmp_path)
        output = tmp_path / "results" / "data_provenance.json"
        generate_provenance(data_files, "3.2.6", output)

        assert output.exists()
        with output.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        assert loaded["card_version"] == "3.2.6"

    def test_output_directory_created_if_missing(self, tmp_path: Path):
        """O diretório de saída deve ser criado automaticamente se não existir."""
        data_files = self._make_data_files(tmp_path)
        output = tmp_path / "new" / "subdir" / "provenance.json"
        assert not output.parent.exists()
        generate_provenance(data_files, "3.2.6", output)
        assert output.exists()
