"""Проверки установки данных: код проекта не должен перезаписываться архивом."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.setup_data import download, install


class DataSetupTests(unittest.TestCase):
    def make_archive(self, folder, name):
        """Создать маленький архив и его контрольную сумму для проверки."""
        archive = folder / "data.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr(name, "example")
        metadata = {
            "bytes": archive.stat().st_size,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }
        return archive, metadata

    def test_install_preserves_student_code(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            destination = folder / "project"
            destination.mkdir()
            code = destination / "solution.py"
            code.write_text("student work")
            archive, metadata = self.make_archive(folder, "players/data/example.txt")
            install(archive, destination, metadata)
            self.assertEqual(code.read_text(), "student work")
            self.assertEqual(
                (destination / "players/data/example.txt").read_text(), "example"
            )

    def test_wrong_checksum_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            archive, metadata = self.make_archive(folder, "data/example.txt")
            metadata["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                install(archive, folder / "project", metadata)
            self.assertFalse((folder / "project").exists())

    def test_archive_cannot_write_code_or_escape_project(self):
        for filename in ["shots/outcome_student.py", "data/../../outside.py"]:
            with self.subTest(
                filename=filename
            ), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                archive, metadata = self.make_archive(folder, filename)
                with self.assertRaises(ValueError):
                    install(archive, folder / "project", metadata)
                self.assertFalse((folder / "project").exists())
                self.assertFalse((folder / "outside.py").exists())

    def test_public_yandex_link_is_resolved_before_download(self):
        response = json.dumps(
            {"href": "https://downloader.disk.yandex.ru/example"}
        ).encode()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "data.zip"
            with patch(
                "scripts.setup_data.urlopen",
                side_effect=[io.BytesIO(response), io.BytesIO(b"zip")],
            ) as request:
                download("https://disk.yandex.ru/d/example", destination, 3)
            self.assertEqual(destination.read_bytes(), b"zip")
            self.assertTrue(
                request.call_args_list[0]
                .args[0]
                .startswith(
                    "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key="
                )
            )
            self.assertEqual(
                request.call_args_list[1].args[0],
                "https://downloader.disk.yandex.ru/example",
            )


if __name__ == "__main__":
    unittest.main()
