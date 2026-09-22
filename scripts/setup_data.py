"""Скачать учебные данные или установить уже скачанный ZIP.

Этот служебный скрипт не относится к заданиям для реализации.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def download(url, destination, size):
    """Скачать ZIP по ссылке Яндекс Диска или прямой HTTPS-ссылке."""
    if urlparse(url).scheme != "https":
        raise ValueError("Нужна HTTPS-ссылка на архив данных")
    host = urlparse(url).hostname
    if host in {"disk.yandex.ru", "disk.yandex.com", "disk.yandex.net", "yadi.sk"}:
        endpoint = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
        with urlopen(endpoint + urlencode({"public_key": url}), timeout=30) as response:
            url = json.load(response)["href"]
        if urlparse(url).scheme != "https":
            raise ValueError("Сервис вернул некорректную ссылку скачивания")
    with urlopen(url, timeout=30) as response:
        content = response.read(size + 1)
    if len(content) != size:
        raise ValueError("Размер ZIP отличается от указанного в assets.json")
    destination.write_bytes(content)


def install(archive, destination, metadata):
    """Проверить SHA-256 и распаковать только папки учебных данных."""
    if archive.stat().st_size != metadata["bytes"]:
        raise ValueError("Размер архива не совпадает. Скачайте файл заново")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError("SHA-256 не совпадает. Нужен архив из assets.json")
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            target = (destination / name).resolve()
            allowed = [
                destination / p for p in ("data", "players/data", "players/videos")
            ]
            if not any(target.is_relative_to(folder.resolve()) for folder in allowed):
                raise ValueError("В архиве обнаружен файл вне папок данных")
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Папка данных не должна вести за пределы проекта")
        bundle.extractall(destination)


def main():
    """Выбрать локальный архив или ссылку и подготовить данные проекта."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, help="Уже скачанный cv-video-data-v1.zip"
    )
    parser.add_argument("--url", help="Ссылка на файл в Яндекс Диске")
    args = parser.parse_args()
    metadata = json.loads((ROOT / "assets.json").read_text(encoding="utf-8"))["data"]
    try:
        if args.archive:
            install(args.archive, ROOT, metadata)
        else:
            url = args.url or os.environ.get("COURSE_DATA_URL") or metadata["url"]
            if not url:
                parser.error(
                    "Ссылка ещё не добавлена. Используйте --archive ПУТЬ_К_ZIP; см. DATA.md"
                )
            with tempfile.TemporaryDirectory() as folder:
                archive = Path(folder) / metadata["filename"]
                download(url, archive, metadata["bytes"])
                install(archive, ROOT, metadata)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Не удалось подготовить данные: {error}\n")
    print("Данные готовы: data/, players/data/, players/videos/.")


if __name__ == "__main__":
    main()
