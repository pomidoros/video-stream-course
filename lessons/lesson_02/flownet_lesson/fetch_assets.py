"""Скачать два checkpoint по закреплённым ссылкам из assets.json."""

import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

DIRECTORY = Path(__file__).resolve().parent


def download_weights():
    weights = json.loads((DIRECTORY / "assets.json").read_text())["weights"]
    (DIRECTORY / "weights").mkdir(exist_ok=True)
    for info in weights.values():
        path = DIRECTORY / "weights" / info["filename"]
        # Уже скачанный и проверенный файл не загружаем повторно.
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]:
            print("Уже скачано:", path.name)
            continue

        # Незавершённая загрузка не должна заменить рабочие веса.
        temporary = path.with_suffix(".part")
        with urlopen(info["url"], timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != info["sha256"]:
            raise RuntimeError(f"Не совпала SHA-256 файла {path.name}")
        temporary.replace(path)
        print("Скачано:", path.name)


if __name__ == "__main__":
    download_weights()
