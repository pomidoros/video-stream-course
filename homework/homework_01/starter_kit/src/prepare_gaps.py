"""Подготовить семь вариантов сцены с разной длиной пропуска измерений."""

import argparse
import csv
import shutil
from pathlib import Path

from .data import load_sequence_info

GAPS = (0, 5, 10, 15, 20, 25, 30)
FILES = ("video.mp4", "seqinfo.yaml", "zones.json", "ground_truth.csv", "expected_counts.json")


def prepare_gaps(source: Path, output: Path, start_frame: int = 40) -> list[Path]:
    """Создать копии сцены с пропусками 0, 5, …, 30 кадров.

    Параметры
    ----------
    source : Path
        Открытая сцена с measurements.csv, видео и разметкой для оценщика.
    output : Path
        Новый каталог, вне исходной сцены. Существующие файлы не заменяются.
    start_frame : int
        Первый кадр пропуска, нумерация с единицы. Измерения удаляются
        на интервале [start_frame, start_frame + gap) для всех объектов.

    Возвращает
    ----------
    list[Path]
        Папки gap_00, gap_05, …, gap_30. Видео, разметка и настройки
        копируются без изменений; gap_00 содержит исходные измерения.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Каталог вариантов должен находиться отдельно от исходной сцены")
    if output.exists():
        raise FileExistsError(f"Каталог уже существует: {output}. Укажите новый --output.")
    for name in (*FILES, "measurements.csv"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    info = load_sequence_info(source / "seqinfo.yaml")
    if start_frame < 1 or start_frame + max(GAPS) - 1 > info.frames:
        raise ValueError("Все интервалы пропусков должны помещаться в исходное видео")
    with (source / "measurements.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or "frame" not in fields:
            raise ValueError("В measurements.csv нужен столбец frame")
        rows = list(reader)
    # Проверяем номера до создания выходных файлов.
    frame_numbers = [int(row["frame"]) for row in rows]
    if any(frame < 1 or frame > info.frames for frame in frame_numbers):
        raise ValueError("Номер кадра в measurements.csv выходит за границы сцены")
    output.mkdir(parents=True)
    variants = []
    for gap in GAPS:
        target = output / f"gap_{gap:02d}"
        target.mkdir()
        for name in FILES:
            shutil.copy2(source / name, target / name)
        if gap == 0:
            shutil.copy2(source / "measurements.csv", target / "measurements.csv")
        else:
            with (target / "measurements.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(row for frame, row in zip(frame_numbers, rows)
                                 if not start_frame <= frame < start_frame + gap)
        variants.append(target)
    return variants


def main() -> None:
    """Подготовить варианты из терминала, не запуская алгоритм студента."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("../data/debug"))
    parser.add_argument("--output", type=Path, default=Path("../data/gap_experiment"))
    parser.add_argument("--start-frame", type=int, default=40)
    args = parser.parse_args()
    for path in prepare_gaps(args.data, args.output, args.start_frame):
        print(path)


if __name__ == "__main__":
    main()
