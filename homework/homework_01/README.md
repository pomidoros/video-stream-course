# Домашнее задание 1: «Камера ослепла»

Вы напишете трекер, который сопровождает людей по шумным координатам, переживает короткие пропуски обнаружений и считает входы и выходы. В комплекте три учебных видео, измерения, заготовки кода, тесты и готовая визуализация. GPU не нужен. Максимальная оценка — **20 баллов**.

1. Прочитайте [условие работы](docs/assignment.md): что дано, что реализовать и что сдавать.
2. Откройте [первое видео](data/debug/video.mp4), чтобы познакомиться со сценой.
3. Следуйте [инструкции установки и запуска](starter_kit/README.md).
4. По ходу работы обращайтесь к [технической памятке](docs/technical_guide.md).

Папки `docs/`, `data/` и `starter_kit/` должны оставаться рядом. Данные уже включены в репозиторий, отдельно скачивать их не требуется. Начальные ошибки `NotImplementedError` в десяти тестах ожидаемы: эти алгоритмы предстоит дописать вам.

## Быстрый старт

Из корня репозитория:

```bash
cd homework/homework_01/starter_kit
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests/test_data_format.py tests/test_prepare_gaps.py tests/test_result_format.py -q
```

На Windows используйте `py -3.12 -m venv .venv` и `.venv\Scripts\Activate.ps1`. Окружение этого задания отдельное от урока 2 и итогового проекта.

После выполнения TODO:

```bash
python -m src.main --data ../data/debug --output results/debug --config configs/easy.yaml
python -m src.evaluate --data ../data/debug --results results/debug
```

Откройте `results/debug/demo.mp4`. Переход к development/challenge и эксперимент с пропусками описаны в [технической памятке](docs/technical_guide.md).

[Все домашние задания](../README.md) · [Материалы урока 2](../../lessons/lesson_02/README.md).
