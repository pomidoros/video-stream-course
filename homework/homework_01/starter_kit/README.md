# С чего начать домашнюю работу

Прочитайте [условие «Камера ослепла»](../docs/assignment.md) и посмотрите [первый ролик](../data/debug/video.mp4). Вы дописываете прогноз движения, трекер и счётчик; чтение видео и визуализация уже готовы.

## Установка

Рекомендуется Python 3.11 или 3.12, GPU не требуется. Из папки `starter_kit` выполните:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests -q
```

На Windows создайте окружение командой `py -m venv .venv` и активируйте
`.venv\Scripts\Activate.ps1` в PowerShell. После активации вместо `python3`
в дальнейших командах можно использовать `python`.

Пока TODO не выполнены, десять алгоритмических тестов падают с `NotImplementedError` — это ожидаемо. Проверки ещё не созданных результатов помечаются `skipped`. Допишите `src/kalman.py`, `src/tracker.py`, `src/counter.py` и отмеченный блок в `src/main.py`.

## Первый запуск

После реализации выполните:

```bash
python3 -m src.main --data ../data/debug --output results/debug --config configs/easy.yaml
python3 -m src.evaluate --data ../data/debug --results results/debug
```

Откройте `results/debug/demo.mp4`: на видео появятся ID, прогнозы и счётчики. Рядом сохранятся `tracks.csv`, `events.csv` и `summary.json`.

Далее переходите к development, challenge и эксперименту. Готовые конфиги можно оставить без изменений для полной оценки. Варианты с пропусками создаёт готовый `python3 -m src.prepare_gaps`; подробные команды приведены в памятке. Форматы данных, правила реализации, остальные команды и обозначения на видео собраны в [технической памятке](../docs/technical_guide.md).
