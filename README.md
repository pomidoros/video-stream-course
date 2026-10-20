# CV: обработка видеопотока

В репозитории отдельно хранятся примеры с занятий и домашние задания. Для начала выберите нужный раздел.

| Раздел | Что внутри | Инструкция |
| --- | --- | --- |
| Урок 2: оптический поток | Три notebook: Lucas–Kanade, Farnebäck, FlowNetS / FlowNetC; ползунки и готовые видео | [Открыть материалы](lessons/lesson_02/README.md) |
| Домашнее задание | Анализ баскетбольного видео: мяч, броски, трекинг и Re-ID | [Открыть задание](homework/final_project/README.md) |

## Скачать

```bash
git clone https://github.com/pomidoros/video-stream-course.git
cd video-stream-course
```

Можно также скачать архив через **Code → Download ZIP**. Используется **Python 3.12**. GPU для примеров и задания не требуется.

## Быстро запустить урок 2

```bash
cd lessons/lesson_02
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python flownet_lesson/fetch_assets.py
jupyter lab
```

На Windows первые две команды окружения замените на:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Откройте один из notebook и нажмите **Run All**:

1. [Разреженный Lucas–Kanade](lessons/lesson_02/optical_flow_lesson/lesson.ipynb) — OpenCV и простая собственная LK.
2. [Плотный Farnebäck](lessons/lesson_02/dense_optical_flow_lesson/lesson.ipynb) — готовый метод OpenCV и интерактивные эксперименты.
3. [FlowNet](lessons/lesson_02/flownet_lesson/lesson.ipynb) — архитектура FlowNetS и сравнение с готовой FlowNetC.

В VS Code откройте папку `lessons/lesson_02`, установите расширения Python и Jupyter от Microsoft и выберите kernel из её `.venv`. Окружение общее для трёх notebook. В [инструкции урока 2](lessons/lesson_02/README.md) есть пояснения ползунков, `gap`, EPE и видеоплееров.

Веса FlowNet скачиваются один раз, около 300 МБ. Для первых двух notebook эта команда не нужна. В Git веса не хранятся. Готовые ролики уже лежат в папках `outputs/`.

## Начать домашнее задание

```bash
cd homework/final_project
```

Далее следуйте [инструкции задания](homework/final_project/README.md): установка зависимостей, получение данных, этапы реализации и проверки. Все команды ДЗ выполняются из этой папки. Исходные заготовки и условия сохранены.

Окружение ДЗ создаётся отдельно от окружения урока: наборы зависимостей различаются. GitHub Actions проверяет домашнее задание из `homework/final_project`.
