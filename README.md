# CV: обработка видеопотока

В репозитории отдельно хранятся примеры с занятий, **два домашних задания** и **итоговый проект**. Для начала выберите нужный раздел.

| Раздел | Что внутри | Инструкция |
| --- | --- | --- |
| Урок 2: оптический поток | Три notebook: Lucas–Kanade, Farnebäck, FlowNetS / FlowNetC; ползунки и готовые видео | [Открыть материалы](lessons/lesson_02/README.md) |
| ДЗ 1: «Камера ослепла» | Калман, трекер, пропуски измерений и подсчёт людей | [Открыть ДЗ 1](homework/homework_01/README.md) |
| ДЗ 2: «Что узнала видеосеть?» | Готовая R3D-18 и исследование роли порядка кадров | [Открыть ДЗ 2](homework/homework_02/README.md) |
| Итоговый проект | Анализ баскетбольного видео: мяч, броски, трекинг и Re-ID | [Открыть проект](homework/final_project/README.md) |

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

## Два домашних задания и итоговый проект

1. **[ДЗ 1: «Камера ослепла»](homework/homework_01/README.md)** — фильтр Калмана, сопровождение людей и подсчёт пересечений линии. Данные и три синтетических видео уже включены.
2. **[ДЗ 2: «Что узнала видеосеть?»](homework/homework_02/README.md)** — исследование ответов готовой R3D-18 на исходные, перемешанные и замороженные кадры. Два видео уже включены; обучение не требуется.
3. **[Итоговый проект: баскетбольное видео](homework/final_project/README.md)** — отдельная работа про броски, трекинг, Re-ID и статистику. Данные устанавливаются из отдельного архива по инструкции проекта.

Все команды выполняются из папки выбранного задания. Для каждой работы создавайте своё окружение: зависимости отличаются от материалов урока 2.

[Общий список заданий и команды начала работы](homework/README.md). GitHub Actions проверяет итоговый проект из `homework/final_project`.
