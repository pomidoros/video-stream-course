# Урок 2: оптический поток

Три Jupyter notebook: разреженный Lucas–Kanade, плотный Farnebäck и нейросети FlowNet. Внутри есть комментарии к коду, интерактивные параметры и готовые видеосравнения.

## Что открыть

| Notebook | Содержание |
| --- | --- |
| [Lucas–Kanade](optical_flow_lesson/lesson.ipynb) | OpenCV PyrLK, короткая собственная LK, поиск движения точек, проверка вперёд–назад |
| [Farnebäck](dense_optical_flow_lesson/lesson.ipynb) | Готовый плотный поток OpenCV, ползунки, разные изображения, компенсация движения |
| [FlowNet](flownet_lesson/lesson.ipynb) | Архитектура FlowNetS на PyTorch, готовая FlowNetC, сравнение с Farnebäck |

Рекомендуемый порядок — сверху вниз по этой таблице. Для всех трёх notebook используется одно окружение **Python 3.12**. GPU не требуется.

## Установка

Скачайте репозиторий через **Code → Download ZIP** и распакуйте его либо выполните:

```bash
git clone https://github.com/pomidoros/video-stream-course.git
cd video-stream-course/lessons/lesson_02
```

В терминале из папки `lessons/lesson_02`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

На Windows вместо первых двух команд:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Для FlowNet один раз скачайте веса:

```bash
python flownet_lesson/fetch_assets.py
```

Это около 300 МБ. Источники и контрольные суммы находятся в [assets.json](flownet_lesson/assets.json). Повторный запуск пропускает уже проверенные файлы. Для первых двух notebook веса не нужны.

## Запуск в VS Code

1. Откройте папку `lessons/lesson_02` в VS Code. Установите расширения **Python** и **Jupyter** от Microsoft, если их ещё нет.
2. Откройте нужный `lesson.ipynb` из таблицы выше.
3. Нажмите **Select Kernel → Python Environments** и выберите `.venv` из этого репозитория. Если её нет в списке, укажите `.venv/bin/python` на macOS/Linux или `.venv\Scripts\python.exe` на Windows.
4. Нажмите **Run All**. Во всех трёх notebook выбирайте одно и то же окружение.

Ноутбуки работают при запуске из папки `lessons/lesson_02` и из своей папки. Первое выполнение FlowNet включает загрузку моделей в память; затем ползунки используют уже загруженные модели.

## Запуск без VS Code

После активации `.venv`:

```bash
jupyter lab
```

Откройте notebook и выберите **Run → Run All Cells**. Ползунки работают при живом kernel; в просмотре файлов на GitHub они не пересчитываются.

## Как пользоваться

- **Lucas–Kanade:** меняйте `dx, dy` в синтетической паре и повторяйте ячейки создания пары и вызова OpenCV. Собственная LK выполняет один линейный шаг и проверяется на малом переносе. В конце есть видеоплееры.
- **Farnebäck:** начните с `kind=Текстура`, `(dx, dy)=(0.5, 0.25)`, `scale=1`. Для большого переноса `(12, 6)` поставьте `scale=16` и сравните L1/L3. `kind` меняет входное изображение, `mode` — цвет или стрелки. Ползунки пересчитывают результат после отпускания мыши.
- **FlowNet:** сравните Farnebäck, FlowNetS и FlowNetC на одной видеопаре. На известном переносе посмотрите EPE. Здесь используются готовые веса, обучение не запускается.
- **Видео:** нажмите ▶ во встроенном плеере. Все готовые MP4 также лежат в папках `outputs/` и открываются отдельно.

Обозначения:

| Обозначение | Значение |
| --- | --- |
| `(u, v)` | Смещение в пикселях: вправо и вниз положительны |
| `gap=5` | Сравниваются кадры, например, №10 и №15; стрелка показывает движение за этот интервал |
| `80/80` | Принято 80 соответствий из 80 выбранных точек; это не 100% точности |
| EPE | Ошибка вектора относительно известного правильного движения; меньше — точнее |
| `scale` | Верхняя граница яркости цветового поля; сам поток не меняет |
| PyrLK `maxLevel=3` | Максимальный индекс пирамиды: разрешены уровни 0…3 |
| Farnebäck `levels=3` | Число уровней вместе с исходным; 1 отключает дополнительные масштабы |

Если цветовое поле почти чёрное, уменьшите `scale` либо увеличьте интервал `gap`. Смена `scale` влияет только на цветовое отображение; в режиме стрелок сохраняется масштаб 1:1. На реальных видео правильный поток неизвестен, поэтому EPE там не вычисляется.

## Проверки и пересоздание видео

Из папки `lessons/lesson_02`, с активным окружением:

```bash
python optical_flow_lesson/validate.py
python dense_optical_flow_lesson/validate.py
python flownet_lesson/validate.py
```

Для последней проверки нужны скачанные веса. Пересоздать видеосравнения:

```bash
python optical_flow_lesson/run_demo.py
python dense_optical_flow_lesson/run_demo.py
python flownet_lesson/run_demo.py
```

Параметры сохранения находятся в конце каждого скрипта. Для обычного просмотра пересоздавать ролики не требуется.

## Источники

[OpenCV optical flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html), [FlowNet 2015](https://arxiv.org/abs/1504.06852), [FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch). Исходники FlowNet и MIT-лицензия автора сохранены в [flownet_lesson/vendor](flownet_lesson/vendor/).
