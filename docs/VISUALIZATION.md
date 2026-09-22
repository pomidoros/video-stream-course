# Запуск и готовые видео

Все команды выполняются из корня репозитория после реализации заготовок.

```bash
# Всё сразу: обучение, оценка, проверка порогов и два видео
python -m project run --models artifacts --output outputs
```

Или по шагам:

```bash
python -m project train --models artifacts
python -m project evaluate --models artifacts --output outputs
python scripts/check.py metrics --models artifacts
python -m project render-ball --results outputs --output outputs/videos
python -m project render-people --results outputs --output outputs/videos
```

Чтобы показать сравнение игроков с разметкой:

```bash
python -m project render-people --results outputs --output outputs/videos --clip basket_S4T1_post --mode comparison
```

Видео сохраняются в `outputs/videos/` в формате H.264 MP4. Для быстрого предпросмотра добавьте `--max-frames 30` к команде рендера. В сдаче нужны полные ролики.

**Мяч:** схема кольца, траектория и исходы по мере выдачи моделью. **Игроки:** настоящие ID вашего трекера, постоянные цвета, рамки и карточки последних активных треков. Сплошная рамка — наблюдение, пунктир — прогноз; все рамки отображаются независимо от места в панели карточек.

В режиме comparison слева разметка, справа предсказания. В обычном режиме разметка не читается. Скрипт не исправляет ошибки трекинга или исходов. Самостоятельно реализовывать отрисовку не нужно.
