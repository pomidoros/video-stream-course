# FlowNetS и FlowNetC

Откройте [lesson.ipynb](lesson.ipynb): собственная запись архитектуры FlowNetS, готовая FlowNetC и сравнение с Farnebäck на видео и известном переносе. Сети используют предобученные веса; обучение здесь не запускается. Готовые видеосравнения находятся в `outputs/`.

[Установка и запуск](../README.md). Из корня репозитория сначала выполните `python flownet_lesson/fetch_assets.py`. Два checkpoint занимают примерно 300 МБ; источники и SHA-256 — в `assets.json`. В Git веса не включены.

Архитектуры адаптированы из [FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch). Исходники и MIT-лицензия сохранены в `vendor/`. Для FlowNetC используется CPU-адаптер correlation на PyTorch.

Для проверки из этой папки запустите `python validate.py`, для пересоздания роликов — `python run_demo.py`.
