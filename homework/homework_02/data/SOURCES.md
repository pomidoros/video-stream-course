# Источники видео

Два учебных файла выделены из ранее подготовленного `sports_stream.mp4`. Это фрагменты следующих открытых материалов Wikimedia Commons:

- **bowling.mp4:** RCraig09, [20190818 EJ Tackett lofting bowling ball](https://commons.wikimedia.org/wiki/File:20190818_EJ_Tackett_lofting_bowling_ball.webm), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- **archery.mp4:** Simon Allen, [Field Archery](https://commons.wikimedia.org/wiki/File:Field_Archery.webm), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).

Изменения: обрезка, приведение к 320×180 и 15 FPS, удаление звука, повторное кодирование в MP4. Подготовленные видео распространяются с этой атрибуцией; для производного материала исходного составного ролика сохраняются условия CC BY-SA 4.0. Сведения об авторах и лицензиях перенесены из манифеста исходного учебного комплекта.

`manifest.json` содержит источники, длительности, границы извлечённых фрагментов и SHA-256 файлов. Внутри видео не добавлены названия действий. Метки используются для сравнения результата и подписей, а вход модели — только пиксели.

Модель: Torchvision R3D-18, веса `R3D_18_Weights.KINETICS400_V1`. [Документация модели](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html). Веса не входят в архив.
