# Интерактивный отчёт

`index.html` — готовая оболочка. Она ожидает рядом файл `data.js`. После получения результатов всех трёх сцен выполните из папки `starter_kit`:

```bash
python3 -m src.export_dashboard \
  --project .. \
  --results-root results \
  --output interactive/data.js
```

Затем перейдите в общую папку, где лежат `data/`, `docs/` и `starter_kit/`, и запустите сервер:

```bash
cd ..
python3 -m http.server 8000
```

Адрес: `http://localhost:8000/starter_kit/interactive/`.

Не редактируйте `data.js` вручную: исправляйте исходные `tracks.csv`, `events.csv`, `summary.json` или код трекера и повторяйте экспорт.
