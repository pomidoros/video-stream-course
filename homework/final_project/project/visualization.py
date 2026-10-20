"""Готовая отрисовка предсказаний. Алгоритм трекинга здесь не реализован."""

from pathlib import Path
import colorsys
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from project.common import dump, checked_detections
from project.contracts import people_rows, ball_rows, events as validate_events
from players.people_io import load_manifest, load_clip, load_labels
from shots.outcome_io import load_recordings, load_features

BG = (13, 20, 32)
CARD = (24, 35, 51)
WHITE = (238, 244, 251)
MUTED = (168, 184, 203)
ORANGE = (255, 170, 95)
FONTS = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in FONTS:
        FONTS[key] = ImageFont.truetype(
            font_manager.findfont(
                font_manager.FontProperties(
                    family="DejaVu Sans", weight="bold" if bold else "normal"
                )
            ),
            size,
        )
    return FONTS[key]


def text(d, xy, value, size=22, color=WHITE, bold=False):
    d.text(xy, str(value), font=font(size, bold), fill=color)


def color(identity):
    return tuple(
        int(255 * v)
        for v in colorsys.hsv_to_rgb((int(identity) * 0.61803398875) % 1, 0.48, 0.97)
    )


class Writer:
    def __init__(self, path, size, fps):
        import imageio_ffmpeg

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio_ffmpeg.write_frames(
            str(self.path),
            size,
            fps=fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
            output_params=["-crf", "21", "-movflags", "+faststart"],
        )
        self.writer.send(None)
        self.count = 0

    def add(self, canvas):
        self.writer.send(np.asarray(canvas, dtype=np.uint8))
        self.count += 1

    def close(self):
        self.writer.close()


def dashed(d, b, c):
    x1, y1, x2, y2 = map(int, b)
    for x in range(x1, x2, 14):
        d.line((x, y1, min(x + 8, x2), y1), fill=c, width=3)
        d.line((x, y2, min(x + 8, x2), y2), fill=c, width=3)
    for y in range(y1, y2, 14):
        d.line((x1, y, x1, min(y + 8, y2)), fill=c, width=3)
        d.line((x2, y, x2, min(y + 8, y2)), fill=c, width=3)


def panel(image, tracks, size=(920, 518), truth=False):
    h, w = image.shape[:2]
    out = Image.fromarray(cv2.cvtColor(cv2.resize(image, size), cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(out)
    occupied = []
    for track in sorted(
        [t for t in tracks if t["bbox"] is not None], key=lambda t: t["bbox"][1]
    ):
        identity = track["person_id"] if truth else int(track["track_id"])
        c = color(identity)
        b = np.rint(
            np.array(track["bbox"])
            * np.array([size[0] / w, size[1] / h, size[0] / w, size[1] / h])
        ).astype(int)
        x1, y1, x2, y2 = b
        if x2 < 0 or y2 < 0 or x1 >= size[0] or y1 >= size[1]:
            continue
        x1, y1, x2, y2 = (
            max(0, x1),
            max(0, y1),
            min(size[0] - 1, x2),
            min(size[1] - 1, y2),
        )
        observed = truth or track["observed"]
        if observed:
            d.rectangle((x1, y1, x2, y2), outline=c, width=3)
        else:
            dashed(d, (x1, y1, x2, y2), c)
        label = (
            ("Эталон " if truth else "ID ")
            + str(identity)
            + (" · прогноз" if not observed else "")
        )
        tw = int(d.textlength(label, font=font(18, True))) + 14
        th = 29
        options = []
        for yy in [y1 - th - 3, y1 + 2, y2 + 3, y1 - 2 * th - 6]:
            for xx in [x1, x2 - tw, x2 + 4, x1 - tw - 4]:
                x = max(1, min(size[0] - tw - 2, xx))
                y = max(1, min(size[1] - th - 2, yy))
                options.append((x, y, x + tw, y + th))

        def overlap(a, b):
            return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

        tag = min(
            options,
            key=lambda r: (
                sum(overlap(r, o) for o in occupied),
                abs(r[0] - x1) + abs(r[1] - (y1 - th - 3)),
            ),
        )
        occupied.append(tag)
        x, y, _, _ = tag
        d.line((x + tw // 2, y + th // 2, (x1 + x2) // 2, y1), fill=c)
        d.rounded_rectangle(tag, radius=5, fill=BG, outline=c, width=2)
        text(d, (x + 7, y + 3), label, 18, c, True)
    return out


def render_people(root, results, output, clip_name, mode="prediction", max_frames=None):
    C = Path(root) / "players"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    clip = next(c for c in load_manifest(C)["clips"] if c["name"] == clip_name)
    data = load_clip(C, clip)
    rows = json.loads(
        (Path(results) / "people" / f"{clip_name}.normal.tracks.json").read_text()
    )
    people_rows(rows, data, clip["fps"])
    lookup = {r["frame"]: r for r in rows}
    gt = load_labels(C, clip) if mode == "comparison" else []
    cap = cv2.VideoCapture(str(C / clip["source_video"]))
    writer = Writer(
        output / f"{clip_name}.{mode}.mp4", (1920, 1080), clip["fps"] / clip["stride"]
    )
    cards = {}
    frame = 0
    try:
        while True:
            ok, im = cap.read()
            if not ok:
                break
            f = frame
            frame += 1
            if f not in lookup:
                continue
            r = lookup[f]
            tracks = [t for t in r["tracks"] if t["confirmed"]]
            for t in tracks:
                identity = int(t["track_id"])
                if identity not in cards and t["observed"]:
                    i = int(t["detection_index"])
                    x1, y1, x2, y2 = data["bbox"][i].astype(int)
                    crop = Image.fromarray(
                        cv2.cvtColor(
                            im[
                                max(0, y1) : min(im.shape[0], y2),
                                max(0, x1) : min(im.shape[1], x2),
                            ],
                            cv2.COLOR_BGR2RGB,
                        )
                    )
                    crop.thumbnail((48, 70))
                    cards[identity] = dict(image=crop, last_frame=f)
                if identity in cards and t["observed"]:
                    cards[identity]["last_frame"] = f
            canvas = Image.new("RGB", (1920, 1080), BG)
            d = ImageDraw.Draw(canvas)
            text(
                d,
                (28, 20),
                "Сопровождение игроков • результат вашего алгоритма",
                34,
                bold=True,
            )
            text(
                d,
                (28, 75),
                f'{clip_name}  •  {f/clip["fps"]:.2f} с  •  Наблюдается: {sum(t["observed"] for t in tracks)}  •  Всего создано подтверждённых ID: {len(cards)}',
                22,
                MUTED,
            )
            if mode == "comparison":
                text(
                    d,
                    (28, 129),
                    "ЭТАЛОН • используется только для сравнения",
                    23,
                    bold=True,
                )
                text(d, (972, 129), "ВАШИ ПРЕДСКАЗАНИЯ", 23, bold=True)
                canvas.paste(
                    panel(im, [g for g in gt if g["frame"] == f], truth=True), (28, 174)
                )
                canvas.paste(panel(im, tracks), (972, 174))
            else:
                canvas.paste(panel(im, tracks, (1320, 594)), (28, 133))
            d = ImageDraw.Draw(canvas)
            if mode == "prediction":
                text(d, (1374, 152), "Как читать видео", 26, bold=True)
                for y, line in [
                    (210, "Цвет закреплён за ID."),
                    (258, "Сплошная рамка:"),
                    (288, "есть наблюдение."),
                    (342, "Пунктир:"),
                    (372, "прогноз без детекции."),
                    (427, "Карточка хранит первое"),
                    (457, "подтверждённое фото."),
                    (513, "Номер не исправляется"),
                    (543, "по правильной разметке."),
                ]:
                    text(d, (1374, y), line, 22, MUTED)
            current = {int(t["track_id"]): t for t in tracks}
            displayed = sorted(
                cards, key=lambda i: (i not in current, -cards[i]["last_frame"], i)
            )[:8]
            for k, identity in enumerate(displayed):
                x = 28 + k % 4 * 472
                y = 751 + k // 4 * 99
                c = color(identity)
                d.rounded_rectangle(
                    (x, y, x + 451, y + 87), radius=8, fill=CARD, outline=c, width=2
                )
                canvas.paste(cards[identity]["image"], (x + 10, y + 8))
                d = ImageDraw.Draw(canvas)
                t = current.get(identity)
                status = (
                    "Наблюдается"
                    if t and t["observed"]
                    else (
                        "Прогноз позиции"
                        if t and t["bbox"] is not None
                        else "Не найден"
                    )
                )
                text(d, (x + 74, y + 9), f"Игрок / трек ID {identity}", 22, c, True)
                text(
                    d,
                    (x + 74, y + 46),
                    status,
                    20,
                    WHITE if t and t["observed"] else ORANGE,
                )
            text(
                d,
                (28, 963),
                "Карточки последних активных треков. Все рамки показаны; номера и ошибки не скрываются.",
                21,
                MUTED,
            )
            text(
                d,
                (28, 1000),
                "Номер трека может отличаться от эталонного ID. Один человек при потере может получить новый номер.",
                21,
                MUTED,
            )
            text(
                d,
                (28, 1040),
                "TrackID3x3 · Yamada et al. · CC BY 4.0 · Видео подготовлено готовым скриптом курса",
                18,
                MUTED,
            )
            writer.add(canvas)
            if writer.count == min(30, max_frames or 30):
                canvas.save(output / f"{clip_name}.{mode}.jpg", quality=94)
            if max_frames is not None and writer.count >= max_frames:
                break
        if max_frames is None and writer.count != len(rows):
            raise ValueError("Video truncated before all prediction frames")
    finally:
        cap.release()
        writer.close()
    report = dict(
        path=str(writer.path),
        frames=writer.count,
        fps=clip["fps"] / clip["stride"],
        mode=mode,
    )
    dump(output / f"{clip_name}.{mode}.render.json", report)
    return report


def render_ball(root, results, output, recording, max_frames=None):
    root = Path(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    row = next(r for r in load_recordings(root) if r["sample_id"] == recording)
    tracks = json.loads((Path(results) / "gru" / f"{recording}.ball.json").read_text())
    pred = json.loads((Path(results) / "gru" / f"{recording}.events.json").read_text())
    ball_rows(tracks, checked_detections(root, row))
    data = load_features(root, row)
    validate_events(pred, data["time_s"])
    times = {int(f) for f in data["frame"]}
    fps = 1 / float(np.median(np.diff(data["time_s"])))
    writer = Writer(output / f"{recording}.ball.mp4", (1280, 800), fps)
    rim = row["rim_bbox"]
    w = rim[2] - rim[0]
    cx = (rim[0] + rim[2]) / 2

    def point(r):
        return int(430 + (r["x"] - cx) / w * 50), int(250 + (r["y"] - rim[1]) / w * 50)

    try:
        for i, r in enumerate(tracks):
            if r["frame"] not in times:
                continue
            canvas = Image.new("RGB", (1280, 800), BG)
            d = ImageDraw.Draw(canvas)
            text(d, (25, 22), "Траектория мяча и исходы бросков", 32, bold=True)
            text(
                d,
                (25, 70),
                f'{recording} • {r["time_s"]:.2f} с • Схема по числовым наблюдениям',
                21,
                MUTED,
            )
            d.rounded_rectangle((25, 125, 860, 690), radius=10, fill=CARD)
            d.ellipse((405, 245, 455, 255), outline=ORANGE, width=4)
            history = [
                point(t) for t in tracks[max(0, i - 40) : i + 1] if t["x"] is not None
            ]
            history = [p for p in history if 30 < p[0] < 855 and 130 < p[1] < 685]
            if len(history) > 1:
                d.line(history, fill=MUTED, width=2)
            if r["x"] is not None:
                x, y = point(r)
                if 30 < x < 855 and 130 < y < 685:
                    d.ellipse(
                        (x - 7, y - 7, x + 7, y + 7),
                        fill=(105, 210, 255) if r["observed"] else ORANGE,
                    )
            emitted = [
                e
                for e in pred
                if e["time_s"] is not None and e["time_s"] <= r["time_s"]
            ]
            lines = [
                "РЕЗУЛЬТАТ МОДЕЛИ",
                f"Выдано: {len(emitted)}",
                f'Попадания: {sum(e["outcome"]=="made" for e in emitted)}',
                f'Промахи: {sum(e["outcome"]=="missed" for e in emitted)}',
            ]
            if emitted:
                last = emitted[-1]
                lines.extend(
                    [
                        "",
                        {
                            "made": "Попадание",
                            "missed": "Промах",
                            "unknown": "Неизвестно",
                        }[last["outcome"]],
                        (
                            f'P = {last["made_probability"]:.3f}'
                            if last["made_probability"] is not None
                            else "Нет вероятности"
                        ),
                    ]
                )
            for k, line in enumerate(lines):
                text(d, (885, 155 + k * 48), line, 23)
            text(
                d,
                (25, 720),
                "Это выходы вашей сети; правильные исходы в визуализацию не передаются.",
                21,
                MUTED,
            )
            text(
                d,
                (25, 756),
                "Координаты вне области схемы не рисуются. MP4 не измеряет скорость работы детектора.",
                20,
                MUTED,
            )
            writer.add(canvas)
            if max_frames is not None and writer.count >= max_frames:
                break
    finally:
        writer.close()
    return dict(path=str(writer.path), frames=writer.count, fps=fps)
