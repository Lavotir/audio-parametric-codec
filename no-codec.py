# -*- coding: utf-8 -*-
"""
================================================================================
 NOT·CODEC v1.0 — параметрический текстовый аудиокодек (формат .not)
================================================================================

 ИДЕЯ ФОРМАТА
 ------------
 В отличие от MIDI, который хранит лишь номера стандартных нот и звучит так,
 как сыграет внешний синтезатор, .not хранит САМУ ЗВУКОВУЮ ТКАНЬ: для каждого
 микроскопического шага времени зафиксировано, какие именно синусоидальные
 тоны (точная частота в Гц + громкость) нужно сложить, чтобы заново собрать
 оригинальный звук — со всеми басами, эффектами и шумами электроники. Файл — чистый текст UTF-8.

 СТРУКТУРА ФАЙЛА .not
 --------------------
   #NOT 1.0                 <- сигнатура формата
   #SRC имя_оригинала.mp3   <- источник (для людей)
   #SR  44100               <- частота дискретизации, Гц
   #WIN 2048                <- окно анализа STFT, сэмплов
   #HOP 512                 <- шаг анализа, сэмплов (один шаг = одна строка)
   #PEAKS 150               <- максимум гармоник в кадре
   #FRAMES 15432            <- всего кадров времени
   #DUR 179.43              <- длительность, секунд
   0|439.98:0.5321,880.13:0.2114,...
   1|440.02:0.5488,880.09:0.1902,...
   ...
 Каждая строка после заголовка = один шаг времени:
   <номер_кадра>|<частота_Гц>:<громкость>,<частота>:<громкость>,...

 МАТЕМАТИКА
 ----------
 КОДЕР:   STFT (окно Хэннинга) -> амплитудный спектр -> локальные максимумы
          -> топ-N по громкости -> параболическая интерполяция вершины
          (частота уточняется до долей Гц) -> текст.
 ДЕКОДЕР: для каждого кадра строится сумма синусоид длиной 2*hop, окно
          Хэннинга даёт константу перекрытия (COLA): w[n] + w[n+hop] == 1,
          поэтому кадры сшиваются без швов и щелчков. Фазы устойчивых тонов
          ПРОДОЛЖАЮТСЯ из кадра в кадр (поиск ближайшей частоты) — звук
          чистый, без "роботизированного" фазового дрожания.

 ЗАВИСИМОСТИ: numpy, soundfile, sounddevice, customtkinter
              (librosa — опционально для MP3; ffmpeg — опционально для
              экспорта в MP3, иначе пишется WAV)
================================================================================
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import customtkinter as ctk
    import tkinter.font as tkfont
    from tkinter import filedialog, messagebox
except ImportError:
    ctk = None

# ==============================================================================
#  ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ ДВИЖКА (все настраиваются)
# ==============================================================================

FORMAT_VERSION = "1.0"

ANALYSIS_WIN   = 2048      # окно анализа STFT, сэмплов (частотное разрешение ~21.5 Гц)
DEFAULT_HOP    = 512       # шаг анализа по умолчанию (11.6 мс при 44.1 кГц)
DEFAULT_PEAKS  = 150       # топ-N доминирующих частот на кадр (150-200 = высокая точность)

NOISE_FLOOR_DB = 60.0      # динамический диапазон: пики тише (максимум кадра - 60 дБ) отбрасываются
ABS_FLOOR_DB   = -90.0     # абсолютный порог, дБ
AMP_CLIP       = 2.0       # страховочный лимит амплитуды одной синусоиды
MAKEUP_GAIN    = 1.0       # макияж громкости декодера (1.0 = без изменения)

PHASE_TRACKING = True      # продолжение фаз тонов между кадрами (качество ↑↑)
FREQ_TOL_BINS  = 0.75      # допуск сопоставления частичных тонов, в ширинах бина

BATCH_FRAMES   = 1024      # кадров за один проход (баланс скорости и памяти)


# ==============================================================================
#  1. ФОРМАТ .not — чтение и запись текста
# ==============================================================================

def write_not_header(f, src_name: str, sr: int, win: int, hop: int,
                     peaks: int, n_frames: int, duration: float) -> None:
    """Записывает текстовый заголовок .not-файла."""
    f.write(f"#NOT {FORMAT_VERSION}\n")
    f.write(f"#SRC {src_name}\n")
    f.write(f"#SR {sr}\n")
    f.write(f"#WIN {win}\n")
    f.write(f"#HOP {hop}\n")
    f.write(f"#PEAKS {peaks}\n")
    f.write(f"#FRAMES {n_frames}\n")
    f.write(f"#DUR {duration:.2f}\n")


def read_not_header(path: Path) -> tuple[dict, int]:
    """
    Читает заголовок .not и подсчитывает фактическое число кадров.
    Возвращает (hdr, n_frames).
    """
    hdr = {"sr": 44100, "win": ANALYSIS_WIN, "hop": DEFAULT_HOP,
           "peaks": DEFAULT_PEAKS, "frames": 0, "dur": 0.0, "src": ""}
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line.startswith("#NOT") and not line.split()[1].startswith("1"):
                    raise ValueError(f"Неподдерживаемая версия формата: {line}")
                parts = line[1:].split(None, 1)
                key = parts[0].upper()
                val = parts[1] if len(parts) > 1 else ""
                if key == "SR":     hdr["sr"] = int(val)
                elif key == "WIN":  hdr["win"] = int(val)
                elif key == "HOP":  hdr["hop"] = int(val)
                elif key == "PEAKS": hdr["peaks"] = int(val)
                elif key == "DUR":  hdr["dur"] = float(val)
                elif key == "SRC":  hdr["src"] = val
                continue
            n += 1  # строка данных
    hdr["frames"] = n
    return hdr, n


def parse_frame_line(body: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Разбирает тело строки '440.00:0.5321,880.13:0.2114' в массивы
    (частоты, амплитуды). Пустая строка -> пустые массивы (тишина).
    """
    if not body:
        return np.zeros(0), np.zeros(0)
    vals = np.array(body.replace(":", ",").split(","), dtype=np.float64)
    vals = vals.reshape(-1, 2)
    return vals[:, 0].copy(), vals[:, 1].copy()


# ==============================================================================
#  2. ЗАГРУЗКА И СОХРАНЕНИЕ АУДИО
# ==============================================================================

def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """
    Загружает аудиофайл (WAV/FLAC/OGG через soundfile; MP3 — через soundfile
    с libsndfile >= 1.1, иначе автоматический fallback на librosa).
    Возвращает (моно float32 в диапазоне [-1, 1], частота_дискретизации).
    """
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if data.shape[1] > 1:                 # стерео -> моно (среднее каналов)
            data = data.mean(axis=1, keepdims=True)
        return data[:, 0].astype(np.float32), sr
    except Exception:
        import librosa                        # запасной путь (умеет MP3 всегда)
        y, sr = librosa.load(str(path), sr=None, mono=True)
        return y.astype(np.float32), sr


def save_audio(dst: Path, y: np.ndarray, sr: int, log=print) -> Path:
    """
    Сохраняет аудио. WAV — напрямую (PCM 16 бит). Для .mp3 пытается
    использовать ffmpeg; если его нет — честно сохраняет WAV рядом.
    Возвращает фактический путь записанного файла.
    """
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    dst = Path(dst)
    if dst.suffix.lower() == ".mp3":
        tmp = dst.with_suffix(".tmp.wav")
        sf.write(str(tmp), y, sr, subtype="PCM_16")
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                            "-i", str(tmp), "-b:a", "320k", str(dst)], check=True)
            tmp.unlink(missing_ok=True)
            log(f"  MP3-экспорт через ffmpeg: {dst}")
            return dst
        except Exception:
            fallback = dst.with_suffix(".wav")
            tmp.replace(fallback)
            log(f"  ⚠ ffmpeg не найден — сохранено как WAV: {fallback}")
            return fallback
    sf.write(str(dst), y, sr, subtype="PCM_16")
    return dst


# ==============================================================================
#  3. КОДЕР: аудио -> текст .not
# ==============================================================================

def _analyze_block(block: np.ndarray, window: np.ndarray, sr: int,
                   n_peaks: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Ядро кодера. block: (F, WIN) — кадры сигнала.
    Для каждого кадра:
      1) rfft -> амплитудный спектр в дБ;
      2) маска локальных максимумов;
      3) топ-N пиков по громкости (argpartition — O(K) на кадр);
      4) параболическая интерполяция вершины по ИСХОДНОМУ спектру mag_db
         (важно: не по маскированному scores, иначе -inf даёт NaN!);
      5) перевод амплитуды спектра в амплитуду синусоиды: A = 2*|X|/sum(window).
    Возвращает (freqs, amps) формы (F, n_peaks); пустые слоты имеют amp=0.
    """
    F, W = block.shape
    K = W // 2 + 1
    n = max(1, min(n_peaks, K - 2))

    spec = np.fft.rfft(block * window, axis=1)
    mag = np.abs(spec)
    mag_db = 20.0 * np.log10(mag + 1e-12)

    # --- локальные максимумы + динамический порог -------------------------
    peak = np.zeros((F, K), dtype=bool)
    peak[:, 1:-1] = (mag_db[:, 1:-1] > mag_db[:, :-2]) & \
                    (mag_db[:, 1:-1] >= mag_db[:, 2:])
    floor = np.maximum(mag_db.max(axis=1, keepdims=True) - NOISE_FLOOR_DB,
                       ABS_FLOOR_DB)
    peak &= mag_db > floor
    peak[:, 0] = False                       # постоянная составляющая не поёт

    # --- топ-N пиков на кадр (отбор — по маскированному спектру) -----------
    scores = np.where(peak, mag_db, -np.inf)
    top = np.argpartition(-scores, n - 1, axis=1)[:, :n]

    rows = np.arange(F)[:, None]
    valid = np.isfinite(scores[rows, top])   # настоящий пик, а не пустой слот

    # --- соседи для интерполяции — из ИСХОДНОГО спектра (все значения конечны)
    beta  = mag_db[rows, top]
    alpha = mag_db[rows, np.clip(top - 1, 0, K - 1)]
    gamma = mag_db[rows, np.clip(top + 1, 0, K - 1)]

    with np.errstate(invalid="ignore", divide="ignore"):
        denom = alpha - 2.0 * beta + gamma
        p = 0.5 * (alpha - gamma) / np.where(np.abs(denom) < 1e-9, -1e-12, denom)
    p = np.clip(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), -0.5, 0.5)

    refined = np.where(valid, top + p, 0.0)
    peak_db = beta - 0.25 * (alpha - gamma) * p
    peak_db = np.where(valid, peak_db, -np.inf)
    peak_db = np.nan_to_num(peak_db, nan=-np.inf, neginf=-np.inf,
                            posinf=ABS_FLOOR_DB)

    freqs = refined * (sr / W)
    amps = np.where(valid,
                    (10.0 ** (peak_db / 20.0)) * (2.0 / float(window.sum())),
                    0.0)
    amps = np.clip(np.nan_to_num(amps, nan=0.0), 0.0, AMP_CLIP)

    # --- сортировка по частоте ----------------------------------------------
    order = np.argsort(freqs, axis=1)
    return (np.take_along_axis(freqs, order, axis=1),
            np.take_along_axis(amps, order, axis=1))


def encode_audio(y: np.ndarray, sr: int, out_path: Path, *,
                 hop: int = DEFAULT_HOP, n_peaks: int = DEFAULT_PEAKS,
                 src_name: str = "", progress_cb=None, log=print) -> dict:
    """
    Полный цикл кодирования: режет сигнал на кадры с шагом hop, анализирует
    пакетами и потоково пишет текстовый .not-файл. Возвращает статистику.
    """
    win = ANALYSIS_WIN
    if len(y) < win:                                   # слишком короткий файл
        y = np.pad(y, (0, win - len(y)))

    window = np.hanning(win).astype(np.float32)
    frames_view = np.lib.stride_tricks.sliding_window_view(y, win)[::hop]
    n_frames = frames_view.shape[0]

    total_partials = 0
    nonempty_lines = 0
    max_amp = 0.0
    sample_line = ""
    t0 = time.perf_counter()

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        write_not_header(f, src_name, sr, win, hop, n_peaks,
                         n_frames, len(y) / sr)

        for start in range(0, n_frames, BATCH_FRAMES):
            block = np.ascontiguousarray(
                frames_view[start:start + BATCH_FRAMES], dtype=np.float32)
            freqs, amps = _analyze_block(block, window, sr, n_peaks)

            # --- потоковая запись строк: <кадр>|<частота>:<амплитуда>,... --
            for i in range(freqs.shape[0]):
                gi = start + i
                keep = amps[i] > 5e-5
                if np.any(keep):
                    pairs = ",".join(
                        f"{fq:.2f}:{am:.4f}"
                        for fq, am in zip(freqs[i][keep], amps[i][keep]))
                    f.write(f"{gi}|{pairs}\n")
                    nonempty_lines += 1
                    total_partials += int(keep.sum())
                    row_max = float(amps[i].max())
                    if row_max > max_amp:
                        max_amp = row_max
                    if not sample_line:
                        sample_line = pairs[:110]
                else:
                    f.write(f"{gi}|\n")                  # кадр тишины

            if progress_cb:
                progress_cb(min(1.0, (start + freqs.shape[0]) / n_frames))

    log(f"  анализ: {time.perf_counter() - t0:.1f} c · {n_frames} кадров · "
        f"{total_partials} гармоник · ненулевых строк {nonempty_lines} "
        f"({100.0 * nonempty_lines / max(1, n_frames):.0f}%) · "
        f"макс. амплитуда {max_amp:.3f}")
    if sample_line:
        log(f"  пример строки: {sample_line}...")
    else:
        log("  ⚠ ВНИМАНИЕ: все строки пустые — на входе была тишина или ошибка анализа!")
    return {"frames": n_frames, "partials": total_partials,
            "nonempty": nonempty_lines, "max_amp": max_amp}


# ==============================================================================
#  4. СИНТЕЗАТОР: текст -> синусоиды (общий для декодера и плеера)
# ==============================================================================

def synth_batch(freqs: np.ndarray, amps: np.ndarray, phases: np.ndarray,
                sr: int, L: int) -> np.ndarray:
    """
    Векторизованный синтез сразу F кадров. На входе массивы (F, K).
    На выходе (F, L) — суммы синусоид:
        s[n] = Σ_k A_k · sin(2π·f_k·n/SR + φ_k),  n = 0..L-1
    """
    n = np.arange(L, dtype=np.float32)
    w = np.float32(2.0 * np.pi / sr)
    arg = np.einsum("fk,l->fkl", freqs.astype(np.float32) * w, n) \
        + phases.astype(np.float32)[:, :, None]
    return np.einsum("fk,fkl->fl", amps.astype(np.float32), np.sin(arg))


def track_phases(freqs: np.ndarray, prev_f: np.ndarray | None,
                 prev_ph: np.ndarray | None, hop: int, sr: int,
                 tol_hz: float) -> np.ndarray:
    """
    ПРОДОЛЖЕНИЕ ФАЗЫ устойчивых тонов. Для каждой частоты текущего кадра
    ищется ближайшая частота предыдущего кадра; если тон "тот же самый"
    (в пределах допуска), его фаза продолжается: φ += 2π·f_prev·hop/SR.
    Новые тоны стартуют с фазы 0. Это убирает фазовое дрожание и делает
    звук плотным, как у настоящего синтезатора.
    """
    if not PHASE_TRACKING or prev_f is None or len(prev_f) == 0:
        return np.zeros_like(freqs)
    pos = np.searchsorted(prev_f, freqs)
    pos_c = np.clip(pos, 0, len(prev_f) - 1)
    pos_l = np.clip(pos - 1, 0, len(prev_f) - 1)
    d_c = np.abs(prev_f[pos_c] - freqs)
    d_l = np.abs(prev_f[pos_l] - freqs)
    j = np.where(d_l < d_c, pos_l, pos_c)
    ok = np.minimum(d_l, d_c) <= tol_hz
    advance = 2.0 * np.pi * prev_f[j] * hop / sr
    return np.where(ok, prev_ph[j] + advance, 0.0)


def synth_window(L: int) -> np.ndarray:
    """
    Периодическое окно Хэннинга длиной L = 2*hop. Его ключевое свойство —
    COLA (Constant Overlap-Add): w[n] + w[n+hop] ≡ 1, поэтому соседние кадры
    сшиваются в единый сигнал без щелчков, швов и провалов громкости.
    """
    return np.hanning(L + 1)[:-1]


# ==============================================================================
#  5. ДЕКОДЕР: .not -> массив аудио -> файл на диске
# ==============================================================================

def decode_not_file(path: Path, hdr: dict, n_frames: int,
                    progress_cb=None, log=print) -> np.ndarray:
    """
    Читает .not построчно, синтезирует кадры пакетами (векторизованно),
    складывает их методом Overlap-Add и возвращает готовый аудиомассив.
    """
    sr, hop = hdr["sr"], hdr["hop"]
    L = 2 * hop
    w = synth_window(L)
    tol = FREQ_TOL_BINS * sr / hdr["win"]

    out = np.zeros(n_frames * hop + L, dtype=np.float64)
    prev_f = prev_ph = None
    batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    batch_start = 0

    def flush():
        """Синтезирует накопленный пакет кадров и добавляет его в out."""
        nonlocal prev_f, prev_ph, batch
        if not batch:
            return
        K = max(len(f) for f, _, _ in batch)
        F = np.zeros((len(batch), K)); A = np.zeros_like(F); PH = np.zeros_like(F)
        for i, (f, a, ph) in enumerate(batch):
            F[i, :len(f)] = f; A[i, :len(f)] = a; PH[i, :len(f)] = ph
        seg = synth_batch(F, A, PH, sr, L) * w[None, :]
        for i in range(len(batch)):
            o = batch_start + i * hop
            out[o:o + L] += seg[i]
        batch = []

    with open(path, "r", encoding="utf-8") as f:
        idx = 0
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            body = line.strip().split("|", 1)[1] if "|" in line else ""
            freqs, amps = parse_frame_line(body)
            phases = track_phases(freqs, prev_f, prev_ph, hop, sr, tol)
            prev_f, prev_ph = freqs, np.mod(phases, 2.0 * np.pi)

            if not batch:
                batch_start = idx * hop
            batch.append((freqs, amps, phases))
            if len(batch) >= BATCH_FRAMES:
                flush()
            idx += 1
            if progress_cb and idx % 512 == 0:
                progress_cb(min(1.0, idx / max(1, n_frames)))
        flush()

    if progress_cb:
        progress_cb(1.0)

    # --- диагностика: если RMS нулевой, проблема в данных .not --------------
    rms = float(np.sqrt(np.mean(out * out))) if out.size else 0.0
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    log(f"  синтез: RMS {rms:.4f} · пик {peak:.3f}")
    if rms < 1e-6:
        log("  ⚠ ВНИМАНИЕ: синтезированная тишина — файл .not пуст или повреждён.")

    # --- финальная нормировка: макияж + защита от клиппинга -----------------
    out *= MAKEUP_GAIN
    peak2 = float(np.max(np.abs(out))) if out.size else 0.0
    if peak2 > 0.99:
        out *= 0.99 / peak2
    return out


# ==============================================================================
#  6. ПЛЕЕР .NOT — синтез В ДИНАМИКИ в реальном времени (sounddevice)
# ==============================================================================

class NotPlayer:
    """
    Читает текстовый .not построчно, синтезирует синусоиды пакетами в фоновом
    потоке и отдаёт их в аудиопоток через callback sounddevice — без единого
    байта на диске. Кнопка "Стоп" мгновенно глушит поток.
    """

    CHUNK_FRAMES = 96      # кадров синтезируется за раз (~1 c звука)

    def __init__(self, path: Path):
        self.path = path
        self.hdr, self.n_frames = read_not_header(path)
        self.sr, self.hop = self.hdr["sr"], self.hdr["hop"]
        self.L = 2 * self.hop
        self._win = synth_window(self.L)
        self._tol = FREQ_TOL_BINS * self.sr / self.hdr["win"]

        self.volume = 0.9
        self.played_samples = 0
        self.finished = False
        self._level = 0.0                       # RMS текущего куска (для осциллографа)

        self._stop_evt = threading.Event()
        self._q: queue.Queue = queue.Queue(maxsize=8)
        self._cur = np.zeros(0, dtype=np.float32)
        self._pos = 0
        self._eof = False

        self._file = open(path, "r", encoding="utf-8")
        self._prev_f = self._prev_ph = None

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._stream = sd.OutputStream(samplerate=self.sr, channels=1,
                                       dtype="float32", blocksize=512,
                                       callback=self._callback)

    # ---------------- фоновый поток: чтение текста + синтез ----------------
    def _worker(self):
        tail = np.zeros(self.L, dtype=np.float64)   # "хвост" перекрытия от прошлого куска
        batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        def emit(buf: np.ndarray):
            """Добавляет кусок в очередь с тайм-аутом (чтобы Стоп сработал мгновенно)."""
            while not self._stop_evt.is_set():
                try:
                    self._q.put(buf.astype(np.float32), timeout=0.2)
                    return
                except queue.Full:
                    continue

        def flush_chunk():
            nonlocal tail
            if not batch:
                return
            K = max(len(f) for f, _, _ in batch)
            B = len(batch)
            F = np.zeros((B, K)); A = np.zeros_like(F); PH = np.zeros_like(F)
            for i, (f, a, ph) in enumerate(batch):
                F[i, :len(f)] = f; A[i, :len(f)] = a; PH[i, :len(f)] = ph
            seg = synth_batch(F, A, PH, self.sr, self.L) * self._win[None, :]
            buf = np.zeros(B * self.hop + self.L, dtype=np.float64)
            for i in range(B):
                buf[i * self.hop: i * self.hop + self.L] += seg[i]
            buf[:self.L] += tail                      # сшиваем с предыдущим куском (OLA)
            tail = buf[B * self.hop:].copy()
            emit((buf[:B * self.hop] * MAKEUP_GAIN))
            batch.clear()

        try:
            for line in self._file:
                if self._stop_evt.is_set():
                    return
                if line.startswith("#") or not line.strip():
                    continue
                body = line.strip().split("|", 1)[1] if "|" in line else ""
                freqs, amps = parse_frame_line(body)
                phases = track_phases(freqs, self._prev_f, self._prev_ph,
                                      self.hop, self.sr, self._tol)
                self._prev_f, self._prev_ph = freqs, np.mod(phases, 2.0 * np.pi)
                batch.append((freqs, amps, phases))
                if len(batch) >= self.CHUNK_FRAMES:
                    flush_chunk()
            flush_chunk()
            if np.abs(tail).max() > 1e-6:             # доиграть хвост последнего кадра
                emit(tail * MAKEUP_GAIN)
        finally:
            try:
                self._q.put(None, timeout=0.5)         # страж конца файла
            except queue.Full:
                pass

    # ---------------- callback аудиопотока (вызывает sounddevice) ----------
    def _callback(self, outdata, frames, time_info, status):
        out = outdata[:, 0]
        filled = 0
        while filled < frames:
            if self._pos >= len(self._cur):            # текущий кусок исчерпан
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    out[filled:] = 0.0                 # буфер пуст -> тишина (не треск!)
                    if self._eof:
                        self.finished = True
                    break
                if item is None:
                    self._eof = True
                    continue
                self._cur = item
                self._pos = 0
                self._level = float(np.sqrt(np.mean(item.astype(np.float64) ** 2)))
            n = min(frames - filled, len(self._cur) - self._pos)
            out[filled:filled + n] = self._cur[self._pos:self._pos + n] * self.volume
            filled += n
            self._pos += n
            self.played_samples += n

    # ---------------- управление -------------------------------------------
    def start(self):
        self._stream.start()
        self._thread.start()

    def stop(self):
        """Мгновенная остановка: флаг потоку, глушим и закрываем аудио."""
        self._stop_evt.set()
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        try:
            self._file.close()
        except Exception:
            pass
        self._thread.join(timeout=1.0)

    @property
    def position_seconds(self) -> float:
        return self.played_samples / self.sr

    @property
    def duration_seconds(self) -> float:
        return self.n_frames * self.hop / self.sr

    def get_level(self) -> float:
        return self._level


# ==============================================================================
#  7. ГРАФИЧЕСКИЙ ИНТЕРФЕЙС (customtkinter)
# ==============================================================================

# --- палитра "осциллографической лаборатории" --------------------------------
PAL = {
    "bg":       "#0a0e13",
    "panel":    "#111823",
    "panel2":   "#182231",
    "border":   "#243447",
    "text":     "#e8eef5",
    "muted":    "#8ba1b7",
    "amber":    "#ffb020",   # кодер
    "amber_d":  "#8a5a10",
    "cyan":     "#38bdf8",   # декодер
    "green":    "#4ade80",   # play
    "red":      "#f87171",   # stop
}


def _pick_family(candidates: list[str]) -> str:
    """Первый доступный в системе шрифт из списка."""
    try:
        fams = set(tkfont.families())
    except Exception:
        fams = set()
    for c in candidates:
        if c in fams:
            return c
    return candidates[-1]


DISPLAY_FAM = _pick_family(["Bahnschrift", "Avenir Next Condensed", "Arial Narrow", "Segoe UI"])
BODY_FAM    = _pick_family(["Segoe UI", "SF Pro Text", "Ubuntu", "DejaVu Sans"])
MONO_FAM    = _pick_family(["Cascadia Mono", "JetBrains Mono", "Consolas", "Menlo", "DejaVu Sans Mono"])

HOP_PRESETS = {
    "Сверхвысокое · hop 256 (≈5.8 мс/кадр)": 256,
    "Высокое · hop 512 (≈11.6 мс/кадр)": 512,
    "Экономное · hop 1024 (≈23.2 мс/кадр)": 1024,
}


class ScopeWidget:
    """
    Живой осциллограф в шапке: в покое дышит синусоидой, во время
    воспроизведения реагирует на реальную громкость синтезируемого сигнала.
    """

    def __init__(self, master, app):
        import tkinter as tk
        self.app = app
        self.w = 300
        self.h = 72
        self.canvas = tk.Canvas(master, width=self.w, height=self.h,
                                bg=PAL["panel2"], highlightthickness=1,
                                highlightbackground=PAL["border"])
        self.phase = 0.0
        self.amp = 0.15
        self._tick()

    def _tick(self):
        import math
        c = self.canvas
        c.delete("all")
        mid = self.h / 2
        # фоновая сетка
        c.create_line(0, mid, self.w, mid, fill="#1f2c3d")
        for gx in range(0, self.w, 30):
            c.create_line(gx, 8, gx, self.h - 8, fill="#16202e")

        target = self.app.scope_target_amp()
        self.amp += (target - self.amp) * 0.12          # плавная реакция на громкость
        self.phase += 0.22

        # плоский список координат: [x0, y0, x1, y1, ...] — надёжный формат для Tk
        coords = []
        for x in range(0, self.w + 1, 2):
            env = 0.55 + 0.45 * math.sin(math.pi * x / self.w)   # веретено по краям
            yv = mid - self.amp * (mid - 6) * env * math.sin(
                3.0 * 2 * math.pi * x / self.w + self.phase)
            coords.extend((x, yv))

        c.create_line(*coords, fill="#1e4a3a", width=5)   # "свечение"
        c.create_line(*coords, fill=PAL["green"], width=2)
        c.after(33, self._tick)


class NotCodecApp(ctk.CTk):
    """Главное окно: три раздела — кодер, декодер, плеер."""

    def __init__(self):
        super().__init__()

        self._events: queue.Queue = queue.Queue()   # мост поток -> GUI
        self._busy = {"enc": False, "dec": False}
        self.player: NotPlayer | None = None
        self._volume = 0.9
        self._polling_player = False

        # --- шрифты ---------------------------------------------------------
        self.F_TITLE = ctk.CTkFont(family=DISPLAY_FAM, size=30, weight="bold")
        self.F_SUB   = ctk.CTkFont(family=BODY_FAM, size=12)
        self.F_CHIP  = ctk.CTkFont(family=MONO_FAM, size=10)
        self.F_TAB   = ctk.CTkFont(family=DISPLAY_FAM, size=13, weight="bold")
        self.F_HEAD  = ctk.CTkFont(family=DISPLAY_FAM, size=14, weight="bold")
        self.F_BODY  = ctk.CTkFont(family=BODY_FAM, size=13)
        self.F_BTN   = ctk.CTkFont(family=DISPLAY_FAM, size=15, weight="bold")
        self.F_MONO  = ctk.CTkFont(family=MONO_FAM, size=11)
        self.F_NOTE  = ctk.CTkFont(family=BODY_FAM, size=12)

        self.title("NOT·CODEC — параметрический текстовый аудиокодек")
        self.geometry("1020x730")
        self.minsize(900, 620)
        self.configure(fg_color=PAL["bg"])

        self._build_ui()
        self._after_events()

    # =======================================================================
    #  Построение интерфейса
    # =======================================================================
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ---------- ШАПКА: название + живой осциллограф ---------------------
        header = ctk.CTkFrame(self, fg_color=PAL["panel"], corner_radius=0,
                              border_width=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        logo = ctk.CTkFrame(header, width=46, height=46, corner_radius=10,
                            fg_color=PAL["amber"])
        logo.grid(row=0, column=0, padx=(18, 12), pady=14)
        logo.grid_propagate(False)
        ctk.CTkLabel(logo, text="≋", font=ctk.CTkFont(family=BODY_FAM, size=26,
                     weight="bold"), text_color="#1a1206").place(relx=0.5, rely=0.5,
                                                                 anchor="center")

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.grid(row=0, column=1, sticky="w", pady=10)
        row1 = ctk.CTkFrame(title_box, fg_color="transparent")
        row1.pack(anchor="w")
        ctk.CTkLabel(row1, text="NOT", font=self.F_TITLE,
                     text_color=PAL["text"]).pack(side="left")
        ctk.CTkLabel(row1, text="·CODEC", font=self.F_TITLE,
                     text_color=PAL["amber"]).pack(side="left", padx=(2, 10))
        chip = ctk.CTkFrame(row1, fg_color=PAL["panel2"], corner_radius=6)
        chip.pack(side="left", pady=6)
        ctk.CTkLabel(chip, text=f" v{FORMAT_VERSION} · STFT · OLA ", font=self.F_CHIP,
                     text_color=PAL["muted"]).pack(padx=6, pady=3)
        ctk.CTkLabel(title_box,
                     text="Звук как математика: точная частота + громкость на каждый шаг времени — чистый текст, синтез на лету",
                     font=self.F_SUB, text_color=PAL["muted"]).pack(anchor="w")

        self.scope = ScopeWidget(header, self)
        self.scope.canvas.grid(row=0, column=2, padx=18, pady=14)

        # ---------- ВКЛАДКИ --------------------------------------------------
        self.tabs = ctk.CTkTabview(self, fg_color=PAL["panel"],
                                   segmented_button_font=self.F_TAB,
                                   segmented_button_fg_color=PAL["panel2"],
                                   segmented_button_selected_color=PAL["amber_d"],
                                   segmented_button_selected_hover_color=PAL["amber_d"],
                                   segmented_button_unselected_color=PAL["panel2"],
                                   segmented_button_unselected_hover_color=PAL["border"],
                                   text_color=PAL["text"], corner_radius=12,
                                   border_width=1, border_color=PAL["border"])
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=14, pady=(4, 8))
        for key, title in (("enc", "  ①  MP3/WAV → .NOT  "),
                           ("dec", "  ②  .NOT → WAV  "),
                           ("ply", "  ③  Плеер .NOT  ")):
            self.tabs.add(key)
            self.tabs.tab(key).configure(fg_color=PAL["panel"])
            self.tabs.tab(key).grid_columnconfigure(0, weight=1)

        self._build_tab_enc()
        self._build_tab_dec()
        self._build_tab_ply()

        # ---------- НИЖНЯЯ ПАНЕЛЬ СОСТОЯНИЯ ----------------------------------
        bar = ctk.CTkFrame(self, fg_color=PAL["panel"], corner_radius=0, height=34)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_columnconfigure(0, weight=1)
        self.status_left = ctk.CTkLabel(bar, text="● Готов к работе",
                                        font=self.F_MONO, text_color=PAL["green"])
        self.status_left.grid(row=0, column=0, sticky="w", padx=16, pady=8)
        ctk.CTkLabel(bar, text="STFT 2048 · overlap-add 50% · float32 · sounddevice",
                     font=self.F_MONO, text_color=PAL["muted"]).grid(
                     row=0, column=1, sticky="e", padx=16)

        # приветственные подсказки в логах
        self._log("enc", "Формат .not: каждая строка — шаг времени, пары ЧАСТОТА_Гц:ГРОМКОСТЬ.")
        self._log("enc", "Совет: топ-N = 150–200 и hop 256 дают максимальную точность.")
        self._log("dec", "Обратный синтез: текст → синусоиды → сшивка overlap-add → WAV.")
        self._log("ply", "Плеер синтезирует звук из текста прямо в динамики, без записи на диск.")

    # ---------------- общие виджеты -----------------------------------------
    def _file_row(self, parent, row: int, label: str, btn_text: str,
                  command, filetypes, editable: bool = False):
        ctk.CTkLabel(parent, text=label, font=self.F_BODY,
                     text_color=PAL["muted"]).grid(row=row, column=0, sticky="w", pady=4)
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=row, column=1, sticky="ew", pady=4)
        box.grid_columnconfigure(0, weight=1)
        entry = ctk.CTkEntry(box, font=self.F_MONO, fg_color=PAL["panel2"],
                             border_color=PAL["border"], text_color=PAL["text"],
                             height=34)
        entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        if not editable:
            entry.configure(state="readonly")
        ctk.CTkButton(box, text=btn_text, width=150, height=34, font=self.F_BODY,
                      fg_color=PAL["panel2"], hover_color=PAL["border"],
                      border_width=1, border_color=PAL["border"],
                      command=command).grid(row=0, column=1)
        return entry

    def _set_entry(self, entry, text: str):
        entry.configure(state="normal")
        entry.delete(0, "end")
        entry.insert(0, text)
        entry.configure(state="readonly")

    def _log_box(self, parent, row: int, key: str):
        tb = ctk.CTkTextbox(parent, font=self.F_MONO, fg_color="#0b111a",
                            border_width=1, border_color=PAL["border"],
                            text_color="#b9c8d8", height=120, state="disabled")
        tb.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=(10, 4))
        self.logs[key] = tb

    def _progress_row(self, parent, row: int, key: str, color: str):
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        box.grid_columnconfigure(0, weight=1)
        bar = ctk.CTkProgressBar(box, orientation="horizontal", mode="determinate",
                                 height=14, corner_radius=7, fg_color=PAL["panel2"],
                                 progress_color=color)
        bar.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        bar.set(0)
        pct = ctk.CTkLabel(box, text="  0.0%", font=self.F_MONO, width=64,
                           text_color=PAL["muted"])
        pct.grid(row=0, column=1)
        self.bars[key], self.pcts[key] = bar, pct

    # ---------------- вкладка 1: кодер ---------------------------------------
    def _build_tab_enc(self):
        self.bars, self.pcts, self.logs = {}, {}, {}
        tab = self.tabs.tab("enc")
        tab.grid_rowconfigure(6, weight=1)

        ctk.CTkLabel(tab, text="Параметрический анализ: STFT → топ-N доминирующих частот с интерполяцией до долей Гц",
                     font=self.F_NOTE, text_color=PAL["muted"], anchor="w").grid(
                     row=0, column=0, columnspan=2, sticky="ew", pady=(4, 8))

        self.enc_src = self._file_row(tab, 1, "Исходный трек:", "Выбрать MP3/WAV…",
                                      self._pick_enc_src,
                                      [("Аудио", "*.mp3 *.wav *.flac *.ogg *.m4a"),
                                       ("Все файлы", "*.*")])
        self.enc_dst = self._file_row(tab, 2, "Файл .not:", "Сохранить как…",
                                      self._pick_enc_dst,
                                      [("NOT-файл", "*.not")], editable=True)

        # настройки анализа
        cfg = ctk.CTkFrame(tab, fg_color=PAL["panel2"], corner_radius=10)
        cfg.grid(row=3, column=0, columnspan=2, sticky="ew", pady=8)
        ctk.CTkLabel(cfg, text="Топ-N гармоник:", font=self.F_BODY).pack(
            side="left", padx=(14, 6), pady=10)
        self.n_peaks_entry = ctk.CTkEntry(cfg, width=70, font=self.F_MONO,
                                          fg_color=PAL["panel"], border_color=PAL["border"])
        self.n_peaks_entry.insert(0, str(DEFAULT_PEAKS))
        self.n_peaks_entry.pack(side="left", padx=(0, 18))
        ctk.CTkLabel(cfg, text="Разрешение времени:", font=self.F_BODY).pack(
            side="left", padx=(0, 6))
        self.hop_var = ctk.StringVar(value="Высокое · hop 512 (≈11.6 мс/кадр)")
        ctk.CTkOptionMenu(cfg, values=list(HOP_PRESETS), variable=self.hop_var,
                          width=320, font=self.F_BODY, fg_color=PAL["panel"],
                          button_color=PAL["amber_d"],
                          button_hover_color=PAL["amber"]).pack(side="left", padx=(0, 14), pady=10)

        self.btn_enc = ctk.CTkButton(
            tab, text="⚙  КОНВЕРТИРОВАТЬ В .NOT", height=46, font=self.F_BTN,
            fg_color=PAL["amber"], hover_color="#c98a12", text_color="#1a1206",
            corner_radius=10, command=self._start_encode)
        self.btn_enc.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        self._progress_row(tab, 5, "enc", PAL["amber"])
        self._log_box(tab, 6, "enc")

    # ---------------- вкладка 2: декодер --------------------------------------
    def _build_tab_dec(self):
        tab = self.tabs.tab("dec")
        tab.grid_rowconfigure(5, weight=1)

        ctk.CTkLabel(tab, text="Обратный синтез: построчное чтение текста → генерация синусоид → бесшовная сшивка overlap-add",
                     font=self.F_NOTE, text_color=PAL["muted"], anchor="w").grid(
                     row=0, column=0, columnspan=2, sticky="ew", pady=(4, 8))

        self.dec_src = self._file_row(tab, 1, "Файл .not:", "Выбрать .not…",
                                      self._pick_dec_src, [("NOT-файл", "*.not")])
        self.dec_dst = self._file_row(tab, 2, "Куда сохранить:", "Сохранить как…",
                                      self._pick_dec_dst,
                                      [("WAV", "*.wav"), ("MP3 (нужен ffmpeg)", "*.mp3")],
                                      editable=True)

        self.btn_dec = ctk.CTkButton(
            tab, text="◈  КОНВЕРТИРОВАТЬ В АУДИО", height=46, font=self.F_BTN,
            fg_color=PAL["cyan"], hover_color="#1e93c9", text_color="#04202e",
            corner_radius=10, command=self._start_decode)
        self.btn_dec.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        self._progress_row(tab, 4, "dec", PAL["cyan"])
        self._log_box(tab, 5, "dec")

    # ---------------- вкладка 3: плеер ----------------------------------------
    def _build_tab_ply(self):
        tab = self.tabs.tab("ply")
        tab.grid_rowconfigure(6, weight=1)

        ctk.CTkLabel(tab, text="Синтез в реальном времени: плеер читает текст и генерирует синусоиды прямо в аудиопоток",
                     font=self.F_NOTE, text_color=PAL["muted"], anchor="w").grid(
                     row=0, column=0, columnspan=2, sticky="ew", pady=(4, 8))

        self.ply_src = self._file_row(tab, 1, "Файл .not:", "Выбрать .not…",
                                      self._pick_ply_src, [("NOT-файл", "*.not")])

        transport = ctk.CTkFrame(tab, fg_color="transparent")
        transport.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        transport.grid_columnconfigure(2, weight=1)
        self.btn_play = ctk.CTkButton(
            transport, text="▶  ВОСПРОИЗВЕСТИ", width=230, height=48, font=self.F_BTN,
            fg_color=PAL["green"], hover_color="#2fbf68", text_color="#04220f",
            corner_radius=10, state="disabled", command=self._start_play)
        self.btn_play.grid(row=0, column=0, padx=(0, 10))
        self.btn_stop = ctk.CTkButton(
            transport, text="■  СТОП", width=140, height=48, font=self.F_BTN,
            fg_color=PAL["red"], hover_color="#d14b4b", text_color="#2a0606",
            corner_radius=10, state="disabled", command=self._stop_play)
        self.btn_stop.grid(row=0, column=1)

        # позиция воспроизведения
        pos = ctk.CTkFrame(tab, fg_color="transparent")
        pos.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        pos.grid_columnconfigure(0, weight=1)
        self.ply_bar = ctk.CTkProgressBar(pos, height=12, corner_radius=6,
                                          fg_color=PAL["panel2"],
                                          progress_color=PAL["green"])
        self.ply_bar.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        self.ply_time = ctk.CTkLabel(pos, text="00:00.0 / 00:00.0", font=self.F_MONO,
                                     text_color=PAL["text"])
        self.ply_time.grid(row=0, column=1)

        # громкость
        vol = ctk.CTkFrame(tab, fg_color=PAL["panel2"], corner_radius=10)
        vol.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        vol.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(vol, text="Громкость", font=self.F_BODY).grid(
            row=0, column=0, padx=(14, 10), pady=12)
        self.vol_slider = ctk.CTkSlider(vol, from_=0, to=1, number_of_steps=100,
                                        button_color=PAL["green"],
                                        button_hover_color="#2fbf68",
                                        progress_color=PAL["green"],
                                        fg_color=PAL["panel"],
                                        command=self._on_volume)
        self.vol_slider.set(self._volume)
        self.vol_slider.grid(row=0, column=1, sticky="ew", padx=4)
        self.vol_label = ctk.CTkLabel(vol, text="90%", font=self.F_MONO, width=44,
                                      text_color=PAL["muted"])
        self.vol_label.grid(row=0, column=2, padx=(6, 14))

        self._log_box(tab, 6, "ply")

    # =======================================================================
    #  Мост "рабочий поток -> GUI" (tkinter не потокобезопасен)
    # =======================================================================
    def _emit(self, kind: str, *args):
        self._events.put((kind, args))

    def _after_events(self):
        try:
            while True:
                kind, args = self._events.get_nowait()
                if kind == "log":
                    self._log(*args)
                elif kind == "progress":
                    self._set_progress(*args)
                elif kind == "done":
                    cb, result = args
                    cb(result)
                elif kind == "error":
                    tab, exc = args
                    self._on_task_error(tab, exc)
        except queue.Empty:
            pass
        self.after(60, self._after_events)

    def _log(self, key: str, msg: str):
        tb = self.logs[key]
        tb.configure(state="normal")
        tb.insert("end", msg + "\n")
        tb.see("end")
        tb.configure(state="disabled")

    def _set_progress(self, key: str, v: float):
        self.bars[key].set(v)
        self.pcts[key].configure(text=f"{v * 100:5.1f}%")

    def _set_status(self, text: str, color: str = PAL["green"]):
        self.status_left.configure(text=text, text_color=color)

    def _run_in_thread(self, fn):
        threading.Thread(target=fn, daemon=True).start()

    def _on_task_error(self, tab: str, exc: Exception):
        self._busy[tab] = False
        self._refresh_busy()
        self._log(tab, f"✖ ОШИБКА: {exc}")
        self._set_status("● Ошибка выполнения", PAL["red"])
        messagebox.showerror("NOT·CODEC", f"Ошибка:\n\n{exc}")

    def _refresh_busy(self):
        enc_busy, dec_busy = self._busy["enc"], self._busy["dec"]
        self.btn_enc.configure(state="disabled" if enc_busy else "normal",
                               text="КОНВЕРТИРАЦИЯ…" if enc_busy else "⚙  КОНВЕРТИРОВАТЬ В .NOT")
        self.btn_dec.configure(state="disabled" if dec_busy else "normal",
                               text="СИНТЕЗ…" if dec_busy else "◈  КОНВЕРТИРОВАТЬ В АУДИО")

    # =======================================================================
    #  Выбор файлов
    # =======================================================================
    def _pick_enc_src(self):
        p = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[("Аудио", "*.mp3 *.wav *.flac *.ogg *.m4a"), ("Все файлы", "*.*")])
        if p:
            self._set_entry(self.enc_src, p)
            self.enc_dst.delete(0, "end")
            self.enc_dst.insert(0, str(Path(p).with_suffix(".not")))

    def _pick_enc_dst(self):
        p = filedialog.asksaveasfilename(title="Сохранить .not", defaultextension=".not",
                                         filetypes=[("NOT-файл", "*.not")])
        if p:
            self.enc_dst.delete(0, "end")
            self.enc_dst.insert(0, p)

    def _pick_dec_src(self):
        p = filedialog.askopenfilename(title="Выберите .not",
                                       filetypes=[("NOT-файл", "*.not")])
        if p:
            self._set_entry(self.dec_src, p)
            self.dec_dst.delete(0, "end")
            self.dec_dst.insert(0, str(Path(p).with_suffix(".wav")))

    def _pick_dec_dst(self):
        p = filedialog.asksaveasfilename(title="Сохранить аудио", defaultextension=".wav",
                                         filetypes=[("WAV", "*.wav"), ("MP3", "*.mp3")])
        if p:
            self.dec_dst.delete(0, "end")
            self.dec_dst.insert(0, p)

    def _pick_ply_src(self):
        p = filedialog.askopenfilename(title="Выберите .not",
                                       filetypes=[("NOT-файл", "*.not")])
        if p:
            self._set_entry(self.ply_src, p)
            self.btn_play.configure(state="normal")
            self._log("ply", f"Файл загружен: {Path(p).name}. Нажмите ▶ ВОСПРОИЗВЕСТИ.")

    # =======================================================================
    #  КОДИРОВАНИЕ
    # =======================================================================
    def _start_encode(self):
        src = Path(self.enc_src.get().strip())
        dst = Path(self.enc_dst.get().strip() or src.with_suffix(".not"))
        try:
            n_peaks = int(self.n_peaks_entry.get())
            assert 8 <= n_peaks <= 400
        except Exception:
            messagebox.showwarning("NOT·CODEC", "Топ-N должен быть целым числом 8…400.")
            return
        hop = HOP_PRESETS[self.hop_var.get()]
        if not src.is_file():
            messagebox.showwarning("NOT·CODEC", "Сначала выберите исходный аудиофайл.")
            return

        self._busy["enc"] = True
        self._refresh_busy()
        self.bars["enc"].set(0)
        self._set_status("● Кодирование: STFT-анализ…", PAL["amber"])
        self._log("enc", f"— Задание: {src.name} → {dst.name} (N={n_peaks}, hop={hop})")

        def job():
            try:
                emit = self._emit
                emit("log", "enc", f"Загружаю аудио: {src.name}")
                y, sr = load_audio(src)
                emit("log", "enc", f"  {sr} Гц · {len(y) / sr:.2f} c · моно")
                emit("log", "enc", f"  окно STFT {ANALYSIS_WIN} · шаг {hop} · "
                                   f"синтез-окно {2 * hop} (overlap-add)")
                stats = encode_audio(
                    y, sr, dst, hop=hop, n_peaks=n_peaks, src_name=src.name,
                    progress_cb=lambda f: emit("progress", "enc", f),
                    log=lambda m: emit("log", "enc", m))
                size_mb = dst.stat().st_size / 1048576
                emit("log", "enc", f"✔ Сохранено: {dst} ({size_mb:.1f} МБ)")
                emit("log", "enc", f"  кадров: {stats['frames']} · "
                                   f"гармоник: {stats['partials']} "
                                   f"(ø {stats['partials'] / max(1, stats['frames']):.0f}/кадр)")
                emit("done", self._on_encode_done, None)
            except Exception as e:
                self._emit("error", "enc", e)

        self._run_in_thread(job)

    def _on_encode_done(self, _):
        self._busy["enc"] = False
        self._refresh_busy()
        self._set_status("● Кодирование завершено", PAL["green"])

    # =======================================================================
    #  ДЕКОДИРОВАНИЕ
    # =======================================================================
    def _start_decode(self):
        src = Path(self.dec_src.get().strip())
        dst = Path(self.dec_dst.get().strip() or src.with_suffix(".wav"))
        if not src.is_file():
            messagebox.showwarning("NOT·CODEC", "Сначала выберите файл .not.")
            return

        self._busy["dec"] = True
        self._refresh_busy()
        self.bars["dec"].set(0)
        self._set_status("● Декодирование: синтез из текста…", PAL["cyan"])
        self._log("dec", f"— Задание: {src.name} → {dst.name}")

        def job():
            try:
                emit = self._emit
                t0 = time.perf_counter()
                emit("log", "dec", f"Читаю заголовок: {src.name}")
                hdr, n = read_not_header(src)
                emit("log", "dec", f"  {hdr['sr']} Гц · hop {hdr['hop']} · "
                                   f"кадров {n} · ≈{n * hdr['hop'] / hdr['sr']:.1f} c")
                y = decode_not_file(src, hdr, n,
                                    progress_cb=lambda f: emit("progress", "dec", f),
                                    log=lambda m: emit("log", "dec", m))
                emit("log", "dec", "Синтез завершён, сохраняю на диск…")
                actual = save_audio(dst, y, hdr["sr"],
                                    log=lambda m: emit("log", "dec", m))
                emit("log", "dec", f"✔ Готово за {time.perf_counter() - t0:.1f} c → "
                                   f"{actual} ({actual.stat().st_size / 1048576:.1f} МБ)")
                emit("done", self._on_decode_done, None)
            except Exception as e:
                self._emit("error", "dec", e)

        self._run_in_thread(job)

    def _on_decode_done(self, _):
        self._busy["dec"] = False
        self._refresh_busy()
        self._set_status("● Декодирование завершено", PAL["green"])

    # =======================================================================
    #  ПЛЕЕР
    # =======================================================================
    def _on_volume(self, v):
        self._volume = float(v)
        self.vol_label.configure(text=f"{int(self._volume * 100)}%")
        if self.player is not None:
            self.player.volume = self._volume

    def _start_play(self):
        path = Path(self.ply_src.get().strip())
        if not path.is_file():
            return
        if self.player is not None:          # перезапуск — сначала глушим старый
            self._stop_play()
        try:
            self.player = NotPlayer(path)
            self.player.volume = self._volume
            self.player.start()
        except Exception as e:
            self.player = None
            messagebox.showerror("NOT·CODEC", f"Не удалось открыть аудиопоток:\n\n{e}")
            self._log("ply", f"✖ {e}")
            return

        self.btn_play.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self._set_status("● Воспроизведение: синтез из текста…", PAL["green"])
        self._log("ply", f"▶ Играю: {path.name} · {self.player.hdr['sr']} Гц · "
                         f"кадров {self.player.n_frames}")
        if not self._polling_player:
            self._polling_player = True
            self._poll_player()

    def _poll_player(self):
        p = self.player
        if p is None:
            self._polling_player = False
            return
        dur = max(1e-9, p.duration_seconds)
        pos = min(p.position_seconds, dur)
        self.ply_bar.set(pos / dur)
        self.ply_time.configure(text=f"{self._fmt_t(pos)} / {self._fmt_t(dur)}")
        if p.finished:
            self._log("ply", "✔ Воспроизведение завершено.")
            self._stop_play(reset_pos=True)
            self._polling_player = False   # <-- добавить: иначе второй Play не обновляет UI
            return
        self.after(100, self._poll_player)

    def _stop_play(self, reset_pos: bool = False):
        if self.player is not None:
            self.player.stop()
            self.player = None
        self.btn_play.configure(state="normal" if self.ply_src.get().strip() else "disabled")
        self.btn_stop.configure(state="disabled")
        if reset_pos:
            self.ply_bar.set(0)
        else:
            self._set_status("● Остановлено", PAL["muted"])

    @staticmethod
    def _fmt_t(s: float) -> str:
        m = int(s // 60)
        return f"{m:02d}:{s - 60 * m:04.1f}"

    def scope_target_amp(self) -> float:
        """Амплитуда для осциллографа в шапке: дышит в покое, живёт от плеера."""
        if self.player is not None and not self.player.finished:
            return min(0.95, 0.18 + self.player.get_level() * 3.0)
        return 0.16 + 0.06 * np.sin(time.time() * 1.7)


# ==============================================================================
#  8. ТОЧКА ВХОДА
# ==============================================================================

def main():
    missing = []
    if sf is None:  missing.append("soundfile")
    if sd is None:  missing.append("sounddevice")
    if ctk is None: missing.append("customtkinter")
    if missing:
        print("=" * 60)
        print(" NOT·CODEC: не хватает зависимостей:")
        print("   " + ", ".join(missing))
        print(" Установите:  pip install numpy soundfile sounddevice customtkinter")
        print("=" * 60)
        return

    app = NotCodecApp()
    ctk.set_appearance_mode("dark")
    app.mainloop()


if __name__ == "__main__":
    main()