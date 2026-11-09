import json
import random
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
import miniaudio
import numpy as np
import pygame

# Initialize hidden Tkinter root for native Windows/Linux file dialogs
root_tk = tk.Tk()
root_tk.withdraw()

# ----------------- Visual & UI Dimensions -----------------
WINDOW_WIDTH = 580
WINDOW_HEIGHT = 490
FPS = 60

NUM_BANDS = 16
FFT_SIZE = 1024
PEAK_FALL_SPEED = 1.3
DECAY_RATE = 0.84

# Winamp Color Palette
COLOR_BG = (12, 13, 17)
COLOR_PANEL = (22, 24, 30)
COLOR_PANEL_BORDER = (45, 50, 62)
COLOR_TEXT_DIM = (120, 135, 150)
COLOR_TEXT_BRIGHT = (0, 255, 170)
COLOR_SEEK_BG = (35, 38, 48)
COLOR_SEEK_FILL = (0, 200, 140)
COLOR_SEEK_HANDLE = (230, 240, 255)
COLOR_GRID = (24, 28, 36)
COLOR_OSC = (80, 230, 255)
COLOR_PEAK = (235, 245, 255)

SEGMENT_COLORS = [
    (0, 220, 60),
    (40, 205, 50),
    (140, 235, 45),
    (240, 215, 0),
    (255, 140, 0),
    (255, 60, 0),
]

# ----------------- Audio Engine -----------------
class MP3AudioEngine:
    def __init__(self):
        self.device = None
        self.stream = None
        self.samples = np.zeros((0, 2), dtype=np.float32)
        self.sample_rate = 44100
        self.nchannels = 2
        self.total_samples = 0
        self.cursor = 0
        self.playing = False
        self.is_loaded = False
        self.fft_buffer = np.zeros(FFT_SIZE, dtype=np.float32)

    def load_file(self, filepath: str):
        path = Path(filepath).expanduser().resolve()
        if not path.is_file():
            return False

        self.stop()
        if self.device:
            self.device.close()

        decoded = miniaudio.mp3_read_file_f32(str(path))
        self.sample_rate = decoded.sample_rate
        self.nchannels = decoded.nchannels
        self.samples = np.array(decoded.samples, dtype=np.float32).reshape(-1, self.nchannels)
        self.total_samples = len(self.samples)
        self.cursor = 0
        self.is_loaded = True

        self.device = miniaudio.PlaybackDevice(
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=self.nchannels,
            sample_rate=self.sample_rate
        )
        self.stream = self._audio_generator()
        next(self.stream)
        self.device.start(self.stream)
        self.playing = True
        return True

    def _audio_generator(self):
        num_frames = yield b""
        while True:
            if not self.playing or not self.is_loaded or self.cursor >= self.total_samples:
                chunk = np.zeros((num_frames, self.nchannels), dtype=np.float32)
            else:
                end_frame = min(self.cursor + num_frames, self.total_samples)
                frames_to_read = end_frame - self.cursor
                chunk = self.samples[self.cursor:end_frame]
                self.cursor = end_frame

                mono = np.mean(chunk, axis=1) if self.nchannels > 1 else chunk[:, 0]
                if len(mono) >= FFT_SIZE:
                    self.fft_buffer = mono[-FFT_SIZE:].copy()
                else:
                    self.fft_buffer = np.roll(self.fft_buffer, -len(mono))
                    self.fft_buffer[-len(mono):] = mono

                if frames_to_read < num_frames:
                    padding = np.zeros((num_frames - frames_to_read, self.nchannels), dtype=np.float32)
                    chunk = np.vstack((chunk, padding))

            num_frames = yield chunk.tobytes()

    def toggle_play(self):
        if self.is_loaded:
            self.playing = not self.playing

    def stop(self):
        self.playing = False
        self.cursor = 0
        self.fft_buffer.fill(0)

    def seek_fraction(self, frac: float):
        if self.is_loaded:
            self.cursor = int(np.clip(frac, 0.0, 1.0) * self.total_samples)

    def seek_relative(self, seconds: float):
        if self.is_loaded:
            delta_samples = int(seconds * self.sample_rate)
            self.cursor = int(np.clip(self.cursor + delta_samples, 0, self.total_samples))

    def get_progress(self):
        if not self.is_loaded or self.total_samples == 0:
            return 0.0, 0, 0
        elapsed = self.cursor / self.sample_rate
        total = self.total_samples / self.sample_rate
        return self.cursor / self.total_samples, int(elapsed), int(total)

    def is_finished(self):
        return self.is_loaded and self.cursor >= self.total_samples

    def close(self):
        if self.device:
            self.device.close()


# ----------------- Visualizer -----------------
class WinampVisualizer:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self.num_bands = NUM_BANDS
        self.band_heights = np.zeros(self.num_bands, dtype=np.float32)
        self.peak_caps = np.zeros(self.num_bands, dtype=np.float32)
        self.mode = 0  # 0: Spectrum, 1: Oscilloscope, 2: Combined
        self.mode_names = ["SPECTRUM", "OSCILLOSCOPE", "COMBINED"]

        self.window = np.hanning(FFT_SIZE)
        freq_bins = np.fft.rfftfreq(FFT_SIZE, d=1.0 / 44100)
        log_edges = np.geomspace(40.0, 16000.0, num=self.num_bands + 1)
        self.band_indices = [
            (int(np.searchsorted(freq_bins, log_edges[i])), int(np.searchsorted(freq_bins, log_edges[i + 1])))
            for i in range(self.num_bands)
        ]

    def cycle_mode(self):
        self.mode = (self.mode + 1) % 3

    def process_fft(self, pcm_samples):
        windowed = pcm_samples * self.window
        fft_data = np.abs(np.fft.rfft(windowed)) / (FFT_SIZE / 2)

        current_energy = np.zeros(self.num_bands, dtype=np.float32)
        for i, (start, end) in enumerate(self.band_indices):
            start = max(0, start)
            end = max(start + 1, end)
            hf_boost = 1.0 + (i / self.num_bands) * 2.2
            current_energy[i] = np.mean(fft_data[start:end]) * hf_boost

        max_h = self.rect.height - 20
        target = np.clip(current_energy * 270.0, 0, max_h)
        self.band_heights = np.maximum(target, self.band_heights * DECAY_RATE)

        for i in range(self.num_bands):
            if self.band_heights[i] >= self.peak_caps[i]:
                self.peak_caps[i] = self.band_heights[i]
            else:
                self.peak_caps[i] = max(0.0, self.peak_caps[i] - PEAK_FALL_SPEED)

    def draw(self, surface, pcm_samples):
        pygame.draw.rect(surface, (8, 9, 12), self.rect)
        pygame.draw.rect(surface, COLOR_PANEL_BORDER, self.rect, 1)

        # Subtle oscilloscope grid lines
        for y in range(self.rect.top + 15, self.rect.bottom, 16):
            pygame.draw.line(surface, COLOR_GRID, (self.rect.left + 5, y), (self.rect.right - 5, y), 1)

        if self.mode in (0, 2):  # Spectrum bars
            bar_w = int((self.rect.width - 20) / self.num_bands) - 4
            base_y = self.rect.bottom - 8
            for i in range(self.num_bands):
                bx = self.rect.left + 10 + i * (bar_w + 4)
                bh = int(self.band_heights[i])
                seg_h = 3
                num_segs = bh // (seg_h + 1)

                for s in range(num_segs):
                    sy = base_y - (s + 1) * (seg_h + 1)
                    c_idx = min(s // 4, len(SEGMENT_COLORS) - 1)
                    pygame.draw.rect(surface, SEGMENT_COLORS[c_idx], (bx, sy, bar_w, seg_h))

                if self.peak_caps[i] > 1:
                    py = base_y - int(self.peak_caps[i]) - seg_h
                    pygame.draw.rect(surface, COLOR_PEAK, (bx, py, bar_w, seg_h))

        if self.mode in (1, 2):  # Oscilloscope waveform
            points = []
            mid_y = self.rect.centery
            step = max(1, FFT_SIZE // self.rect.width)
            sub = pcm_samples[::step]
            for idx, val in enumerate(sub):
                px = self.rect.left + int((idx / len(sub)) * self.rect.width)
                py = mid_y - int(val * (self.rect.height * 0.4))
                points.append((px, np.clip(py, self.rect.top + 2, self.rect.bottom - 2)))
            if len(points) > 1:
                pygame.draw.lines(surface, COLOR_OSC, False, points, 1)


# ----------------- UI Button & List Component -----------------
class Button:
    def __init__(self, rect, text, callback, toggle_state=None):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.toggle_state = toggle_state

    def draw(self, surface, font):
        active = self.toggle_state() if callable(self.toggle_state) else False
        bg = (0, 90, 70) if active else (28, 32, 42)
        border = COLOR_TEXT_BRIGHT if active else COLOR_PANEL_BORDER
        text_col = COLOR_TEXT_BRIGHT if active else (200, 210, 220)

        pygame.draw.rect(surface, bg, self.rect, border_radius=3)
        pygame.draw.rect(surface, border, self.rect, 1, border_radius=3)
        txt = font.render(self.text, True, text_col)
        surface.blit(txt, txt.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()
                return True
        return False


# ----------------- Playlist Manager -----------------
class PlaylistManager:
    def __init__(self):
        self.tracks = []        # List of file path strings
        self.current_idx = 0
        self.shuffle = False

    def add_files(self, paths):
        for p in paths:
            if p not in self.tracks:
                self.tracks.append(p)

    def current_track(self):
        if 0 <= self.current_idx < len(self.tracks):
            return self.tracks[self.current_idx]
        return None

    def next_index(self):
        if not self.tracks:
            return 0
        if self.shuffle:
            return random.randint(0, len(self.tracks) - 1)
        return (self.current_idx + 1) % len(self.tracks)

    def prev_index(self):
        if not self.tracks:
            return 0
        if self.shuffle:
            return random.randint(0, len(self.tracks) - 1)
        return (self.current_idx - 1 + len(self.tracks)) % len(self.tracks)

    def save_to_file(self, json_path):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"tracks": self.tracks}, f, indent=2)

    def load_from_file(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.tracks = [t for t in data.get("tracks", []) if Path(t).is_file()]
            self.current_idx = 0


# ----------------- Main Application -----------------
def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Winamp Classic Multi-Player")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("monospace", 12, bold=True)
    font_large = pygame.font.SysFont("monospace", 15, bold=True)

    audio = MP3AudioEngine()
    playlist = PlaylistManager()
    vis = WinampVisualizer((20, 48, WINDOW_WIDTH - 40, 110))

    # Load initial song if passed in CLI
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        playlist.add_files([sys.argv[1]])
        audio.load_file(playlist.current_track())

    # Transport Callbacks
    def on_play_pause():
        if not audio.is_loaded and playlist.tracks:
            play_track(playlist.current_idx)
        else:
            audio.toggle_play()

    def on_stop():
        audio.stop()

    def play_track(index):
        if 0 <= index < len(playlist.tracks):
            playlist.current_idx = index
            audio.load_file(playlist.tracks[index])

    def on_next():
        if playlist.tracks:
            play_track(playlist.next_index())

    def on_prev():
        if playlist.tracks:
            play_track(playlist.prev_index())

    def on_rewind():
        audio.seek_relative(-5.0)

    def on_forward():
        audio.seek_relative(5.0)

    def toggle_shuffle():
        playlist.shuffle = not playlist.shuffle

    def add_files_dialog():
        files = filedialog.askopenfilenames(
            title="Select MP3 Files",
            filetypes=[("MP3 Files", "*.mp3"), ("All Files", "*.*")]
        )
        if files:
            first_add = len(playlist.tracks) == 0
            playlist.add_files(files)
            if first_add:
                play_track(0)

    def save_playlist_dialog():
        f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Playlist JSON", "*.json")])
        if f:
            playlist.save_to_file(f)

    def load_playlist_dialog():
        f = filedialog.askopenfilename(filetypes=[("Playlist JSON", "*.json")])
        if f:
            playlist.load_from_file(f)
            if playlist.tracks:
                play_track(0)

    # Control Buttons
    buttons = [
        Button((20, 204, 38, 24), "|<", on_prev),
        Button((62, 204, 38, 24), "<<", on_rewind),
        Button((104, 204, 46, 24), "PLAY", on_play_pause, lambda: audio.playing),
        Button((154, 204, 44, 24), "STOP", on_stop),
        Button((202, 204, 38, 24), ">>", on_forward),
        Button((244, 204, 38, 24), ">|", on_next),
        Button((290, 204, 52, 24), "SHUF", toggle_shuffle, lambda: playlist.shuffle),
        Button((346, 204, 62, 24), "MODE", vis.cycle_mode),
        Button((416, 204, 46, 24), "+ADD", add_files_dialog),
        Button((466, 204, 44, 24), "SAVE", save_playlist_dialog),
        Button((514, 204, 46, 24), "LOAD", load_playlist_dialog),
    ]

    # Geometry for Seek Bar and Playlist Box
    seek_bar_rect = pygame.Rect(20, 172, WINDOW_WIDTH - 40, 14)
    playlist_rect = pygame.Rect(20, 244, WINDOW_WIDTH - 40, 226)
    dragging_seek = False
    scroll_offset = 0

    running = True
    while running:
        # Autoplay Next Track
        if audio.is_finished() and playlist.tracks:
            on_next()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    on_play_pause()
                elif event.key == pygame.K_v:
                    vis.cycle_mode()
                elif event.key == pygame.K_s:
                    toggle_shuffle()
                elif event.key == pygame.K_LEFT:
                    on_rewind()
                elif event.key == pygame.K_RIGHT:
                    on_forward()
                elif event.key == pygame.K_b:
                    on_next()
                elif event.key == pygame.K_z:
                    on_prev()
                elif event.key == pygame.K_o:
                    add_files_dialog()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Seek Bar Click
                    if seek_bar_rect.collidepoint(event.pos):
                        dragging_seek = True
                        frac = (event.pos[0] - seek_bar_rect.x) / seek_bar_rect.width
                        audio.seek_fraction(frac)
                    # Playlist Item Click
                    elif playlist_rect.collidepoint(event.pos):
                        rel_y = event.pos[1] - playlist_rect.y
                        clicked_slot = scroll_offset + rel_y // 22
                        if 0 <= clicked_slot < len(playlist.tracks):
                            play_track(clicked_slot)

                # Mouse Wheel Scroll for Playlist
                elif event.button == 4:  # Scroll Up
                    scroll_offset = max(0, scroll_offset - 1)
                elif event.button == 5:  # Scroll Down
                    max_scroll = max(0, len(playlist.tracks) - (playlist_rect.height // 22))
                    scroll_offset = min(max_scroll, scroll_offset + 1)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging_seek = False

            elif event.type == pygame.MOUSEMOTION and dragging_seek:
                frac = (event.pos[0] - seek_bar_rect.x) / seek_bar_rect.width
                audio.seek_fraction(frac)

            for btn in buttons:
                btn.handle_event(event)

        # ----------------- Visual Processing & Drawing -----------------
        vis.process_fft(audio.fft_buffer)
        screen.fill(COLOR_BG)

        # Header Info Bar
        curr = playlist.current_track()
        title = Path(curr).stem if curr else "No Track Loaded - Click [+ADD] or Drop File"
        title_surf = font_large.render(title[:45], True, COLOR_TEXT_BRIGHT)
        screen.blit(title_surf, (20, 16))

        mode_badge = font.render(f"[{vis.mode_names[vis.mode]}]", True, COLOR_TEXT_DIM)
        screen.blit(mode_badge, (WINDOW_WIDTH - 130, 18))

        # Visualizer Canvas
        vis.draw(screen, audio.fft_buffer)

        # Progress / Scrub Bar & Timer
        frac, elapsed, total = audio.get_progress()
        pygame.draw.rect(screen, COLOR_SEEK_BG, seek_bar_rect, border_radius=4)
        fill_width = int(frac * seek_bar_rect.width)
        if fill_width > 0:
            pygame.draw.rect(screen, COLOR_SEEK_FILL, (seek_bar_rect.x, seek_bar_rect.y, fill_width, seek_bar_rect.height), border_radius=4)
        
        # Scrubber Knob
        handle_x = seek_bar_rect.x + fill_width
        pygame.draw.circle(screen, COLOR_SEEK_HANDLE, (handle_x, seek_bar_rect.centery), 6)

        # Time Stamps
        time_str = f"{elapsed // 60:02d}:{elapsed % 60:02d} / {total // 60:02d}:{total % 60:02d}"
        time_surf = font.render(time_str, True, (210, 220, 230))
        screen.blit(time_surf, (seek_bar_rect.right - 90, seek_bar_rect.y - 16))

        # Control Buttons
        for btn in buttons:
            btn.draw(screen, font)

        # Playlist Window
        pygame.draw.rect(screen, COLOR_PANEL, playlist_rect)
        pygame.draw.rect(screen, COLOR_PANEL_BORDER, playlist_rect, 1)

        items_to_show = playlist_rect.height // 22
        for idx in range(items_to_show):
            track_idx = scroll_offset + idx
            if track_idx >= len(playlist.tracks):
                break

            t_path = playlist.tracks[track_idx]
            is_active = (track_idx == playlist.current_idx)
            item_y = playlist_rect.y + idx * 22

            if is_active:
                pygame.draw.rect(screen, (0, 60, 50), (playlist_rect.x + 2, item_y + 1, playlist_rect.width - 4, 20))

            indicator = ">> " if is_active else f"{track_idx + 1:02d}. "
            item_name = Path(t_path).name
            col = COLOR_TEXT_BRIGHT if is_active else (180, 190, 200)
            txt = font.render(f"{indicator}{item_name[:55]}", True, col)
            screen.blit(txt, (playlist_rect.x + 8, item_y + 3))

        pygame.display.flip()
        clock.tick(FPS)

    audio.close()
    pygame.quit()

if __name__ == "__main__":
    main()