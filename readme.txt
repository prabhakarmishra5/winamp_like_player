# Winamp Classic Mini-Player

A zero-dependency, standalone MP3 player inspired by classic Winamp. Built with Python, `miniaudio`, `numpy`, and `pygame-ce`, featuring real-time 16-band spectrum analysis, oscilloscope visualization, interactive track scrubbing, and playlist management.

---

## Features

- **Self-Contained Audio Engine:** Powered by `miniaudio` (embedding `dr_mp3`), eliminating dependencies on OS media frameworks (no Windows Media Player, GStreamer, or DirectShow required).
- **Winamp-Style Visualizer:**
  - 16-band logarithmic spectrum analyzer with gravity-decay peak caps.
  - Oscilloscope waveform visualizer.
  - Combined mode.
- **Full Transport Controls:** Dedicated `PLAY`, `PAUSE`, `STOP` (cursor reset to `00:00`), `<<` (Rewind 5s), `>>` (Fast Forward 5s), `|<` (Previous), and `>|` (Next).
- **Interactive Scrubber:** Click or drag anywhere along the timeline to seek immediately.
- **Playlist & Shuffle:** Add individual or batch MP3s via native file browser dialogs, toggle shuffle playback, and export/import playlists to `.json`.

---

## Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| `SPACE` | Toggle Play / Pause |
| `X` | Play / Resume |
| `C` | Pause |
| `V` | Stop (Halts and rewinds to beginning) |
| `TAB` | Cycle Visualizer Mode (Spectrum / Oscilloscope / Combined) |
| `LEFT ARROW` | Seek Backward (-5 seconds) |
| `RIGHT ARROW`| Seek Forward (+5 seconds) |
| `Z` | Previous Track |
| `B` | Next Track |
| `S` | Toggle Shuffle Mode |
| `O` | Open File Dialog (+ADD tracks) |

---

## Development Setup

### 1. Prerequisites
- Python 3.10 – 3.13
- `pip`

### 2. Install Dependencies
```bash
pip install miniaudio numpy pygame-ce pyinstaller

(Note: On minimal Linux distributions without GUI tools, ensure standard Tkinter is present: sudo apt install python3-tk on Debian/Ubuntu or sudo dnf install python3-tkinter on Fedora).

### 3. Run directly

# Run empty or launch with a file:
python player.py
python player.py "path/to/song.mp3"

######

Standalone Compilation (No Python Required)
PyInstaller packages the Python interpreter, libraries, and C-bindings into a single self-contained executable.

Important Cross-Compilation Rule: PyInstaller creates binaries for the OS it is executed on.
--> To make a .exe for Windows, run the build on a Windows machine.
--> To make an ELF binary for Linux, run the build on a Linux machine (or inside WSL/Docker).

######

Windows Build Instructions
Open PowerShell or Command Prompt inside your virtual environment and run:

pyinstaller --onefile --noconsole --name "WinampMini" player.py

Flags Explained:
--> --onefile: Compresses everything into a single standalone .exe.
--> --noconsole: Hides the background black command prompt window so only the GUI launches.
--> --name "WinampMini": Sets the executable name (dist/WinampMini.exe).

(Optional: Add an icon using --icon=icon.ico)

Linux Build Instructions
Open your terminal inside your virtual environment and run:

# 1. Ensure binutils and Tk bindings are installed
sudo apt update && sudo apt install -y python3-tk binutils
# 2. Build the ELF binary
pyinstaller --onefile --noconsole --name "winamp-mini" player.py
# 3. Grant execution permissions
chmod +x dist/winamp-mini
# 4. Run the executable
./dist/winamp-mini

Troubleshooting Common Build Issues
1. Hidden Imports Warning: If PyInstaller fails to bundle tkinter on Windows, add --collect-all tkinter to the build command:
pyinstaller --onefile --noconsole --collect-all tkinter player.py

2. Antivirus False Positives on Windows: Single-file PyInstaller .exe files occasionally trigger Windows Defender heuristics. If this occurs, 
sign the binary, package it inside a .zip, or build using the --onedir directory flag instead of --onefile.