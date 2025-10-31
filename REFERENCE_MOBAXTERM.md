┌─────────────────────────────────────────┐
│ MobaXterm + VSCode X11 Quick Setup      │
├─────────────────────────────────────────┤
│ 1. Open MobaXterm (X server auto-starts│
│ 2. Create SSH session with X11-Forward  │
│ 3. In VSCode: Add ForwardX11 yes to SSH│
│ 4. Connect via VSCode                   │
│ 5. In terminal: export DISPLAY=:10.0   │
│ 6. Run: python transcription_gui_qt.py │
└─────────────────────────────────────────┘