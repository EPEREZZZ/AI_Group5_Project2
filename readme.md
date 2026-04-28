
Reversi setup

Prereqs:
- Python 3.10+ (We used Python 3.12)

1) Install dependencies
    From the project directory, run:

    macOS:
        1. python3 -m venv .venv
        2. source .venv/bin/activate
        3. pip install -r requirements.txt

    Windows (PowerShell):
        python -m venv .venv
        .\.venv\Scripts\Activate.ps1
        pip install -r requirements.txt

    Windows (Command Prompt):
        python -m venv .venv
        .venv\Scripts\activate
        pip install -r requirements.txt

    Alternative (VS Code users)
    1. Press Ctrl + Shift + P
    2. Select "Python: Create Environment"
    3. Choose "Venv"
    4. Select requirements.txt

2) Run game
    Open 3 terminals in the project folder (with the same virtual environment activated using (source .venv/bin/activate) for mac)

    Terminal 1 (Reversi server):
    python3 reversi_server.py

    Terminal 2 (player 1 - White):
    python3 RL_player.py

    Terminal 3 (player 2 - Black):
    python3 greedy_player.py

Notes:
    - After both players connect, click server window to start game
    - To force stop the code, run (control + c) in all 3 terminals
