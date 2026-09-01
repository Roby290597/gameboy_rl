# Super Mario Land – Reinforcement Learning

Ein kleines, funktionierendes RL-Projekt für Super Mario Land (Game Boy),
als einfacherer Einstieg vor dem größeren Pokémon-Crystal-Vorhaben.

**Wichtiger Unterschied zu Pokémon Crystal:** Super Mario Land ist ein
normales Game-Boy-Modul (MBC1, kein MBC30/RTC) und wird von PyBoy direkt
über einen eingebauten Game-Wrapper unterstützt. Es ist **keine** eigene
RAM-Adress-Suche und **kein** Custom-Build von PyBoy nötig – die ganz
normale Version von PyPI reicht.

## Was ist enthalten

- `mario_env.py` – Gymnasium-Environment, gebaut auf `pyboy.game_wrapper`.
  Liefert als Beobachtung ein 16×20-Kachelraster des Bildschirms, als
  Aktionen eine kompakte Auswahl an Tastenkombinationen (rechts laufen,
  springen, rennen, etc.). Die Belohnung basiert auf Level-Fortschritt
  (Hauptsignal), Score, Münzen und einer Strafe bei Lebensverlust.
- `train_mario.py` – Trainingsskript (PPO, stable-baselines3), mit
  mehreren parallelen Emulator-Instanzen, Tensorboard-Logging und
  automatischem Speichern von Checkpoints.
- `watch_agent.py` – zeigt einen trainierten Agenten (oder erstmal nur
  Zufallsaktionen, zum Testen) in einem sichtbaren Fenster beim Spielen.
- `Super_Mario_Land_World_Rev_1.gb` – deine hochgeladene ROM (Kopie).

**Bereits erfolgreich getestet:** Environment (Boot, Belohnung,
Fortschrittsmessung) und ein kurzer Trainingslauf (PPO mit 2 parallelen
Umgebungen) wurden vor der Auslieferung verifiziert und liefen fehlerfrei.

## Einrichtung (Windows 11)

```powershell
cd mario_rl
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Das dauert nur wenige Minuten, da hier (anders als bei Crystal) keine
Kompilierung von PyBoy aus dem Quellcode nötig ist – ganz normale
`pip install`-Pakete reichen.

## Ausprobieren, ob alles läuft

```powershell
python watch_agent.py Super_Mario_Land_World_Rev_1.gb
```

Öffnet ein Fenster, Mario bewegt sich mit Zufallsaktionen (noch kein
trainierter Agent) – nur zum Testen, dass die Umgebung funktioniert.

## Training starten

```powershell
python train_mario.py Super_Mario_Land_World_Rev_1.gb --steps 100000 --envs 4
```

- `--steps`: Gesamtzahl der Trainingsschritte (500.000 ist ein sinnvoller
  erster Versuch, dauert je nach PC von ca. 20 Minuten bis über eine
  Stunde; mehr Schritte = besserer Agent)
- `--envs`: Anzahl paralleler Emulator-Instanzen (an CPU-Kerne anpassen,
  z. B. 4–8)
- Fortschritt live verfolgen: `tensorboard --logdir runs` und dann
  `http://localhost:6006` im Browser öffnen
- Checkpoints landen automatisch unter `checkpoints/mario_ppo/`

Training fortsetzen (z. B. am nächsten Tag):

```powershell
python train_mario.py Super_Mario_Land_World_Rev_1.gb --resume checkpoints/mario_ppo/mario_20000_steps.zip --steps 500000
```

## Trainierten Agenten zuschauen

```powershell
python watch_agent.py Super_Mario_Land_World_Rev_1.gb --model checkpoints/mario_ppo/mario_final.zip
```

## Wie die Belohnung funktioniert (zum Nachvollziehen/Anpassen)

In `mario_env.py`, Funktion `_compute_reward`:

- **+1 pro Kachel Fortschritt nach rechts** (`level_progress`) – das
  Hauptsignal, damit der Agent lernt, sich vorwärtszubewegen
- **+0.02 pro Score-Punkt** (Gegner besiegt, Bonusgegenstände)
- **+1 pro eingesammelter Münze**
- **-0.05 pro Schritt** (kleine Zeitstrafe, damit der Agent nicht einfach
  stehen bleibt)
- **-25 bei Lebensverlust / Game Over**, Episode endet dann

Diese Gewichte lassen sich anpassen, falls der Agent sich merkwürdig
verhält (z. B. zu vorsichtig ist oder Risiken für Münzen eingeht, die
nicht lohnen).
