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
- `config.yaml` / `config.py` – alle Hyperparameter (Environment, Belohnung,
  Training, PPO) an einer Stelle, siehe unten.
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
python train_mario.py Super_Mario_Land_World_Rev_1.gb --steps 500000 --envs 4
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

Alle Gewichte stehen in `config.yaml` unter `env.reward` – dort anpassen,
kein Code-Aendern noetig (siehe `_compute_reward` in `mario_env.py` fuer die
genaue Umsetzung):

- **+1 pro Kachel Fortschritt nach rechts** (`level_progress`) – das
  Hauptsignal, damit der Agent lernt, sich vorwärtszubewegen
- **+0.02 pro Score-Punkt** (Gegner besiegt, Bonusgegenstände)
- **+1 pro eingesammelter Münze**
- **+5 Bonus, wenn ein Gegner in der Nähe verschwindet UND im selben
  Schritt der Score steigt** (`enemy_defeat_bonus`) – starkes Indiz, dass
  er besiegt wurde (draufgesprungen/Feuerball), nicht nur aus dem Bild
  gelaufen ist
- **-0.05 pro Schritt** (kleine Zeitstrafe, damit der Agent nicht einfach
  stehen bleibt)
- **-25 bei Lebensverlust / Game Over**, Episode endet dann

### Gezieltes Springen (2026-09-03)

Auf Vorschlag des Nutzers: Statt Springen pauschal zu bestrafen (alte
`jump_cost`-Lösung) oder Gegner pauschal zu ignorieren, prüft
`_compute_reward` jetzt pro Schritt, ob in den nächsten `jump_lookahead`
Kacheln (Standard: 4) **rechts vor Mario** einer von drei Fällen zutrifft:

- ein **Gegner** auf Marios eigener Höhe,
- ein **festes Hindernis** (Pfeife, Block, Mauer) auf Marios eigener Höhe,
- eine **Schlucht** – keine Bodenkachel mehr in der zuletzt bekannten
  Boden-Reihe (auch während Mario gerade in der Luft ist, damit ein
  laufender Sprung über die Schlucht nicht fälschlich als "kein Grund"
  gewertet wird).

Ist **keiner** dieser Fälle erkannt und A wird trotzdem gehalten, kostet
das `jump_cost` (Standard 0.05). Ist **einer** davon erkannt, entfällt die
Strafe und es gibt stattdessen `jump_bonus` (Standard 0.02). Damit lernt
der Agent direkt "lauf normal, spring nur wenn nötig", statt das nur
indirekt über eine pauschale Kosten/Nutzen-Abwägung zu finden.

Die ältere, umgebungsweite Gegner-Nähe-Strafe (`enemy_proximity_penalty`,
bis -0.3/Schritt bei Nähe zu einem Gegner auf Marios Höhe, sofern nicht
darüber gesprungen) bleibt als Mechanik im Code erhalten, ist aber
standardmäßig auf 0 gestellt – sie würde denselben Fall wie oben doppelt
bewerten. Bei Bedarf (z. B. falls der Agent trotz gezieltem Springen zu
nah an Gegner herangeht) lässt sie sich in `config.yaml` wieder aktivieren
(`enemy_proximity_penalty: 0.05–0.1`).

**Grenzen der Heuristik:** Die Erkennung basiert auf dem 16×20-Kachelraster
aus `game_area()`, nicht auf exakten Kollisionsdaten. Insbesondere die
Schlucht-Erkennung merkt sich die zuletzt unter Mario gesehene Bodenreihe
und kann bei sehr unregelmäßigem Terrain (Treppen, Höhenwechsel) auch mal
fälschlich anschlagen oder ausbleiben. `jump_lookahead`, `jump_cost` und
`jump_bonus` lassen sich in `config.yaml` anpassen, falls der Agent zu
vorsichtig (zu oft springt) oder zu riskant (zu selten springt) wird.

### Hohe Hindernisse / langer Sprung nötig (2026-09-03)

Manche Hindernisse (z. B. hohe Pfeifen) lassen sich nicht mit einem kurzen
Hüpfer überwinden – die Sprunghöhe in Super Mario Land hängt direkt davon
ab, wie viele Schritte am Stück A gehalten wird (siehe Bugfix "hohe
Sprünge" oben). Damit "gezieltes Springen" nicht nur *ob*, sondern auch
*wie lange* gesprungen wird, mit abdeckt: Sobald A während eines erkannten
Grundes (Gegner/Hindernis/Schlucht voraus) `sustained_jump_steps`
Schritte am Stück durchgehend gehalten wurde (Standard: 4), gibt es
**einmalig** einen Zusatzbonus `sustained_jump_bonus` (Standard 0.1).

Eine genauere Lösung wäre gewesen, die tatsächliche Höhe des Hindernisses
aus dem Kachelraster zu messen und den nötigen Sprung entsprechend zu
dosieren – das haben wir per Stichprobe im echten Spiel geprüft, aber
normales flaches Terrain und echte Hindernisse ließen sich dabei nicht
zuverlässig genug unterscheiden (beide erschienen im Test oft mit
derselben gemessenen "Höhe"). Der Sustained-Bonus umgeht dieses Problem:
er belohnt einfach längeres Halten, wenn ein Grund dafür besteht – das ist
unabhängig von der tatsächlichen Hindernishöhe immer die richtige
Strategie, um überhaupt eine höhere Sprungoption auszuprobieren, ohne dass
die genaue Höhe bekannt sein muss.

Die Gegner-Erkennung nutzt das Kachelraster aus `game_area()` (Mario und
Gegner werden dort als Sprites mit erkennbaren Kategorie-Werten geführt)
und ist eine Heuristik, kein exaktes Kollisionssystem – aber im Test
messbar zuverlässig.

Diese Gewichte lassen sich anpassen, falls der Agent sich merkwürdig
verhält (z. B. zu vorsichtig ist oder Risiken für Münzen eingeht, die
nicht lohnen).

### Warum der Agent nur hoppelt statt normal zu laufen (2026-09-02, verfeinert 2026-09-03)

Falls dein Agent in `watch_agent.py` durchgehend hüpft statt zu laufen: Ohne
eigene Kosten fürs Springen ist "dauerhaft A gedrückt halten" reward-neutral
bis reward-positiv, egal ob gerade wirklich ein Grund zum Springen besteht –
eine Art kostenlose Versicherung, die PPO zuverlässig findet. Seit 2026-09-03
löst das "Gezielte Springen" oben das direkter: Springen kostet nur dann
etwas, wenn nichts voraus ist. Das gilt für **neue** Trainingsläufe. Ein
bereits trainierter Checkpoint hat das Hoppeln aber schon gelernt und
"verlernt" es nicht von selbst – dafür muss mit dem aktualisierten
`mario_env.py`/`config.yaml` neu trainiert werden (am besten von vorne, nicht
mit `--resume` von einem alten Checkpoint, da der schon auf das alte
Verhalten eingeschwungen ist).

**Hoppelt es nach einem Neu-Training immer noch?** Zwei Dinge zuerst prüfen:

1. Läuft wirklich der aktuelle Code? `mario_env.py` sollte die Funktion
   `_reward_jump_bonus`/`_jump_lookahead` in `_compute_reward` enthalten
   (Suche nach `jump_lookahead` im Ordner). Wurde vor dem Training wirklich
   die neue Version von `mario_env.py`/`config.yaml` in den Ordner kopiert
   und nicht die alte weiterverwendet?
2. `jump_cost` in `config.yaml` schrittweise erhöhen (z. B. 0.08, dann
   0.15) und/oder `jump_bonus` senken, dann neu trainieren.

**Zur Trainingsdauer:** 500.000–2.000.000 Schritte sind für dieses Setup
ein sinnvoller Rahmen (Minuten bis wenige Stunden). Eine Milliarde Schritte
wäre um Größenordnungen mehr, als für ein PPO-Setup dieser Größe üblich
oder nötig ist – falls das kein Tippfehler war, lohnt es sich eher, mit
kürzeren Läufen (z. B. 1–2 Mio. Schritte) schnell zu iterieren und die
Belohnung/Architektur zu prüfen, statt sehr lange auf denselben (evtl.
fehlerhaften) Aufbau zu setzen.
