"""
Debug-Tool: startet PyBoy sichtbar (wie watch_agent.py) und zeigt zusaetzlich
im Terminal das Kachelraster, das MarioLandEnv fuer die Belohnung sieht -
mit Mario- und Gegner-Position, dem Vorschau-Fenster fuer "gezieltes
Springen" und ob/warum genau in diesem Moment ein Sprung-Grund erkannt wird.
Das ist exakt dieselbe Sicht, die _compute_reward() in mario_env.py fuer die
Belohnung nutzt - nuetzlich, um nachzuvollziehen, wie weit Gegner/Hindernisse
von Mario entfernt sind, ohne den Code selbst lesen zu muessen.

Aufruf:
    python debug_view.py Super_Mario_Land_World_Rev_1.gb
    python debug_view.py Super_Mario_Land_World_Rev_1.gb --model checkpoints/mario_ppo/mario_final.zip
    python debug_view.py Super_Mario_Land_World_Rev_1.gb --speed 0        # ungebremst
    python debug_view.py Super_Mario_Land_World_Rev_1.gb --print-interval 1.0  # seltener aktualisieren
    python debug_view.py Super_Mario_Land_World_Rev_1.gb --clear          # Terminal vor jeder Ausgabe leeren

Ohne --model spielt ein Zufallsagent (wie bei watch_agent.py), nuetzlich um
in Ruhe zu beobachten, ohne dass ein trainiertes Modell noetig ist.

Zeichen im Kachelraster:
    M = Mario         E = Gegner (Kategorie-Wert daneben in Klammern)
    # = feste Kachel (Boden, Bloecke, Pfeifen, ...)   . = leer
Die Vorschau-Spalten fuer "gezieltes Springen" (jump_lookahead rechts von
Mario) sind mit einem "v" in der Zeile darueber markiert, die zuletzt
bekannte Bodenreihe mit einem "-" am linken Rand.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from pyboy.utils import WindowEvent

from config import load_config
from mario_env import MarioLandEnv


def _render_grid(env: MarioLandEnv, area: np.ndarray) -> str:
    lines = []

    mario_cells = np.argwhere(area == env._mario_category)
    enemy_mask = np.isin(area, env._enemy_categories)
    enemy_cells = np.argwhere(enemy_mask)

    mario_row_top = mario_row_bottom = None
    mario_col_max = None
    if mario_cells.size:
        mario_row_top = int(mario_cells[:, 0].min())
        mario_row_bottom = int(mario_cells[:, 0].max())
        mario_col_max = int(mario_cells[:, 1].max())

    lookahead_cols: list[int] = []
    if mario_col_max is not None:
        lookahead_cols = list(
            range(mario_col_max + 1, min(mario_col_max + 1 + env._jump_lookahead, area.shape[1]))
        )

    # Markerzeile fuer das Vorschau-Fenster, oberhalb des Rasters.
    marker_row = ["  "] + [" v " if c in lookahead_cols else "   " for c in range(area.shape[1])]
    lines.append("".join(marker_row))

    header = "    " + "".join(f"{c:>3}" for c in range(area.shape[1]))
    lines.append(header)

    for r in range(area.shape[0]):
        row_marker = "->" if r == env._ground_row else "  "
        cells = []
        for c in range(area.shape[1]):
            val = int(area[r, c])
            if val == env._mario_category:
                sym = "M"
            elif val in env._enemy_categories:
                sym = "E"
            elif val != 0:
                sym = "#"
            else:
                sym = "."
            cells.append(f"{sym:>3}")
        lines.append(f"{row_marker}{r:>2}" + "".join(cells))

    lines.append("")

    if mario_cells.size:
        lines.append(
            f"Mario: Zeile {mario_row_top}-{mario_row_bottom}, Spalte (bis) {mario_col_max}   "
            f"Boden-Referenz (zuletzt bekannt): Zeile {env._ground_row}"
        )

    if enemy_cells.size and mario_cells.size:
        col_diff = enemy_cells[:, 1].astype(np.float32) - float(mario_cells[:, 1].mean())
        row_diff = enemy_cells[:, 0].astype(np.int32) - mario_row_top
        same_level = (row_diff >= 0) & (row_diff <= 1)
        ahead = col_diff > 0

        ahead_same_level = ahead & same_level
        if np.any(ahead_same_level):
            nearest = float(col_diff[ahead_same_level].min())
            lines.append(f"Naechster Gegner VORAUS AUF MARIOS EBENE: {nearest:.1f} Kacheln entfernt (das ist die relevante Gefahr)")
        elif np.any(ahead):
            nearest = float(col_diff[ahead].min())
            lines.append(
                f"Naechster Gegner voraus: {nearest:.1f} Kacheln entfernt, aber NICHT auf Marios "
                "Ebene (Zeile passt nicht) - zaehlt laut Reward-Logik nicht als Gefahr"
            )

        behind = ~ahead
        if np.any(behind):
            nearest_behind = float((-col_diff[behind]).min())
            lines.append(f"Naechster Gegner dahinter: {nearest_behind:.1f} Kacheln entfernt (fuer Reward irrelevant)")
    else:
        lines.append("Kein Gegner im aktuellen Bildausschnitt sichtbar")

    # Dieselbe Grund-Erkennung wie in _compute_reward, nur zur Anzeige.
    threat_reasons = []
    if mario_cells.size and lookahead_cols:
        if enemy_cells.size:
            e_row_diff = enemy_cells[:, 0].astype(np.int32) - mario_row_top
            e_col_diff = enemy_cells[:, 1].astype(np.float32) - float(mario_cells[:, 1].mean())
            ahead_enemy = (
                (e_row_diff >= 0) & (e_row_diff <= 1)
                & (e_col_diff > 0) & (e_col_diff <= env._jump_lookahead)
            )
            if np.any(ahead_enemy):
                threat_reasons.append("Gegner voraus")

        row_span = area[mario_row_top : mario_row_bottom + 1, lookahead_cols]
        solid = (row_span != 0) & (row_span != env._mario_category)
        if np.any(solid):
            threat_reasons.append("Hindernis voraus")

        if env._ground_row is not None and env._ground_row < area.shape[0]:
            ground_span = area[env._ground_row, lookahead_cols]
            if np.all(ground_span == 0):
                threat_reasons.append("Schlucht voraus")

    jumping = WindowEvent.PRESS_BUTTON_A in env._held_buttons
    if threat_reasons:
        lines.append(f"GRUND ZUM SPRINGEN ERKANNT: {', '.join(threat_reasons)}  (A gehalten: {jumping})")
    else:
        lines.append(f"Kein Sprung-Grund erkannt  (A gehalten: {jumping})")

    return "\n".join(lines)


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("rom", help="Pfad zur Super-Mario-Land-.gb-Datei")
    parser.add_argument("--config", default=None, help="Pfad zu einer alternativen config.yaml")
    parser.add_argument("--model", default=None, help="Pfad zu einem trainierten PPO-Modell (.zip)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--speed", type=int, default=1, help="0=ungebremst, 1=normal, bis 5")
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        help="Wie oft (in Sekunden) das Kachelraster im Terminal aktualisiert wird.",
    )
    parser.add_argument("--clear", action="store_true", help="Terminal vor jeder Ausgabe leeren")
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)

    env = MarioLandEnv.from_config(args.rom, cfg, headless=False)
    env.pyboy.set_emulation_speed(args.speed)

    model = None
    if args.model:
        from stable_baselines3 import PPO

        model = PPO.load(args.model)
        print(f"Modell geladen: {args.model}")
    else:
        print("Kein Modell angegeben - spiele mit Zufallsaktionen.")

    last_print = 0.0
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            now = time.time()
            if now - last_print >= args.print_interval:
                last_print = now
                area = obs.reshape(16, 20)
                if args.clear:
                    print("\033c", end="")
                print(_render_grid(env, area))
                print(f"Reward (dieser Schritt): {reward:+.3f}   Gesamt: {total_reward:+.1f}")
                print("=" * 70)

        print(
            f"Episode {ep + 1}: Belohnung={total_reward:.1f}  "
            f"Welt={info['world']}  Fortschritt={info['level_progress']}  "
            f"Leben={info['lives_left']}  Score={info['score']}"
        )

    env.close()


if __name__ == "__main__":
    main()
