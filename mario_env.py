"""
Gymnasium-Environment fuer Super Mario Land (Game Boy), gebaut auf PyBoys
eingebautem Game-Wrapper fuer dieses Spiel (pyboy.game_wrapper).

Der Wrapper liefert bereits fertig ausgelesene Werte wie Level-Fortschritt
(X-Position), Leben, Score, Muenzen und ein sauberes Kachel-Raster des
Bildschirms - dadurch ist hier (anders als beim Pokemon-Crystal-Versuch)
KEINE eigene RAM-Adress-Suche noetig.

Beobachtung: 16x20-Kachelraster des Spielbereichs (ohne HUD), als flacher
uint8-Vektor. Aktionen: eine kleine, sinnvolle Auswahl an Tastenkombinationen
(rechts laufen, springen, rennen/Feuerball, etc.) statt aller moeglichen
Tastenkombinationen einzeln.

Belohnung:
    + Fortschritt nach rechts (level_progress steigt)          -> Hauptsignal
    + Muenzen eingesammelt
    + Score gestiegen (Gegner besiegt, Bonus-Items etc.)
    + Level geschafft (naechste Welt/Level erreicht)            -> grosser Bonus
    - Leben verloren / Game Over                                -> grosse Strafe
    - kleine Zeitstrafe pro Schritt (Anreiz, nicht zu trödeln)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent


# Jede Aktion ist die MENGE der Tasten, die waehrend dieses Schritts gehalten
# werden sollen (nicht nur kurz angetippt). Der Sprung in Super Mario Land ist
# wie im Original ueber die Haltedauer von A gesteuert: A nur 1 Frame gedrueckt
# = kurzer Hueper, A mehrere Schritte am Stueck gehalten (weil der Agent
# wiederholt dieselbe "...+A"-Aktion waehlt) = hoher Sprung. Damit das
# funktioniert, darf der Knopf zwischen zwei Schritten NICHT losgelassen
# werden, solange die neue Aktion ihn weiter enthaelt - siehe step().
_ACTIONS: list[frozenset[int]] = [
    frozenset(),  # 0: nichts tun
    frozenset({WindowEvent.PRESS_ARROW_RIGHT}),  # 1: rechts laufen
    frozenset({WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A}),  # 2: rechts + springen
    frozenset({WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_B}),  # 3: rechts + rennen/Feuerball
    frozenset(
        {WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B}
    ),  # 4: rechts + rennender (hoher) Sprung
    frozenset({WindowEvent.PRESS_BUTTON_A}),  # 5: auf der Stelle springen
    frozenset({WindowEvent.PRESS_ARROW_LEFT}),  # 6: links laufen
    frozenset({WindowEvent.PRESS_ARROW_DOWN}),  # 7: ducken / in Rohr
]
ACTION_NAMES = [
    "NOOP", "RIGHT", "RIGHT+A", "RIGHT+B", "RIGHT+A+B", "A", "LEFT", "DOWN",
]

_RELEASE_OF = {
    WindowEvent.PRESS_ARROW_UP: WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.PRESS_ARROW_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.PRESS_BUTTON_A: WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B: WindowEvent.RELEASE_BUTTON_B,
}


class MarioLandEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str,
        headless: bool = True,
        frame_skip: int = 4,
        world_level: tuple[int, int] | None = None,
        max_steps_without_progress: int = 300,
    ):
        super().__init__()
        window = "null" if headless else "SDL2"
        self.pyboy = PyBoy(rom_path, window=window)
        if not headless:
            self.pyboy.set_emulation_speed(1)
        else:
            self.pyboy.set_emulation_speed(0)

        self.game_wrapper = self.pyboy.game_wrapper
        self.game_wrapper.game_area_mapping(self.game_wrapper.mapping_compressed, 0)

        self._world_level = world_level
        self._frame_skip = frame_skip
        self._max_steps_without_progress = max_steps_without_progress

        self.action_space = spaces.Discrete(len(_ACTIONS))
        # game_area() liefert ein 16x20-Raster (Zeilen x Spalten).
        self.observation_space = spaces.Box(low=0, high=27, shape=(16 * 20,), dtype=np.uint8)

        self._started = False
        self._last_progress = 0
        self._steps_since_progress = 0
        self._last_lives = 0
        self._last_score = 0
        self._last_coins = 0
        # Aktuell physisch gehaltene Tasten (PRESS_*-Konstanten). Wird in step()
        # gegen die neue Ziel-Aktion abgeglichen, damit z.B. A ueber mehrere
        # Schritte hinweg durchgehend gedrueckt bleiben kann (siehe _ACTIONS).
        self._held_buttons: frozenset[int] = frozenset()

    # -- Gymnasium API -----------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if not self._started:
            self.game_wrapper.start_game(world_level=self._world_level)
            self._started = True
        else:
            self.game_wrapper.reset_game()

        self._last_progress = self.game_wrapper.level_progress
        self._steps_since_progress = 0
        self._last_lives = self.game_wrapper.lives_left
        self._last_score = self.game_wrapper.score
        self._last_coins = self.game_wrapper.coins

        # Alle evtl. noch gehaltenen Tasten aus der vorherigen Episode loesen,
        # damit kein Zustand (z.B. "A haengt noch") in die neue Episode leckt.
        for held in self._held_buttons:
            self.pyboy.send_input(_RELEASE_OF[held])
        self._held_buttons = frozenset()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        target = _ACTIONS[action]

        # Nur die Tasten aendern, die sich gegenueber dem letzten Schritt
        # tatsaechlich unterscheiden. Eine Taste (z.B. A), die in beiden
        # Aktionen enthalten ist, bleibt so ueber mehrere step()-Aufrufe
        # hinweg PHYSISCH durchgehend gedrueckt - genau das braucht Super
        # Mario Land, um ueberhaupt einen hohen Sprung ausloesen zu koennen.
        for held in self._held_buttons - target:
            self.pyboy.send_input(_RELEASE_OF[held])
        for new in target - self._held_buttons:
            self.pyboy.send_input(new)
        self._held_buttons = target

        for _ in range(self._frame_skip - 1):
            self.pyboy.tick(1, False)
        self.pyboy.tick(1, True)  # letzten Frame rendern, fuer Beobachtung/Anzeige

        reward, terminated = self._compute_reward()
        truncated = self._steps_since_progress > self._max_steps_without_progress

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        return np.array(self.pyboy.screen.ndarray)

    def close(self):
        self.pyboy.stop(save=False)

    # -- Hilfsfunktionen -----------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        area = np.asarray(self.game_wrapper.game_area(), dtype=np.uint8)
        return area.flatten()

    def _get_info(self) -> dict:
        return {
            "world": self.game_wrapper.world,
            "level_progress": self.game_wrapper.level_progress,
            "lives_left": self.game_wrapper.lives_left,
            "score": self.game_wrapper.score,
            "coins": self.game_wrapper.coins,
            "time_left": self.game_wrapper.time_left,
        }

    def _compute_reward(self) -> tuple[float, bool]:
        progress = self.game_wrapper.level_progress
        lives = self.game_wrapper.lives_left
        score = self.game_wrapper.score
        coins = self.game_wrapper.coins

        reward = 0.0

        progress_delta = progress - self._last_progress
        reward += progress_delta * 1.0
        if progress_delta > 0:
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1

        reward += (score - self._last_score) * 0.02
        reward += (coins - self._last_coins) * 1.2
        reward -= 0.05  # kleine Zeitstrafe pro Schritt

        terminated = False
        if lives < self._last_lives or self.game_wrapper.game_over():
            reward -= 50.0 #25.0
            terminated = True

        self._last_progress = progress
        self._last_lives = lives
        self._last_score = score
        self._last_coins = coins

        return reward, terminated
