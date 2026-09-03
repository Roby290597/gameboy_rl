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
    + zusaetzlicher Bonus, wenn ein Gegner in der Naehe verschwindet UND
      gleichzeitig der Score steigt (= sehr wahrscheinlich besiegt)
    + Level geschafft (naechste Welt/Level erreicht)            -> grosser Bonus
    + kleiner Bonus fuers Springen, wenn GENAU DANN tatsaechlich ein Gegner,
      Hindernis oder eine Schlucht voraus ist ("gezielter Sprung")
    - Leben verloren / Game Over                                -> grosse Strafe
    - Gegner auf gleicher Hoehe in der Naehe (nicht darueber gesprungen)
      -> kleine, mit der Naehe wachsende Strafe, Anreiz zum Ausweichen/Draufspringen
      (Standardgewicht 0 - siehe "Gezieltes Springen" unten)
    - kleine Zeitstrafe pro Schritt (Anreiz, nicht zu trödeln)
    - kleine Strafe fuer jeden Schritt, in dem A/Sprung gehalten wird, WENN
      dabei kein Gegner/Hindernis/keine Schlucht voraus ist
      -> ohne das ist Dauerspringen "kostenlos" (siehe unten), der Agent lernt
         sonst staendiges Hoppeln statt normal zu laufen

Warum die Sprung-Strafe noetig ist: Ohne eigene Kosten fuers Springen ist
"immer A gedrueckt halten" reward-neutral bis reward-positiv, egal ob gerade
wirklich ein Grund zum Springen besteht - PPO entdeckt das zuverlaessig und
der Agent hoppelt dann durchgehend, anstatt normal zu laufen.

Gezieltes Springen (2026-09-03): Statt Springen pauschal zu bestrafen, wird
in _compute_reward() pro Schritt geprueft, ob im Kachelraster VORAUS (die
naechsten `jump_lookahead` Spalten rechts von Mario) ein Gegner auf Marios
Hoehe, ein festes Hindernis (z.B. Pfeife/Block) auf Marios Hoehe, oder eine
Schlucht (keine Bodenkachel in der zuletzt bekannten Boden-Reihe) erkannt
wird. Nur wenn KEINS davon zutrifft, kostet Springen etwas (`jump_cost`);
ist tatsaechlich etwas voraus, entfaellt die Strafe und es gibt stattdessen
einen kleinen Bonus (`jump_bonus`). Das soll den Agenten direkt darauf
trimmen, normal zu laufen und nur dann zu springen, wenn es noetig ist -
anstatt (wie bei der alten pauschalen Sprung-Strafe) indirekt ueber
Kosten/Nutzen-Abwaegung dorthin zu finden. Die aeltere, umgebungsweite
Gegner-Naehe-Strafe (`enemy_proximity_penalty`) bleibt als Mechanik erhalten,
ist standardmaessig aber auf 0 gestellt, um nicht mit diesem gezielteren
Signal zu kollidieren (beide bestrafen sonst denselben Fall doppelt).

Sustained-Jump-Bonus (2026-09-03): Manche Hindernisse (z.B. hohe Pfeifen)
brauchen einen laenger gehaltenen, hohen Sprung statt nur einen kurzen
Hueper - die Sprunghoehe haengt direkt von der Haltedauer von A ab. Eine
verlaessliche Kachel-basierte Messung "wie hoch genau ist dieses Hindernis"
war nicht sauber von normalem, flachem Terrain zu unterscheiden (per
Stichprobe im echten Spiel geprueft, siehe Projekt-Notizen), daher gibt es
stattdessen einen einmaligen Zusatzbonus (`sustained_jump_bonus`), sobald A
waehrend eines erkannten Grundes `sustained_jump_steps` Schritte am Stueck
durchgehend gehalten wurde - unabhaengig von der tatsaechlichen
Hindernishoehe ist laenger halten immer die richtige Strategie, um
ueberhaupt eine hoehere Sprungoption auszuprobieren.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from pyboy.plugins.game_wrapper_super_mario_land import (
    base_scripts,
    goomba,
    koopa,
    plant,
    moth,
    flying_moth,
    sphinx,
    big_sphinx,
    fist,
    bill,
    projectiles,
    shell,
    spike,
)


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

# Rohe Tile-IDs aller Gegnertypen (aus dem PyBoy-Wrapper uebernommen) plus
# Stacheln (spike) als zusaetzliche Gefahr. "explosion" (Sterbe-Animation
# eines Gegners) ist bewusst NICHT dabei, das ist kein Gegner mehr.
# Die zugehoerigen game_area()-Kategorien werden erst in __init__ bestimmt,
# da sie von der jeweils aktiven Tile-Mapping-Tabelle abhaengen.
_ENEMY_RAW_TILES = (
    goomba + koopa + plant + moth + flying_moth + sphinx + big_sphinx + fist + bill + projectiles + shell + spike
)

# Wie viele Kacheln Abstand (horizontal) noch als "Gefahr" gilt, und wie
# stark das pro Schritt bestraft wird (linear staerker, je naeher). Nur
# Gegner auf oder knapp unter Marios Hoehe zaehlen als Gefahr - darueber
# springen (Mario also hoeher als der Gegner) loest bewusst KEINE Strafe
# aus, das ist ja der erwuenschte Ausweich-/Stomp-Weg.
_ENEMY_DANGER_RADIUS = 2
_ENEMY_PROXIMITY_PENALTY = 0.1

# Bonus, wenn in einem Schritt sowohl ein Gegner aus dem Sichtfeld
# verschwindet als auch der Score steigt - starkes Indiz fuer "besiegt"
# (Sprung drauf, Feuerball) statt "aus dem Bildschirm gelaufen".
_ENEMY_DEFEAT_BONUS = 5.0


class MarioLandEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str,
        headless: bool = True,
        frame_skip: int = 4,
        world_level: tuple[int, int] | None = None,
        max_steps_without_progress: int = 300,
        reward_progress: float = 1.0,
        reward_score: float = 0.02,
        reward_coin: float = 1.0,
        reward_time_penalty: float = 0.05,
        reward_death: float = 25.0,
        reward_jump_cost: float = 0.02,
        reward_jump_bonus: float = 0.02,
        reward_sustained_jump_bonus: float = 0.1,
        sustained_jump_steps: int = 4,
        jump_lookahead: int = 4,
        enemy_defeat_bonus: float = _ENEMY_DEFEAT_BONUS,
        enemy_danger_radius: float = _ENEMY_DANGER_RADIUS,
        enemy_proximity_penalty: float = _ENEMY_PROXIMITY_PENALTY,
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

        # Kategorie-Werte (nicht die rohen Tile-IDs!) fuer Mario und alle
        # Gegnertypen INNERHALB der aktiven mapping_compressed-Tabelle, fuer
        # die Gegner-Erkennung in _compute_reward(). game_area() liefert
        # bereits diese Kategorien (Hintergrund-Kacheln UND Sprites
        # gemischt, siehe PyBoy-Basisplugin), keine rohen Tile-IDs.
        mapping = self.game_wrapper.mapping_compressed
        self._mario_category = int(mapping[base_scripts[0]])
        self._enemy_categories = tuple(sorted({int(mapping[t]) for t in _ENEMY_RAW_TILES}))
        self._last_enemy_count = 0

        self._world_level = world_level
        self._frame_skip = frame_skip
        self._max_steps_without_progress = max_steps_without_progress

        # Belohnungsgewichte - konfigurierbar (siehe config.yaml/from_config),
        # Standardwerte entsprechen der urspruenglichen, fest verdrahteten
        # Belohnung.
        self._reward_progress = reward_progress
        self._reward_score = reward_score
        self._reward_coin = reward_coin
        self._reward_time_penalty = reward_time_penalty
        self._reward_death = reward_death
        self._reward_jump_cost = reward_jump_cost
        self._reward_jump_bonus = reward_jump_bonus
        self._reward_sustained_jump_bonus = reward_sustained_jump_bonus
        self._sustained_jump_steps = sustained_jump_steps
        self._jump_lookahead = jump_lookahead
        self._enemy_defeat_bonus = enemy_defeat_bonus
        self._enemy_danger_radius = enemy_danger_radius
        self._enemy_proximity_penalty = enemy_proximity_penalty

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
        # Zeilen-Index der zuletzt unter Mario erkannten festen Bodenkachel
        # (fuer die Schlucht-Erkennung waehrend eines Sprungs, siehe
        # _compute_reward). None = noch nicht bekannt (z.B. direkt nach reset()).
        self._ground_row: int | None = None
        # Wie viele Schritte am Stueck A gehalten wurde, WAEHREND ein Grund
        # dafuer voraus war (siehe _compute_reward) - fuer den Sustained-
        # Jump-Bonus, der laengeres Halten (= hoeherer Sprung) foerdert.
        self._consecutive_jump_ahead_steps = 0

    @classmethod
    def from_config(
        cls,
        rom_path: str,
        config: dict,
        headless: bool = True,
        world_level: tuple[int, int] | None = None,
    ) -> "MarioLandEnv":
        """Baut ein MarioLandEnv aus einem geladenen config.yaml-dict (siehe config.py).

        Fehlende Werte fallen auf dieselben Standardwerte wie der normale
        Konstruktor zurueck - config.yaml muss also nicht vollstaendig sein.
        """
        env_cfg = config.get("env", {}) or {}
        reward_cfg = env_cfg.get("reward", {}) or {}
        return cls(
            rom_path,
            headless=headless,
            world_level=world_level,
            frame_skip=env_cfg.get("frame_skip", 4),
            max_steps_without_progress=env_cfg.get("max_steps_without_progress", 300),
            reward_progress=reward_cfg.get("progress", 1.0),
            reward_score=reward_cfg.get("score", 0.02),
            reward_coin=reward_cfg.get("coin", 1.0),
            reward_time_penalty=reward_cfg.get("time_penalty", 0.05),
            reward_death=reward_cfg.get("death", 25.0),
            reward_jump_cost=reward_cfg.get("jump_cost", 0.02),
            reward_jump_bonus=reward_cfg.get("jump_bonus", 0.02),
            reward_sustained_jump_bonus=reward_cfg.get("sustained_jump_bonus", 0.1),
            sustained_jump_steps=reward_cfg.get("sustained_jump_steps", 4),
            jump_lookahead=reward_cfg.get("jump_lookahead", 4),
            enemy_defeat_bonus=reward_cfg.get("enemy_defeat_bonus", _ENEMY_DEFEAT_BONUS),
            enemy_danger_radius=reward_cfg.get("enemy_danger_radius", _ENEMY_DANGER_RADIUS),
            enemy_proximity_penalty=reward_cfg.get("enemy_proximity_penalty", _ENEMY_PROXIMITY_PENALTY),
        )

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
        self._ground_row = None
        self._consecutive_jump_ahead_steps = 0

        area = np.asarray(self.game_wrapper.game_area(), dtype=np.uint8)
        self._last_enemy_count = int(np.isin(area, self._enemy_categories).sum())

        return self._get_obs(area), self._get_info()

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

        area = np.asarray(self.game_wrapper.game_area(), dtype=np.uint8)
        reward, terminated = self._compute_reward(area)
        truncated = self._steps_since_progress > self._max_steps_without_progress

        return self._get_obs(area), reward, terminated, truncated, self._get_info()

    def render(self):
        return np.array(self.pyboy.screen.ndarray)

    def close(self):
        self.pyboy.stop(save=False)

    # -- Hilfsfunktionen -----------------------------------------------------

    def _get_obs(self, area: np.ndarray | None = None) -> np.ndarray:
        if area is None:
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

    def _compute_reward(self, area: np.ndarray) -> tuple[float, bool]:
        progress = self.game_wrapper.level_progress
        lives = self.game_wrapper.lives_left
        score = self.game_wrapper.score
        coins = self.game_wrapper.coins

        reward = 0.0

        progress_delta = progress - self._last_progress
        reward += progress_delta * self._reward_progress
        if progress_delta > 0:
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1

        reward += (score - self._last_score) * self._reward_score
        reward += (coins - self._last_coins) * self._reward_coin
        reward -= self._reward_time_penalty  # kleine Zeitstrafe pro Schritt

        # --- Gegner meiden bzw. besiegen -----------------------------------
        # game_area() enthaelt Hintergrund-Kacheln UND Sprites (Mario,
        # Gegner) als dieselben Kategorie-Zahlen, siehe __init__.
        enemy_mask = np.isin(area, self._enemy_categories)
        enemy_cells = np.argwhere(enemy_mask)
        enemy_count = int(enemy_cells.shape[0])
        mario_cells = np.argwhere(area == self._mario_category)

        mario_row_top = mario_row_bottom = None
        mario_col = None
        mario_col_max = None
        if mario_cells.size:
            mario_row_top = int(mario_cells[:, 0].min())
            mario_row_bottom = int(mario_cells[:, 0].max())
            mario_col = float(mario_cells[:, 1].mean())
            mario_col_max = int(mario_cells[:, 1].max())

        if mario_cells.size and enemy_cells.size:
            row_diff = enemy_cells[:, 0].astype(np.int32) - mario_row_top
            col_diff = np.abs(enemy_cells[:, 1].astype(np.float32) - mario_col)
            # Nur Gegner auf oder knapp unter Marios Hoehe zaehlen als
            # Gefahr. Ist Mario hoeher (row_diff < 0, z.B. durch Sprung),
            # gilt das nicht als Gefahr - das ist der erwuenschte Weg,
            # einem Gegner auszuweichen oder ihn zu besiegen.
            same_level = (row_diff >= 0) & (row_diff <= 1)
            if np.any(same_level):
                nearest = float(col_diff[same_level].min())
                if nearest <= self._enemy_danger_radius:
                    reward -= self._enemy_proximity_penalty * (self._enemy_danger_radius - nearest + 1)

        # --- Gezieltes Springen: nur dann kein Kostenpflicht/Bonus, wenn --
        # --- tatsaechlich ein Gegner, Hindernis oder eine Schlucht voraus ist
        threat_ahead = False
        if mario_col_max is not None:
            n_cols = area.shape[1]
            lookahead_cols = range(
                mario_col_max + 1, min(mario_col_max + 1 + self._jump_lookahead, n_cols)
            )

            # Bodenreihe direkt unter Mario merken, sobald er sichtbar auf
            # einer festen Kachel steht - das ist die Referenz fuer die
            # Schlucht-Erkennung, auch waehrend er gerade in der Luft ist
            # (die Bodenlinie selbst aendert sich beim Springen ja nicht).
            n_rows = area.shape[0]
            below_row = mario_row_bottom + 1
            if below_row < n_rows:
                col_idx = min(max(int(round(mario_col)), 0), area.shape[1] - 1)
                below_cell = int(area[below_row, col_idx])
                if below_cell != 0 and below_cell != self._mario_category:
                    self._ground_row = below_row

            if lookahead_cols:
                cols = list(lookahead_cols)

                # Gegner auf Marios Hoehe, VORAUS (nicht hinter ihm) und
                # innerhalb der Vorschau-Distanz.
                if enemy_cells.size:
                    e_row_diff = enemy_cells[:, 0].astype(np.int32) - mario_row_top
                    e_col_diff = enemy_cells[:, 1].astype(np.float32) - mario_col
                    ahead_enemy = (
                        (e_row_diff >= 0) & (e_row_diff <= 1)
                        & (e_col_diff > 0) & (e_col_diff <= self._jump_lookahead)
                    )
                    if np.any(ahead_enemy):
                        threat_ahead = True

                # Festes Hindernis (z.B. Pfeife/Block) auf Marios eigener
                # Hoehe voraus - jede nicht-leere, nicht-Mario-Kachel in
                # Marios Zeilenspanne zaehlt.
                if not threat_ahead:
                    row_span = area[mario_row_top : mario_row_bottom + 1, cols]
                    solid = (row_span != 0) & (row_span != self._mario_category)
                    if np.any(solid):
                        threat_ahead = True

                # Schlucht: in der zuletzt bekannten Bodenreihe sind alle
                # Vorschau-Spalten leer (keine Kachel = kein Boden).
                if not threat_ahead and self._ground_row is not None and self._ground_row < n_rows:
                    ground_span = area[self._ground_row, cols]
                    if np.all(ground_span == 0):
                        threat_ahead = True

        # Kleine Kosten fuers Springen OHNE erkennbaren Grund voraus (siehe
        # Modul-Docstring): ohne das ist A-gedrueckt-halten reward-neutral
        # bis reward-positiv und der Agent lernt staendiges Hoppeln statt
        # normal zu laufen. Ist tatsaechlich etwas voraus, entfaellt die
        # Strafe und es gibt stattdessen einen kleinen Bonus. Nur EIN fixer
        # Betrag, egal ob A allein oder in Kombination (Aktionen 2/4/5) gehalten wird.
        jumping = WindowEvent.PRESS_BUTTON_A in self._held_buttons
        if jumping and threat_ahead:
            self._consecutive_jump_ahead_steps += 1
        else:
            self._consecutive_jump_ahead_steps = 0

        if jumping:
            if threat_ahead:
                reward += self._reward_jump_bonus
                # Manche Hindernisse (z.B. hohe Pfeifen) brauchen einen
                # laenger gehaltenen, hohen Sprung statt nur einen kurzen
                # Hueper - die Sprunghoehe haengt in Super Mario Land direkt
                # von der Haltedauer von A ab (siehe Bugfix "hohe Sprünge"
                # oben). Da sich "wie hoch ist dieses Hindernis" aus dem
                # Kachelraster nicht zuverlaessig messen liess (normales
                # flaches Terrain und echte Hindernisse waren dabei nicht
                # sauber unterscheidbar, siehe Projekt-Notizen), wird
                # stattdessen direkt das laengere DURCHGEHENDE Halten von A
                # waehrend ein Grund voraus ist belohnt - das ist unabhaengig
                # von der Hindernishoehe immer die richtige Strategie, um
                # ueberhaupt eine hoehere Sprungoption auszuprobieren. Nur
                # EIN einmaliger Bonus beim Erreichen der Schwelle (nicht pro
                # weiterem Schritt danach), damit blosses Dauerhalten nicht
                # zusaetzlich belohnt wird.
                if self._consecutive_jump_ahead_steps == self._sustained_jump_steps:
                    reward += self._reward_sustained_jump_bonus
            else:
                reward -= self._reward_jump_cost

        # Bonus: Ein Gegner ist verschwunden UND der Score ist im selben
        # Schritt gestiegen -> sehr wahrscheinlich besiegt (draufgesprungen
        # oder mit Feuerball getroffen), nicht nur aus dem Bild gelaufen.
        if enemy_count < self._last_enemy_count and score > self._last_score:
            reward += self._enemy_defeat_bonus
        self._last_enemy_count = enemy_count

        terminated = False
        if lives < self._last_lives or self.game_wrapper.game_over():
            reward -= self._reward_death
            terminated = True

        self._last_progress = progress
        self._last_lives = lives
        self._last_score = score
        self._last_coins = coins

        return reward, terminated