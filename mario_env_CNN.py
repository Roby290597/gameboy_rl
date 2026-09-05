"""
Experimentelle Variante von mario_env.py mit CNN-tauglicher Beobachtung,
inspiriert von der Modellarchitektur aus dem PyTorch-Mario-RL-Tutorial
(https://github.com/yuansongFeng/MadMario/), das der Nutzer als Jupyter-
Notebook hochgeladen hat.

WICHTIG - was hier UEBERNOMMEN und was NICHT uebernommen wurde:

Das Tutorial nutzt DDQN (Deep Q-Learning) mit eigener Infrastruktur
(Replay-Buffer, Epsilon-Greedy-Exploration, Online-/Target-Netz-Sync) auf
rohen, gestapelten 84x84-Graustufen-Pixel-Frames. Das komplett zu
uebernehmen wuerde unser PPO-Training (stable-baselines3), das config.yaml-
System und die gesamte Reward-Arbeit in mario_env.py (Gegner-Erkennung,
gezieltes Springen, Sustained-Jump-Bonus) verwerfen - siehe Projekt-Notizen
vom 2026-09-03 fuer die ausfuehrliche Begruendung.

Uebernommen wird NUR die Modellarchitektur-Idee: ein kleines CNN
("3x (Conv2d + ReLU) -> Flatten -> Dense + ReLU") statt eines flachen MLP,
damit das Netz raeumliche Muster im Kachelraster erkennen kann (z.B. "Kachel
X liegt direkt ueber/neben Kachel Y"), was beim Plaetten zu einem 320-Werte-
Vektor (wie in mario_env.py) komplett verloren geht.

Zwei Anpassungen waren dafuer noetig, weil unser Zustand kein 84x84-Pixel-
Bild ist, sondern PyBoys semantisches 16x20-Kachelraster (Kategorie-Zahlen,
kein Graustufenwert):

1. One-Hot statt Graustufen: Kategorie-Zahlen sind NOMINAL (Kategorie 8 ist
   nicht "mehr" als Kategorie 3), anders als ein Pixel-Grauwert. Ein rohes
   CNN direkt auf den Kategorie-Zahlen wuerde faelschlich eine Ordnung
   zwischen ihnen annehmen. Deshalb wird das Kachelraster hier zu einem
   One-Hot-Tensor der Form (Anzahl_Kategorien, 16, 20) - ein eigener 0/1-
   Kanal pro Kategorie, statt (16, 20) mit rohen Zahlen oder (320,) geplaettet.
2. Kleinere Kernel/Stride: Die Original-Architektur nutzt Kernel-Groessen
   8/4/3 mit Stride 4/2/1, ausgelegt auf 84x84-Bilder - auf unserem winzigen
   16x20-Raster wuerde das die raeumliche Aufloesung sofort auf (fast) nichts
   reduzieren. Hier stattdessen Kernel 3, Stride 1, Padding 1 in allen drei
   Conv-Schichten (Aufloesung bleibt 16x20 erhalten) - gleiche "Form" der
   Architektur (3x Conv+ReLU, dann Flatten, dann Dense+ReLU auf 512), nur
   auf die tatsaechliche Eingabegroesse zugeschnitten.

MarioLandEnv2 erbt ansonsten ALLES von MarioLandEnv (Aktionen, Reward-Logik
inkl. gezieltem Springen/Sustained-Jump-Bonus/Gegner-Erkennung) unveraendert
- nur observation_space und _get_obs() sind ueberschrieben. from_config()
muss nicht ueberschrieben werden, da die geerbte Methode intern `cls(...)`
verwendet und dadurch automatisch eine MarioLandEnv2-Instanz erzeugt, wenn
man sie als MarioLandEnv2.from_config(...) aufruft.

Verwendung mit stable-baselines3 (PPO):

    from mario_env2 import MarioLandEnv2, MarioCNNExtractor, CNN_POLICY_KWARGS
    from stable_baselines3 import PPO
    from config import load_config

    cfg = load_config()
    env = MarioLandEnv2.from_config("Super_Mario_Land_World_Rev_1.gb", cfg, headless=True)
    model = PPO("CnnPolicy", env, policy_kwargs=CNN_POLICY_KWARGS, verbose=1)
    model.learn(total_timesteps=100_000)

Diese Datei ist bewusst eigenstaendig (mario_env.py bleibt unveraendert) -
ein Experiment, das erst dann relevant wird, falls das gezielte Springen mit
dem normalen MLP das Huerden-Problem nicht ausreichend loest (siehe
Projekt-Notizen, Stand 2026-09-03: Nutzer trainiert zunaechst mit dem
bestehenden MLP-Setup neu, CNN-Umbau folgt nur bei Bedarf).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from mario_env import MarioLandEnv


class MarioLandEnv2(MarioLandEnv):
    """Wie MarioLandEnv, aber mit One-Hot-kodierter (Kategorien, 16, 20)-
    Beobachtung statt geplättetem (320,)-Vektor - fürs Training mit einem
    CNN (siehe MarioCNNExtractor unten) statt eines flachen MLP.

    Reward-Logik, Aktionen und alles andere sind unverändert von
    MarioLandEnv geerbt.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Anzahl der Kategorien in der aktiven mapping_compressed-Tabelle
        # (0 = leer, bis zum groessten vorkommenden Kategorie-Wert). Wird
        # aus der Tabelle selbst bestimmt statt hart codiert, bleibt aber
        # ueber eine Episode/den ganzen Lauf hinweg konstant (die Tabelle
        # aendert sich zur Laufzeit nicht).
        self._num_categories = int(np.asarray(self.game_wrapper.mapping_compressed).max()) + 1

        # Kanal-zuerst (Kategorien, 16, 20), wie von PyTorch-CNNs erwartet.
        # uint8/0-1 statt 0-255, siehe MarioCNNExtractor (normalize_images=False).
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self._num_categories, 16, 20), dtype=np.uint8
        )

    def _get_obs(self, area: np.ndarray | None = None) -> np.ndarray:
        if area is None:
            area = np.asarray(self.game_wrapper.game_area(), dtype=np.uint8)
        # One-Hot: (16, 20, Kategorien) via Fancy-Indexing in die
        # Einheitsmatrix, dann auf Kanal-zuerst (Kategorien, 16, 20) bringen.
        one_hot = np.eye(self._num_categories, dtype=np.uint8)[area]
        return np.transpose(one_hot, (2, 0, 1))


class MarioCNNExtractor(BaseFeaturesExtractor):
    """Kleines CNN im 'Geist' der MarioNet-Architektur aus dem PyTorch-
    Tutorial (3x Conv2d+ReLU -> Flatten -> Linear+ReLU), aber mit Kernel 3 /
    Stride 1 / Padding 1 statt 8/4-4/2-3/1, weil unsere Eingabe 16x20 statt
    84x84 ist (siehe Modul-Docstring fuer die ausfuehrliche Begruendung).
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Flatten-Groesse einmalig per Dummy-Forward-Pass bestimmen (SB3-
        # ueblich), statt sie von Hand auszurechnen.
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations.float()))


# Fertige policy_kwargs zum direkten Durchreichen an PPO(..., policy_kwargs=...).
# normalize_images=False ist wichtig: SB3 teilt bei Bild-Beobachtungen sonst
# standardmaessig durch 255 (fuer normale 0-255-Pixelwerte) - unsere One-Hot-
# Werte sind aber bereits 0/1, das Teilen durch 255 waere hier falsch.
CNN_POLICY_KWARGS = dict(
    features_extractor_class=MarioCNNExtractor,
    features_extractor_kwargs=dict(features_dim=512),
    normalize_images=False,
)