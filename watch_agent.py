"""
Zeigt einen trainierten Agenten in einem sichtbaren Fenster beim Spielen zu,
oder (ohne --model) laesst per Zufallsaktionen einfach nur die Umgebung
laufen - nuetzlich, um zu pruefen, dass alles funktioniert.
 
Aufruf:
    python watch_agent.py PATH/ZUR/ROM.gb --model checkpoints/mario_ppo/mario_final.zip
    python watch_agent.py PATH/ZUR/ROM.gb                 # nur Zufallsaktionen, zum Testen
    python watch_agent.py PATH/ZUR/ROM.gb --model ... --speed 3   # 3x Geschwindigkeit
    python watch_agent.py PATH/ZUR/ROM.gb --model ... --speed 0   # ungebremst, so schnell wie moeglich
 
Vorspulen geht auch jederzeit live im Fenster: Leertaste druecken schaltet
zwischen normalem Tempo und ungebremstem Tempo um (Umschalter, nicht halten).
"""
 
from __future__ import annotations
 
import argparse
 
from mario_env import MarioLandEnv, ACTION_NAMES
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rom", help="Pfad zur Super-Mario-Land-.gb-Datei")
    parser.add_argument("--model", default=None, help="Pfad zu einem trainierten PPO-Modell (.zip)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--speed",
        type=int,
        default=1,
        help="Emulationsgeschwindigkeit als Vielfaches von Echtzeit (1=normal, 2=doppelt, "
        "0=ungebremst/so schnell wie moeglich). Max. 5. Im Fenster jederzeit mit "
        "der Leertaste umschaltbar.",
    )
    args = parser.parse_args()
 
    env = MarioLandEnv(args.rom, headless=False, frame_skip=4)
    env.pyboy.set_emulation_speed(args.speed)
 
    model = None
    if args.model:
        from stable_baselines3 import PPO
 
        model = PPO.load(args.model)
        print(f"Modell geladen: {args.model}")
    else:
        print("Kein Modell angegeben - spiele mit Zufallsaktionen (nur zum Testen der Umgebung).")
 
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
        print(
            f"Episode {ep + 1}: Belohnung={total_reward:.1f}  "
            f"Welt={info['world']}  Fortschritt={info['level_progress']}  "
            f"Leben={info['lives_left']}  Score={info['score']}"
        )
 
    env.close()
 
 
if __name__ == "__main__":
    main()