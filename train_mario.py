"""
Trainiert einen PPO-Agenten (stable-baselines3) auf Super Mario Land.

Aufruf:
    python train_mario.py PATH/ZUR/ROM.gb [--steps 500000] [--envs 4]
    python train_mario.py PATH/ZUR/ROM.gb --config andere_config.yaml

Alle Hyperparameter (Environment, Belohnung, PPO, Trainingsablauf) stehen
gesammelt in config.yaml - Kommandozeilen-Optionen wie --steps ueberschreiben
bei Bedarf einzelne Werte daraus, ohne die Datei aendern zu muessen.

Zwischenspeicherstaende landen unter checkpoints/, Tensorboard-Logs unter
runs/ (dort mit "tensorboard --logdir runs" anschaubar).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from config import load_config
from mario_env_CNN import MarioLandEnv


def make_env(rom_path: str, config: dict):
    def _init():
        env = MarioLandEnv.from_config(rom_path, config, headless=True)
        return Monitor(env)

    return _init


def main():
    cfg = load_config()
    training_cfg = cfg.get("training", {}) or {}
    ppo_cfg = training_cfg.get("ppo", {}) or {}

    parser = argparse.ArgumentParser()
    parser.add_argument("rom", help="Pfad zur Super-Mario-Land-.gb-Datei")
    parser.add_argument("--config", default=None, help="Pfad zu einer alternativen config.yaml")
    parser.add_argument("--steps", type=int, default=training_cfg.get("steps", 500_000), help="Trainings-Zeitschritte gesamt")
    parser.add_argument("--envs", type=int, default=training_cfg.get("envs", 4), help="Anzahl paralleler Emulator-Instanzen")
    parser.add_argument("--frame-skip", type=int, default=(cfg.get("env", {}) or {}).get("frame_skip", 4))
    parser.add_argument("--checkpoint-every", type=int, default=training_cfg.get("checkpoint_every", 25_000))
    parser.add_argument("--run-name", default=training_cfg.get("run_name", "mario_ppo"))
    parser.add_argument("--resume", default=None, help="Pfad zu einem .zip-Checkpoint, um weiterzutrainieren")
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        training_cfg = cfg.get("training", {}) or {}
        ppo_cfg = training_cfg.get("ppo", {}) or {}

    # --frame-skip auf der Kommandozeile hat Vorrang vor der config.yaml.
    env_cfg = dict(cfg.get("env", {}) or {})
    env_cfg["frame_skip"] = args.frame_skip
    run_cfg = {**cfg, "env": env_cfg}

    checkpoints_dir = Path("checkpoints") / args.run_name
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("runs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env(args.rom, run_cfg) for _ in range(args.envs)]
    vec_env = DummyVecEnv(env_fns) if args.envs == 1 else SubprocVecEnv(env_fns)

    if args.resume:
        print(f"Setze Training fort von: {args.resume}")
        model = PPO.load(args.resume, env=vec_env, tensorboard_log=str(logs_dir))
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            n_steps=ppo_cfg.get("n_steps", 512),
            batch_size=ppo_cfg.get("batch_size", 256),
            learning_rate=ppo_cfg.get("learning_rate", 0.00005),
            gamma=ppo_cfg.get("gamma", 0.99999),
            ent_coef=ppo_cfg.get("ent_coef", 0.01),
            tensorboard_log=str(logs_dir),
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_every // args.envs, 1),
        save_path=str(checkpoints_dir),
        name_prefix="mario",
    )

    print(f"Training startet: {args.steps} Schritte, {args.envs} parallele Umgebungen")
    model.learn(total_timesteps=args.steps, callback=checkpoint_callback, tb_log_name=args.run_name)

    final_path = checkpoints_dir / "mario_final.zip"
    model.save(str(final_path))
    print(f"Fertig. Modell gespeichert unter: {final_path}")


if __name__ == "__main__":
    main()
