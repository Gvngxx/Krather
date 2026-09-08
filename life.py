import json
import math
import os
import pickle
import random
import threading
import time
import zlib
from copy import deepcopy
import numpy as np

from cryptography.fernet import Fernet, InvalidToken

import proc

STATE_PATH = "learning_state.bin"
LEGACY_STATE_PATH = "learning_state.json"
KEY_PATH = "learning_state.key"
STATE_MAGIC = b"LIFEENC1\n"
TRAJECTORY_LIMIT = 300
MEMORY_LIMIT = 120


class ExperienceMemory:
    def __init__(self, capacity=MEMORY_LIMIT):
        self.capacity = int(capacity)
        self.entries = []

    def remember(self, sensors, action, reward, result, timestamp):
        experience = {
            "sensors": {key: float(value) for key, value in sensors.items()},
            "action": {key: float(value) for key, value in action.items()},
            "reward": float(reward),
            "result": dict(result),
            "timestamp": int(timestamp),
        }
        self.entries.append(experience)
        self.entries = self.entries[-self.capacity:]
        return experience

    def recall(self, limit=5):
        return deepcopy(self.entries[-max(0, int(limit)):])

    def clear_memory(self):
        self.entries.clear()

    def serialize_memory(self):
        return {"capacity": self.capacity, "entries": self.recall(self.capacity)}

    def restore_memory(self, payload):
        if not isinstance(payload, dict):
            return
        self.capacity = max(1, int(payload.get("capacity", self.capacity)))
        self.entries = list(payload.get("entries", []))[-self.capacity:]

    def reward_baseline(self, limit=20):
        rewards = [entry["reward"] for entry in self.entries[-limit:]]
        return sum(rewards) / len(rewards) if rewards else 0.0


class Organism:
    def __init__(self, x=640.0, y=260.0, organism_id="A"):
        self.organism_id = organism_id
        self.start_x = float(x)
        self.start_y = float(y)
        self.reset()

    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.energy = 100.0
        self.size = 18.0
        self.weight = 1.0
        self.age = 0
        self.no_food_ticks = 0
        self.food_eaten = 0
        self.alive = True

    def get_sensors(self, food, width, height):
        food_dx = (food["x"] - self.x) / width
        food_dy = (food["y"] - self.y) / height
        distance = math.hypot(food["x"] - self.x, food["y"] - self.y)
        return {
            "x": self.x / width,
            "y": self.y / height,
            "food_x": food["x"] / width,
            "food_y": food["y"] / height,
            "food_dx": food_dx,
            "food_dy": food_dy,
            "distance": distance / math.hypot(width, height),
            "energy": self.energy / 160.0,
            "size": self.size / 34.0,
        }

    def move(self, action, width, height):
        previous = (self.x, self.y)
        speed = 10.0 + min(8.0, self.energy / 20.0)
        half_size = self.size / 2.0
        self.x = min(max(self.x + float(action["move_x"]) * speed, half_size), width - half_size)
        self.y = min(max(self.y + float(action["move_y"]) * speed, half_size), height - half_size)
        moved = math.hypot(self.x - previous[0], self.y - previous[1])
        self.energy -= 0.22 + moved * 0.035
        return moved

    def to_dict(self):
        return {
            "id": self.organism_id,
            "x": self.x,
            "y": self.y,
            "energy": self.energy,
            "size": self.size,
            "weight": self.weight,
            "age": self.age,
            "no_food_ticks": self.no_food_ticks,
            "food_eaten": self.food_eaten,
            "alive": self.alive,
        }


class LifeSimulation:
    def __init__(self, width=860, height=520):
        self.width = width
        self.height = height
        self.lock = threading.RLock()
        self.running = False
        self.generation = 1
        self.tick = 0
        self.food = {"x": self.width / 2.0, "y": self.height / 2.0}
        self.organisms = [Organism()]
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.last_action = {"move_x": 0.0, "move_y": 0.0}
        self.last_brain = self._empty_brain()
        self.trajectory = []
        self.memory = ExperienceMemory()
        self._legacy_state_loaded = False
        self.learning = self._load_state()
        self._restore_state(self.learning)
        if self._legacy_state_loaded:
            self._save_state()
        self.thread = threading.Thread(target=self._run, name="life-simulation", daemon=True)
        self.thread.start()

    @property
    def organism(self):
        return self.organisms[0]

    def remember(self, sensors, action, reward, result, timestamp=None):
        return self.memory.remember(sensors, action, reward, result, self.tick if timestamp is None else timestamp)

    def recall(self, limit=5):
        return self.memory.recall(limit)

    def clear_memory(self):
        self.memory.clear_memory()

    def serialize_memory(self):
        return self.memory.serialize_memory()

    def restore_memory(self, payload):
        self.memory.restore_memory(payload)

    def _empty_brain(self):
        return {"tick": 0, "neurons": [], "connections": [], "steps": [], "outputs": dict(self.last_action), "final_output": 0.0, "sensors": {}}

    def _get_cipher(self):
        configured_key = os.environ.get("LIFE_STATE_KEY")
        if configured_key:
            return Fernet(configured_key.encode("ascii"))

        try:
            with open(KEY_PATH, "rb") as stream:
                key = stream.read().strip()
        except FileNotFoundError:
            key = Fernet.generate_key()
            descriptor = os.open(KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(key)
            except BaseException:
                os.close(descriptor)
                raise
        os.chmod(KEY_PATH, 0o600)
        return Fernet(key)

    def _decode_state(self, encrypted_state):
        if not encrypted_state.startswith(STATE_MAGIC):
            raise ValueError("Formato de estado cifrado no reconocido")
        encrypted_payload = encrypted_state[len(STATE_MAGIC):]
        compressed_payload = self._get_cipher().decrypt(encrypted_payload)
        return pickle.loads(zlib.decompress(compressed_payload))

    def _load_state(self):
        try:
            with open(STATE_PATH, "rb") as stream:
                data = self._decode_state(stream.read())
                if isinstance(data, dict):
                    return data
        except (OSError, ValueError, InvalidToken, EOFError, pickle.UnpicklingError, zlib.error):
            pass

        try:
            with open(LEGACY_STATE_PATH, "r", encoding="utf-8") as stream:
                data = json.load(stream)
                if isinstance(data, dict):
                    self._legacy_state_loaded = True
                    return data
        except (OSError, json.JSONDecodeError):
            pass
        return {"generations": [], "total_ticks": 0, "total_reward": 0.0}

    def _restore_state(self, saved):
        brain_state = saved.get("brain_state") or {}
        if brain_state:
            proc.restore_brain(brain_state)
            proc.restore_learning(saved.get("learning_stats", {}))
            self.last_brain = deepcopy(saved.get("last_brain", self._empty_brain()))
        world = saved.get("world_state") or {}
        if not world:
            return
        self.generation = int(world.get("generation", self.generation))
        self.tick = int(world.get("tick", self.tick))
        self.food = dict(world.get("food", self.food))
        self.last_reward = float(world.get("reward", 0.0))
        self.total_reward = float(world.get("total_reward", 0.0))
        self.last_action = dict(world.get("last_action", self.last_action))
        self.trajectory = list(world.get("trajectory", []))[-TRAJECTORY_LIMIT:]
        self.memory.restore_memory(saved.get("memory", {}))
        saved_organisms = world.get("organisms", [])
        if saved_organisms:
            self.organisms = []
            for saved_organism in saved_organisms:
                organism = Organism(saved_organism.get("x", 640), saved_organism.get("y", 260), saved_organism.get("id", "A"))
                for key, value in saved_organism.items():
                    if hasattr(organism, key) and key not in {"start_x", "start_y"}:
                        setattr(organism, key, value)
                self.organisms.append(organism)

    def _save_state(self):
        self.learning["brain_state"] = proc.serialize_brain()
        self.learning["learning_stats"] = proc.serialize_learning()
        self.learning["last_brain"] = deepcopy(self.last_brain)
        self.learning["world_state"] = {
            "generation": self.generation,
            "tick": self.tick,
            "food": dict(self.food),
            "organisms": [organism.to_dict() for organism in self.organisms],
            "reward": self.last_reward,
            "total_reward": self.total_reward,
            "last_action": dict(self.last_action),
            "trajectory": self.trajectory[-TRAJECTORY_LIMIT:],
        }
        self.learning["memory"] = self.memory.serialize_memory()
        serialized_state = pickle.dumps(self.learning, protocol=pickle.HIGHEST_PROTOCOL)
        encrypted_state = STATE_MAGIC + self._get_cipher().encrypt(zlib.compress(serialized_state, level=9))
        temporary_path = f"{STATE_PATH}.tmp"
        with open(temporary_path, "wb") as stream:
            stream.write(encrypted_state)
        os.replace(temporary_path, STATE_PATH)
        if self._legacy_state_loaded and os.path.exists(LEGACY_STATE_PATH):
            os.replace(LEGACY_STATE_PATH, f"{LEGACY_STATE_PATH}.legacy")
            self._legacy_state_loaded = False

    def _run(self):
        while True:
            with self.lock:
                should_tick = self.running and any(organism.alive for organism in self.organisms)
                delay = random.uniform(0.08, 0.30)
            if should_tick:
                self.step()
            time.sleep(delay)

    def step(self):
        with self.lock:
            if not self.running or not any(organism.alive for organism in self.organisms):
                return self.snapshot()
            primary_brain = None
            tick_reward = 0.0
            for organism in list(self.organisms):
                if not organism.alive:
                    continue
                sensors = organism.get_sensors(self.food, self.width, self.height)
                brain = proc.process_sensors(sensors, "TANH")
                before_position = (organism.x, organism.y)
                before_distance = math.hypot(self.food["x"] - organism.x, self.food["y"] - organism.y)
                moved = organism.move(brain["outputs"], self.width, self.height)
                actual_move = {
                    "move_x": (organism.x - before_position[0]) / max(1.0, moved),
                    "move_y": (organism.y - before_position[1]) / max(1.0, moved),
                }
                after_distance = math.hypot(self.food["x"] - organism.x, self.food["y"] - organism.y)
                distance_progress = before_distance - after_distance
                # Reward basado únicamente en el progreso real hacia la comida.
                # Acercarse = recompensa positiva progresiva.
                # Alejarse = penalización progresiva.
                # No avanzar hacia la comida = 0 reward.
                if distance_progress > 0:
                    # Mientras más distancia recorra hacia la comida,
                    # mayor recompensa, hasta un máximo de +0.20 20 de rewards.
                    reward = min(0.20, max(0.01, distance_progress / 0.50))
                elif distance_progress < 0:
                    # Alejarse de la comida produce una penalización.
                    # Cuanto más se aleje, mayor será la penalización,
                    # hasta un máximo de -0.90 90 de penalisacion.
                    reward = max(-0.90, min(-0.05, distance_progress / 0.20))
                else:
                    reward = 0.0
                ate_food = after_distance <= organism.size + 8.0
                if ate_food:
                    # Reward +20 despues de comer
                    reward += 2.00
                    organism.energy = min(160.0, organism.energy + 42.0)
                    organism.size = min(34.0, organism.size + 1.5)
                    organism.weight += 0.08
                    organism.no_food_ticks = 0
                    organism.food_eaten += 1
                    self.food = self._new_food(organism)
                else:
                    organism.no_food_ticks += 1
                    if organism.no_food_ticks % 20 == 0:
                        organism.size = max(10.0, organism.size - 1.0)
                organism.age += 1
                terminal = False
                if organism.no_food_ticks > 180 or organism.energy <= 0:
                    organism.energy = max(0.0, organism.energy)
                    organism.alive = False
                    reward = -4.00
                    terminal = True
                memory_baseline = self.memory.reward_baseline()
                learning_reward = float(np.clip(reward, -4.00, 2.00))
                brain.update({"moved": moved, "reward": reward, "ate_food": ate_food, "organism_id": organism.organism_id})
                self.memory.remember(
                    sensors,
                    brain["outputs"],
                    reward,
                    {
                        "moved": moved,
                        "ate_food": ate_food,
                        "alive": organism.alive,
                        "energy": organism.energy,
                        "distance_progress": distance_progress,
                    },
                    self.tick + 1,
                )
                learning_update = proc.learn_from_reward(learning_reward, terminal=terminal)
                brain["learning_update"] = learning_update
                brain["actual_move"] = actual_move
                if organism is self.organism:
                    primary_brain = brain
                tick_reward += reward

            self._try_reproduce()
            self.tick += 1
            self.last_reward = tick_reward
            self.total_reward += tick_reward
            if primary_brain:
                self.last_action = dict(primary_brain["actual_move"])
                primary_brain["tick"] = self.tick
                self.last_brain = primary_brain
            self.trajectory.append({
                "t": self.tick,
                "o": [[round(organism.x, 1), round(organism.y, 1), organism.alive] for organism in self.organisms],
                "f": [round(self.food["x"], 1), round(self.food["y"], 1)],
                "r": round(tick_reward, 3),
            })
            self.trajectory = self.trajectory[-TRAJECTORY_LIMIT:]
            if not any(organism.alive for organism in self.organisms):
                self.running = False
            self.learning["total_ticks"] = int(self.learning.get("total_ticks", 0)) + 1
            self.learning["total_reward"] = float(self.learning.get("total_reward", 0.0)) + tick_reward
            self.learning["last_tick"] = self.tick
            self.learning["last_reward"] = tick_reward
            self._save_state()
            return self.snapshot()

    def _try_reproduce(self):
        parent = self.organism
        if parent.food_eaten < 3 or parent.food_eaten % 3 != 0 or parent.energy < 125 or len(self.organisms) >= 4:
            return
        parent.energy -= 70.0
        parent.size = max(10.0, parent.size * 0.62)
        child = Organism(parent.x + random.uniform(-35, 35), parent.y + random.uniform(-35, 35), f"G{self.generation}-{len(self.organisms) + 1}")
        child.energy = 45.0
        child.size = parent.size
        child.weight = parent.weight
        self.organisms.append(child)
        self.learning.setdefault("brain_copies", []).append(deepcopy(proc.serialize_brain()))

    def _new_food(self, organism):
        for _ in range(30):
            food = {"x": float(random.randint(30, self.width - 30)), "y": float(random.randint(30, self.height - 30))}
            if math.hypot(food["x"] - organism.x, food["y"] - organism.y) >= 150:
                return food
        return {"x": 80.0, "y": 80.0}

    def start(self):
        with self.lock:
            if any(organism.alive for organism in self.organisms):
                self.running = True
                self._save_state()

    def stop(self):
        with self.lock:
            self.running = False
            self._save_state()

    def reset(self):
        with self.lock:
            self.running = False
            self.organisms = [Organism()]
            self.food = {"x": self.width / 2.0, "y": self.height / 2.0}
            self.tick = 0
            self.last_reward = 0.0
            self.total_reward = 0.0
            self.last_action = {"move_x": 0.0, "move_y": 0.0}
            self.last_brain = self._empty_brain()
            self.trajectory = []
            self.memory.clear_memory()
            self._save_state()

    def new_generation(self):
        with self.lock:
            self.learning.setdefault("generations", []).append({"generation": self.generation, "ticks": self.tick, "reward": self.total_reward, "food_eaten": self.organism.food_eaten})
            self.generation += 1
            self.reset()

    def snapshot(self):
        with self.lock:
            return {
                "running": self.running,
                "generation": self.generation,
                "tick": self.tick,
                "organism": self.organism.to_dict(),
                "organisms": [organism.to_dict() for organism in self.organisms],
                "food": dict(self.food),
                "reward": self.last_reward,
                "total_reward": self.total_reward,
                "food_eaten": sum(organism.food_eaten for organism in self.organisms),
                "last_action": dict(self.last_action),
                "brain": deepcopy(self.last_brain),
                "trajectory_length": len(self.trajectory),
                "learning": {"total_ticks": self.learning.get("total_ticks", 0), "total_reward": self.learning.get("total_reward", 0.0), "generations": len(self.learning.get("generations", [])), "brain_copies": len(self.learning.get("brain_copies", [])), **proc.serialize_learning()},
                "memory": {
                    "count": len(self.memory.entries),
                    "capacity": self.memory.capacity,
                    "used_ratio": len(self.memory.entries) / self.memory.capacity,
                    "last": self.memory.recall(1)[0] if self.memory.entries else None,
                    "recent": self.memory.recall(5),
                },
            }


simulation = LifeSimulation()
