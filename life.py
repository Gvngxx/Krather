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
import pymunk

from cryptography.fernet import Fernet, InvalidToken

import proc

STATE_PATH = "learning_state.bin"
LEGACY_STATE_PATH = "learning_state.json"
KEY_PATH = "learning_state.key"
STATE_MAGIC = b"LIFEENC1\n"
TRAJECTORY_LIMIT = 300
MEMORY_LIMIT = 120
MEMORY_VISUAL_GAIN = 4.0
PHYSICS_DT = 1.0 / 30.0
RAYCAST_LENGTH = 220.0
ORGANISM_CATEGORY = 0x1
WORLD_CATEGORY = 0x2
FOOD_CATEGORY = 0x4


class ExperienceMemory:
    def __init__(self, capacity=MEMORY_LIMIT):
        self.capacity = int(capacity)
        self.entries = []

    def remember(self, sensors, action, reward, result, timestamp=None, generation=1, organism_id="A", encoded_input=0.0):
        experience = {
            "generation": int(generation),
            "organism_id": str(organism_id),
            "sensors": {key: float(value) for key, value in sensors.items()},
            "encoded_input": float(encoded_input),
            "action": {key: float(value) for key, value in action.items()},
            "reward": float(reward),
            "result": dict(result),
            "timestamp": int(time.time() if timestamp is None else timestamp),
        }
        for key in ("distance_before", "distance_after", "distance_progress", "energy", "neural_output"):
            if key in result:
                experience[key] = float(result[key])
        experience["food_eaten"] = bool(result.get("ate_food", False))
        experience["died"] = bool(result.get("died", False))
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

    def recall_relevant(self, sensors, action, reward, limit=12):
        """Find the closest recent experience using state, action, and reward."""
        if not self.entries:
            return 0.0, None
        current = {
            "food_dx": float(sensors.get("food_dx", 0.0)),
            "food_dy": float(sensors.get("food_dy", 0.0)),
            "distance": float(sensors.get("distance", 0.0)),
            "energy": float(sensors.get("energy", 0.0)),
            "move_x": float(action.get("move_x", 0.0)),
            "move_y": float(action.get("move_y", 0.0)),
            "reward": float(reward),
        }
        scales = {
            "food_dx": 2.0,
            "food_dy": 2.0,
            "distance": 1.0,
            "energy": 1.0,
            "move_x": 2.0,
            "move_y": 2.0,
            "reward": 40.0,
        }
        best_score = 0.0
        best_entry = None
        for entry in self.entries[-max(1, int(limit)):]:
            saved = entry.get("sensors", {})
            remembered_action = entry.get("action", {})
            remembered = {
                "food_dx": float(saved.get("food_dx", 0.0)),
                "food_dy": float(saved.get("food_dy", 0.0)),
                "distance": float(saved.get("distance", 0.0)),
                "energy": float(saved.get("energy", 0.0)),
                "move_x": float(remembered_action.get("move_x", 0.0)),
                "move_y": float(remembered_action.get("move_y", 0.0)),
                "reward": float(entry.get("reward", 0.0)),
            }
            normalized_difference = sum(
                min(1.0, abs(current[key] - remembered[key]) / scales[key])
                for key in current
            ) / len(current)
            score = float(np.clip(1.0 - normalized_difference, 0.0, 1.0))
            if score > best_score:
                best_score = score
                best_entry = entry
        return best_score, deepcopy(best_entry) if best_entry else None

    def recall_strength(self, sensors, action=None, reward=0.0, limit=12):
        score, _ = self.recall_relevant(sensors, action or {}, reward, limit)
        return score


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
        self.memory_state = {
            "signal": 0.0,
            "neural_signal": 0.0,
            "recall_strength": 0.0,
            "neurons": {node_id: 0.0 for node_id in ("N6", "N7", "N8", "N9")},
            "valence": "NEUTRAL",
            "latest_action": {"move_x": 0.0, "move_y": 0.0},
            "latest_result": {},
        }
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.physics_bodies = {}
        self.food_body = None
        self.food_shape = None
        self.wall_shapes = []
        self.auto_generation = False
        self.status = "STOPPED"
        self.next_generation_at = None
        self._legacy_state_loaded = False
        self.learning = self._load_state()
        self._restore_state(self.learning)
        self._rebuild_physics()
        if self._legacy_state_loaded:
            self._save_state()
        self.thread = threading.Thread(target=self._run, name="life-simulation", daemon=True)
        self.thread.start()

    @property
    def organism(self):
        return self.organisms[0]

    def remember(self, sensors, action, reward, result, timestamp=None):
        return self.memory.remember(
            sensors,
            action,
            reward,
            result,
            timestamp=timestamp,
            generation=self.generation,
            organism_id=result.get("organism_id", self.organism.organism_id),
            encoded_input=result.get("encoded_input", 0.0),
        )

    def recall(self, limit=5):
        return self.memory.recall(limit)

    def clear_memory(self):
        self.memory.clear_memory()

    def serialize_memory(self):
        return self.memory.serialize_memory()

    def restore_memory(self, payload):
        self.memory.restore_memory(payload)

    def _empty_brain(self):
        return {
            "tick": 0,
            "neurons": [],
            "connections": [],
            "steps": [],
            "outputs": dict(self.last_action),
            "final_output": 0.0,
            "sensors": {},
            "memory_signal": 0.0,
            "memory_neurons": {node_id: 0.0 for node_id in ("N6", "N7", "N8", "N9")},
        }

    def _rebuild_physics(self):
        """Recreate Pymunk bodies from the serializable simulation state."""
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.physics_bodies = {}
        self.food_body = None
        self.food_shape = None
        self.wall_shapes = []

        static_body = self.space.static_body
        boundary_segments = [
            pymunk.Segment(static_body, (0.0, 0.0), (self.width, 0.0), 1.0),
            pymunk.Segment(static_body, (self.width, 0.0), (self.width, self.height), 1.0),
            pymunk.Segment(static_body, (self.width, self.height), (0.0, self.height), 1.0),
            pymunk.Segment(static_body, (0.0, self.height), (0.0, 0.0), 1.0),
        ]
        for shape in boundary_segments:
            shape.collision_type = 2
            shape.filter = pymunk.ShapeFilter(categories=WORLD_CATEGORY, mask=ORGANISM_CATEGORY)
        self.wall_shapes = boundary_segments
        self.space.add(*boundary_segments)

        for organism in self.organisms:
            body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0.0, organism.size / 2.0))
            body.position = (organism.x, organism.y)
            body.velocity = (0.0, 0.0)
            shape = pymunk.Circle(body, organism.size / 2.0)
            shape.collision_type = 1
            shape.filter = pymunk.ShapeFilter(categories=ORGANISM_CATEGORY, mask=WORLD_CATEGORY | FOOD_CATEGORY)
            self.space.add(body, shape)
            self.physics_bodies[organism.organism_id] = (body, shape)

        self._rebuild_food_body()

    def _rebuild_food_body(self):
        if self.food_body is not None:
            if self.food_shape is not None:
                self.space.remove(self.food_shape)
            self.space.remove(self.food_body)
        self.food_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.food_body.position = (self.food["x"], self.food["y"])
        self.food_shape = pymunk.Circle(self.food_body, 12.0)
        self.food_shape.sensor = True
        self.food_shape.collision_type = 3
        self.food_shape.filter = pymunk.ShapeFilter(categories=FOOD_CATEGORY, mask=ORGANISM_CATEGORY)
        self.space.add(self.food_body, self.food_shape)

    def _raycast_sensors(self, organism):
        body_shape = self.physics_bodies[organism.organism_id][1]
        origin = pymunk.Vec2d(organism.x, organism.y)
        food_vector = pymunk.Vec2d(self.food["x"] - organism.x, self.food["y"] - organism.y)
        base_angle = math.atan2(food_vector.y, food_vector.x) if food_vector.length > 1e-6 else 0.0
        ray_angles = (-0.75, -0.375, 0.0, 0.375, 0.75)
        distances = []
        food_hits = []
        ray_filter = pymunk.ShapeFilter(mask=WORLD_CATEGORY | FOOD_CATEGORY)
        for offset in ray_angles:
            angle = base_angle + offset
            endpoint = origin + pymunk.Vec2d(math.cos(angle), math.sin(angle)) * RAYCAST_LENGTH
            hits = self.space.segment_query(origin, endpoint, 0.0, ray_filter)
            hit = next((item for item in hits if item.shape is not body_shape), None)
            distances.append(float(hit.alpha if hit else 1.0))
            food_hits.append(bool(hit and hit.shape is self.food_shape))
        return distances, food_hits

    def _physics_sensors(self, organism):
        sensors = organism.get_sensors(self.food, self.width, self.height)
        ray_distances, food_hits = self._raycast_sensors(organism)
        sensors["ray_front"] = ray_distances[2]
        sensors["ray_left"] = ray_distances[0]
        sensors["ray_right"] = ray_distances[4]
        sensors["ray_food_visible"] = 1.0 if any(food_hits) else 0.0
        return sensors

    def _move_with_physics(self, organism, action):
        body, shape = self.physics_bodies[organism.organism_id]
        previous = pymunk.Vec2d(*body.position)
        speed = 10.0 + min(8.0, organism.energy / 20.0)
        body.velocity = (
            float(action["move_x"]) * speed / PHYSICS_DT,
            float(action["move_y"]) * speed / PHYSICS_DT,
        )
        self.space.step(PHYSICS_DT)
        radius = float(shape.radius)
        clamped_x = min(max(float(body.position.x), radius), self.width - radius)
        clamped_y = min(max(float(body.position.y), radius), self.height - radius)
        if clamped_x != body.position.x or clamped_y != body.position.y:
            body.position = (clamped_x, clamped_y)
        body.velocity = (0.0, 0.0)
        organism.x, organism.y = body.position
        moved = (pymunk.Vec2d(*body.position) - previous).length
        organism.energy -= 0.22 + moved * 0.035
        return moved

    def _food_collision(self, organism):
        _, organism_shape = self.physics_bodies[organism.organism_id]
        return any(query.shape is self.food_shape for query in self.space.shape_query(organism_shape))

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
        self.auto_generation = bool(saved.get("auto_generation", False))
        self.memory.restore_memory(saved.get("memory", {}))
        if not world:
            return
        self.generation = int(world.get("generation", self.generation))
        self.tick = int(world.get("tick", self.tick))
        self.food = dict(world.get("food", self.food))
        self.last_reward = float(world.get("reward", 0.0))
        self.total_reward = float(world.get("total_reward", 0.0))
        self.last_action = dict(world.get("last_action", self.last_action))
        self.trajectory = list(world.get("trajectory", []))[-TRAJECTORY_LIMIT:]
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
        self.learning["auto_generation"] = self.auto_generation
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
                if self.next_generation_at is not None and self.auto_generation:
                    remaining = self.next_generation_at - time.monotonic()
                    if remaining <= 0:
                        self.create_new_generation()
                        remaining = 0.0
                    delay = min(0.15, max(0.02, remaining))
                    should_tick = False
                else:
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
                sensors = self._physics_sensors(organism)
                brain = proc.process_sensors(sensors, "TANH")
                neural_memory_signal = float(brain.get("memory_signal", 0.0))
                before_position = (organism.x, organism.y)
                before_distance = math.hypot(self.food["x"] - organism.x, self.food["y"] - organism.y)
                moved = self._move_with_physics(organism, brain["outputs"])
                actual_move = {
                    "move_x": (organism.x - before_position[0]) / max(1.0, moved),
                    "move_y": (organism.y - before_position[1]) / max(1.0, moved),
                }
                after_distance = math.hypot(self.food["x"] - organism.x, self.food["y"] - organism.y)
                distance_progress = before_distance - after_distance
                # Reward
                # No avanzar hacia la comida = 0 reward.
                MIN_PROGRESS = 1.5
                if distance_progress >= MIN_PROGRESS:
                    # Solo premiar avances suficientemente significativos.
                    reward = min(0.20, distance_progress / 5.0)

                elif distance_progress <= -MIN_PROGRESS:
                    # Alejarse de forma significativa sí genera castigo.
                    reward = max(-0.90, min(-0.05, distance_progress / 0.20))

                else:
                    # Movimiento prácticamente neutro: no recompensa ni castigo.
                    reward = 0.0

                ate_food = self._food_collision(organism)
                if ate_food:
                    # Reward +20 despues de comer
                    reward += 20.0
                    organism.energy = min(160.0, organism.energy + 42.0)
                    organism.size = min(34.0, organism.size + 1.5)
                    organism.weight += 0.08
                    organism.no_food_ticks = 0
                    organism.food_eaten += 1
                    self.food = self._new_food(organism)
                    self._rebuild_food_body()
                else:
                    organism.no_food_ticks += 1
                    if organism.no_food_ticks % 20 == 0:
                        organism.size = max(10.0, organism.size - 1.0)
                organism.age += 1
                terminal = False
                if organism.no_food_ticks > 180 or organism.energy <= 0:
                    organism.energy = max(0.0, organism.energy)
                    organism.alive = False
                    reward = -40.0
                    terminal = True
                memory_recall_strength, recalled_entry = self.memory.recall_relevant(
                    sensors,
                    brain["outputs"],
                    reward,
                )
                memory_signal = float(np.clip(
                    neural_memory_signal * memory_recall_strength,
                    0.0,
                    1.0,
                ))
                visual_memory_signal = float(np.clip(
                    memory_signal * MEMORY_VISUAL_GAIN,
                    0.0,
                    1.0,
                ))
                recalled_reward = float(recalled_entry.get("reward", 0.0)) if recalled_entry else 0.0
                recalled_valence = "POSITIVE" if recalled_reward > 0 else "NEGATIVE" if recalled_reward < 0 else "NEUTRAL"
                self.memory_state = {
                    "signal": visual_memory_signal,
                    "raw_signal": memory_signal,
                    "neural_signal": neural_memory_signal,
                    "recall_strength": memory_recall_strength,
                    "neurons": dict(brain.get("memory_neurons", {})),
                    "valence": recalled_valence,
                    "latest_action": dict(recalled_entry.get("action", {})) if recalled_entry else {"move_x": 0.0, "move_y": 0.0},
                    "latest_result": dict(recalled_entry.get("result", {})) if recalled_entry else {},
                }
                brain["memory_signal"] = memory_signal
                brain["visual_memory_signal"] = visual_memory_signal
                brain["memory_recall_strength"] = memory_recall_strength
                brain["memory_recall"] = recalled_entry
                memory_baseline = self.memory.reward_baseline()
                learning_reward = float(np.clip(reward, -40.00, 20.00))
                brain.update({"moved": moved, "reward": reward, "ate_food": ate_food, "organism_id": organism.organism_id})
                self.memory.remember(
                    sensors,
                    brain["outputs"],
                    reward,
                    {
                        "organism_id": organism.organism_id,
                        "generation": self.generation,
                        "encoded_input": brain.get("encoded_input", 0.0),
                        "moved": moved,
                        "ate_food": ate_food,
                        "alive": organism.alive,
                        "died": terminal,
                        "energy": organism.energy,
                        "distance_before": before_distance,
                        "distance_after": after_distance,
                        "distance_progress": distance_progress,
                        "neural_output": brain.get("final_output", 0.0),
                        "memory_reward_baseline": memory_baseline,
                    },
                    timestamp=int(time.time()),
                    generation=self.generation,
                    organism_id=organism.organism_id,
                    encoded_input=brain.get("encoded_input", 0.0),
                )
                learning_update = proc.learn_from_reward(learning_reward, terminal=terminal)
                brain["learning_update"] = learning_update
                brain["memory_context"] = {
                    "recent_count": len(self.memory.entries),
                    "reward_baseline": memory_baseline,
                    "recall_strength": memory_recall_strength,
                }
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
            else:
                self.memory_state = {
                    "signal": 0.0,
                    "neural_signal": 0.0,
                    "recall_strength": 0.0,
                    "neurons": {node_id: 0.0 for node_id in ("N6", "N7", "N8", "N9")},
                    "valence": "NEUTRAL",
                    "latest_action": {"move_x": 0.0, "move_y": 0.0},
                    "latest_result": {},
                }
            self.trajectory.append({
                "t": self.tick,
                "o": [[round(organism.x, 1), round(organism.y, 1), organism.alive] for organism in self.organisms],
                "f": [round(self.food["x"], 1), round(self.food["y"], 1)],
                "r": round(tick_reward, 3),
            })
            self.trajectory = self.trajectory[-TRAJECTORY_LIMIT:]
            if not any(organism.alive for organism in self.organisms):
                self.running = False
                self.handle_death()
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
        self._rebuild_physics()
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
                self.status = "RUNNING"
                self._save_state()

    def stop(self):
        with self.lock:
            self.running = False
            self.status = "STOPPED"
            self.memory_state["signal"] = 0.0
            self.memory_state["recall_strength"] = 0.0
            self._save_state()

    def reset(self):
        with self.lock:
            self.running = False
            self.next_generation_at = None
            self.status = "STOPPED"
            self.organisms = [Organism()]
            self.food = {"x": self.width / 2.0, "y": self.height / 2.0}
            self.tick = 0
            self.last_reward = 0.0
            self.total_reward = 0.0
            self.last_action = {"move_x": 0.0, "move_y": 0.0}
            self.last_brain = self._empty_brain()
            self.memory_state = {
                "signal": 0.0,
                "neural_signal": 0.0,
                "recall_strength": 0.0,
                "neurons": {node_id: 0.0 for node_id in ("N6", "N7", "N8", "N9")},
                "valence": "NEUTRAL",
                "latest_action": {"move_x": 0.0, "move_y": 0.0},
                "latest_result": {},
            }
            self.trajectory = []
            self._rebuild_physics()
            self._save_state()

    def new_generation(self):
        with self.lock:
            self.create_new_generation()

    def create_new_generation(self):
        self.learning.setdefault("generations", []).append({"generation": self.generation, "ticks": self.tick, "reward": self.total_reward, "food_eaten": self.organism.food_eaten})
        self.generation += 1
        self.next_generation_at = None
        self.organisms = [Organism()]
        self.food = {"x": self.width / 2.0, "y": self.height / 2.0}
        self.tick = 0
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.last_action = {"move_x": 0.0, "move_y": 0.0}
        self.last_brain = self._empty_brain()
        self.memory_state = {
            "signal": 0.0,
            "neural_signal": 0.0,
            "recall_strength": 0.0,
            "neurons": {node_id: 0.0 for node_id in ("N6", "N7", "N8", "N9")},
            "valence": "NEUTRAL",
            "latest_action": {"move_x": 0.0, "move_y": 0.0},
            "latest_result": {},
        }
        self.trajectory = []
        self._rebuild_physics()
        self.running = True
        self.status = "RUNNING"
        self._save_state()

    def handle_death(self):
        self.status = "GENERATION_COMPLETE" if self.auto_generation else "DEAD"
        self.memory_state["signal"] = 0.0
        self.memory_state["recall_strength"] = 0.0
        if self.auto_generation:
            self.next_generation_at = time.monotonic() + 1.0
        self._save_state()

    def set_auto_generation(self, enabled):
        with self.lock:
            self.auto_generation = bool(enabled)
            if not self.auto_generation:
                self.next_generation_at = None
                if not any(organism.alive for organism in self.organisms):
                    self.status = "DEAD"
            elif not any(organism.alive for organism in self.organisms):
                self.handle_death()
            self._save_state()
            return self.auto_generation

    def snapshot(self):
        with self.lock:
            next_generation_in = 0.0
            if self.next_generation_at is not None and self.auto_generation:
                next_generation_in = max(0.0, self.next_generation_at - time.monotonic())
            latest = self.memory.recall(1)[0] if self.memory.entries else None
            remembered_experiences = [
                {
                    "generation": entry.get("generation", 0),
                    "organism_id": entry.get("organism_id", ""),
                    "reward": float(entry.get("reward", 0.0)),
                    "timestamp": entry.get("timestamp", 0),
                    "result": {
                        "moved": float(entry.get("result", {}).get("moved", 0.0)),
                        "ate_food": bool(entry.get("result", {}).get("ate_food", False)),
                        "died": bool(entry.get("result", {}).get("died", False)),
                    },
                }
                for entry in self.memory.recall(5)
            ]
            return {
                "running": self.running,
                "generation": self.generation,
                "auto_generation": self.auto_generation,
                "status": self.status,
                "next_generation_in": round(next_generation_in, 3),
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
                    "last": latest,
                    "recent": self.memory.recall(5),
                    "last_reward": latest["reward"] if latest else 0.0,
                    "average_recent_reward": self.memory.reward_baseline(10),
                    "latest_positive": bool(latest and latest["reward"] > 0),
                    "latest_negative": bool(latest and latest["reward"] < 0),
                    "signal": float(self.memory_state["signal"]),
                    "neural_signal": float(self.memory_state["neural_signal"]),
                    "recall_strength": float(self.memory_state["recall_strength"]),
                    "valence": self.memory_state["valence"],
                    "neurons": dict(self.memory_state["neurons"]),
                    "latest_action": dict(self.memory_state["latest_action"]),
                    "latest_result": dict(self.memory_state["latest_result"]),
                    "remembered_experiences": remembered_experiences,
                },
            }


simulation = LifeSimulation()
