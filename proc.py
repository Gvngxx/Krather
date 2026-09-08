import numpy as np

INPUT_NODE = "INPUT"
OUTPUT_NODE = "OUTPUT"


class Neuron:
    def __init__(self, id_name, bias=0.0):
        self.id_name = id_name
        self.activation = 0.0
        self.bias = float(bias)
        self.received = 0.0
        self.weighted_sum = 0.0

    def process(self, received, weighted_sum, activation_function):
        self.received = float(received)
        self.weighted_sum = float(weighted_sum)
        self.activation = float(activation_function(self.weighted_sum))
        return self.activation

    def to_dict(self):
        return {
            "id": self.id_name,
            "bias": self.bias,
            "activation": self.activation,
            "received": self.received,
            "weighted_sum": self.weighted_sum,
        }


class Connection:
    def __init__(self, source, target, weight):
        self.source = source
        self.target = target
        self.weight = float(weight)
        self.signal = 0.0
        self.weighted_signal = 0.0

    def to_dict(self):
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "signal": self.signal,
            "weighted_signal": self.weighted_signal,
        }


ACTIVATION_FUNCTIONS = {
    "LINEAR": lambda value: value,
    "RELU": lambda value: np.maximum(0.0, value),
    "SIGMOID": lambda value: 1.0 / (1.0 + np.exp(-value)),
    "TANH": np.tanh,
}


network = [
    Neuron("N1"),
    Neuron("N2"),
    Neuron("N3"),
    Neuron("N4"),
    Neuron("N5"),
    Neuron("N6"),
    Neuron("N7"),
    Neuron("N8"),
    Neuron("N9"),
    Neuron("N10"),
    Neuron("N11"),
    Neuron("N12"),
    Neuron("N13"),
    Neuron("N14"),
]

network_by_id = {neuron.id_name: neuron for neuron in network}

connections = [
    Connection(INPUT_NODE, "N1", 0.8),

    Connection("N1", "N2", 1.5),
    Connection("N1", "N3", -1.5),

    Connection("N2", "N4", 1.6),
    Connection("N3", "N5", 1.7),
    Connection("N4", "N10", -1.5),

    Connection("N5", "N7", 1.5),
    Connection("N5", "N9", -1.6),

    Connection("N5", "N6", 1.6),
    Connection("N5", "N8", 1.7),

    Connection("N10", "N11", 1.4),
    Connection("N11", "N12", 1.4),

    Connection("N13", "N14", 1.8),

    Connection("N12", "N13", 1.4),
    Connection("N13", OUTPUT_NODE, 0.8),
]
LEARNING_RATE = 0.03
WEIGHT_LIMIT = 2.0
MAX_WEIGHT_STEP = 0.06
MIN_ACTIVE_WEIGHT = 0.12
OUTPUT_SCALING = 7.0
learning_stats = {
    "updates": 0,
    "last_reward": 0.0,
    "last_delta": 0.0,
    "total_abs_delta": 0.0,
    "total_weight_change": 0.0,
    "average_weight_change": 0.0,
    "active_connections": 0,
    "near_dead_connections": 0,
    "average_signal": 0.0,
    "reward_total": 0.0,
    "reward_history": [],
    "reward_average": 0.0,
    "reward_last_10": 0.0,
}


def get_neuron_count():
    return len(network)


def _serialize_neuron(neuron):
    return neuron.to_dict()


def _serialize_connection(connection):
    return connection.to_dict()


def serialize_brain():
    return {
        "neurons": [_serialize_neuron(neuron) for neuron in network],
        "connections": [_serialize_connection(connection) for connection in connections],
    }


def restore_brain(payload):
    for saved_neuron in payload.get("neurons", []):
        neuron = network_by_id.get(saved_neuron.get("id"))
        if neuron is None:
            continue
        neuron.bias = float(saved_neuron.get("bias", neuron.bias))
        neuron.activation = float(saved_neuron.get("activation", 0.0))
        neuron.received = float(saved_neuron.get("received", 0.0))
        neuron.weighted_sum = float(saved_neuron.get("weighted_sum", 0.0))

    for index, saved_connection in enumerate(payload.get("connections", [])):
        if index >= len(connections):
            break
        connection = connections[index]
        connection.weight = float(saved_connection.get("weight", connection.weight))
        connection.signal = float(saved_connection.get("signal", 0.0))
        connection.weighted_signal = float(saved_connection.get("weighted_signal", 0.0))


def serialize_learning():
    serialized = dict(learning_stats)
    serialized["reward_history"] = list(learning_stats["reward_history"])
    return serialized


def restore_learning(payload):
    for key in learning_stats:
        if key in payload:
            if key in {"updates", "active_connections", "near_dead_connections"}:
                learning_stats[key] = int(payload[key])
            elif key == "reward_history":
                learning_stats[key] = [float(value) for value in payload[key]][-100:]
            else:
                learning_stats[key] = float(payload[key])


def _bounded_update(connection, delta):
    """Apply a small update while preserving an active route's direction."""
    old_weight = connection.weight
    bounded_delta = float(np.clip(delta, -MAX_WEIGHT_STEP, MAX_WEIGHT_STEP))
    proposed_weight = float(np.clip(old_weight + bounded_delta, -WEIGHT_LIMIT, WEIGHT_LIMIT))
    if abs(connection.signal) > 1e-4 and abs(old_weight) >= MIN_ACTIVE_WEIGHT and abs(proposed_weight) < MIN_ACTIVE_WEIGHT:
        proposed_weight = float(np.copysign(MIN_ACTIVE_WEIGHT, old_weight))
    connection.weight = proposed_weight
    return proposed_weight - old_weight


def learn_from_reward(reward, learning_rate=LEARNING_RATE, terminal=False):
    """Apply small reward-modulated updates only to connections that carried signal."""
    reward = float(np.clip(reward, -20.0, 20.0))
    normalized_reward = float(np.clip(reward / 10.0, -1.0, 1.0))
    total_delta = 0.0
    active_count = 0
    signal_magnitudes = []

    for connection in connections:
        source_signal = float(np.clip(connection.signal, -1.0, 1.0))
        signal_magnitude = abs(source_signal)
        signal_magnitudes.append(signal_magnitude)
        if signal_magnitude <= 1e-4:
            continue
        active_count += 1
        target = network_by_id.get(connection.target)
        target_activation = target.activation if target is not None else source_signal
        participation = float(np.clip(source_signal * target_activation, -1.0, 1.0))
        reward_factor = -1.35 if terminal else normalized_reward
        delta = learning_rate * reward_factor * participation
        total_delta += _bounded_update(connection, delta)

    for neuron in network:
        if abs(neuron.activation) <= 1e-4:
            continue
        bias_factor = -1.35 if terminal else normalized_reward
        bias_delta = float(np.clip(learning_rate * bias_factor * neuron.activation * 0.5, -MAX_WEIGHT_STEP, MAX_WEIGHT_STEP))
        neuron.bias = float(np.clip(neuron.bias + bias_delta, -WEIGHT_LIMIT, WEIGHT_LIMIT))
        total_delta += bias_delta

    learning_stats["updates"] += 1
    learning_stats["last_reward"] = reward
    learning_stats["last_delta"] = total_delta
    learning_stats["total_abs_delta"] += abs(total_delta)
    learning_stats["total_weight_change"] += abs(total_delta)
    learning_stats["average_weight_change"] = learning_stats["total_weight_change"] / max(1, learning_stats["updates"])
    learning_stats["active_connections"] = active_count
    learning_stats["near_dead_connections"] = sum(1 for connection in connections if abs(connection.weight) < MIN_ACTIVE_WEIGHT)
    learning_stats["average_signal"] = float(np.mean(signal_magnitudes)) if signal_magnitudes else 0.0
    learning_stats["reward_total"] += reward
    learning_stats["reward_history"].append(reward)
    learning_stats["reward_history"] = learning_stats["reward_history"][-100:]
    learning_stats["reward_average"] = float(np.mean(learning_stats["reward_history"]))
    learning_stats["reward_last_10"] = float(np.mean(learning_stats["reward_history"][-10:]))
    return {
        "reward": reward,
        "delta": total_delta,
        "updates": learning_stats["updates"],
        "active_connections": active_count,
        "average_signal": learning_stats["average_signal"],
    }


def _all_nodes():
    nodes = {INPUT_NODE, OUTPUT_NODE}
    nodes.update(neuron.id_name for neuron in network)
    nodes.update(connection.source for connection in connections)
    nodes.update(connection.target for connection in connections)
    return nodes


def _topological_layers():
    nodes = _all_nodes()
    indegree = {node: 0 for node in nodes}
    outgoing = {node: [] for node in nodes}

    for connection in connections:
        outgoing[connection.source].append(connection)
        indegree[connection.target] += 1

    layers = []
    remaining = set(nodes)
    while remaining:
        current_layer = sorted(node for node in remaining if indegree[node] == 0)
        if not current_layer:
            raise ValueError("La red debe ser un grafo dirigido acíclico")
        layers.append(current_layer)
        for node in current_layer:
            remaining.remove(node)
            for connection in outgoing[node]:
                indegree[connection.target] -= 1

    return layers


def get_config():
    layers = _topological_layers()
    known_neurons = {neuron.id_name: neuron for neuron in network}
    all_node_ids = sorted(_all_nodes() - {INPUT_NODE, OUTPUT_NODE})
    neurons_payload = []

    for node_id in all_node_ids:
        neuron = known_neurons.get(node_id)
        if neuron is not None:
            neurons_payload.append({"id": neuron.id_name, "bias": neuron.bias})
        else:
            neurons_payload.append({"id": node_id, "bias": 0.0})

    return {
        "input_node": INPUT_NODE,
        "output_node": OUTPUT_NODE,
        "layers": layers,
        "neurons": neurons_payload,
        "connections": [_serialize_connection(connection) for connection in connections],
        "activation_functions": list(ACTIVATION_FUNCTIONS.keys()),
        "default_activation": "LINEAR",
    }


def send_signal(input_signal, activation_name="LINEAR"):
    activation_name = activation_name.upper()
    if activation_name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Función de activación no soportada: {activation_name}")

    layers = _topological_layers()
    layer_index = {node: index for index, layer in enumerate(layers) for node in layer}
    activation_function = ACTIVATION_FUNCTIONS[activation_name]

    for neuron in network:
        neuron.activation = 0.0
        neuron.received = 0.0
        neuron.weighted_sum = 0.0

    for connection in connections:
        connection.signal = 0.0
        connection.weighted_signal = 0.0

    node_values = {INPUT_NODE: float(input_signal)}
    step_records = []
    output_connections = [connection for connection in connections if connection.target == OUTPUT_NODE]
    reachable_nodes = {INPUT_NODE}
    changed = True
    while changed:
        changed = False
        for connection in connections:
            if connection.source in reachable_nodes and connection.target not in reachable_nodes:
                reachable_nodes.add(connection.target)
                changed = True

    for layer in layers:
        if INPUT_NODE in layer:
            continue

        for node_id in layer:
            if node_id == OUTPUT_NODE or node_id not in reachable_nodes:
                continue

            neuron = network_by_id.get(node_id)
            if neuron is None:
                continue

            incoming_connections = [
                connection
                for connection in connections
                if connection.target == node_id and connection.source in reachable_nodes
            ]
            incoming_signals = []
            weighted_inputs = []

            for connection in incoming_connections:
                connection_index = connections.index(connection)
                source_value = node_values.get(connection.source, 0.0)
                weighted_signal = float(source_value * connection.weight)
                connection.signal = float(source_value)
                connection.weighted_signal = float(weighted_signal)
                incoming_signals.append(float(source_value))
                weighted_inputs.append(float(weighted_signal))

                step_records.append({
                    "source": connection.source,
                    "target": connection.target,
                    "weight": connection.weight,
                    "signal": float(source_value),
                    "weighted_signal": float(weighted_signal),
                    "phase": layer_index[node_id],
                    "connection_index": connection_index,
                })

            if incoming_connections:
                received = float(np.sum(incoming_signals))
                weighted_sum = float(np.sum(weighted_inputs)) + neuron.bias
                neuron.process(received, weighted_sum, activation_function)
                node_values[node_id] = neuron.activation
            else:
                neuron.activation = float(activation_function(neuron.bias))
                neuron.received = 0.0
                neuron.weighted_sum = float(neuron.bias)
                node_values[node_id] = neuron.activation

    output_signal = 0.0
    reached_output = False
    for connection in output_connections:
        if connection.source not in reachable_nodes:
            continue
        connection_index = connections.index(connection)
        source_value = node_values.get(connection.source, 0.0)
        weighted_signal = float(source_value * connection.weight)
        connection.signal = float(source_value)
        connection.weighted_signal = float(weighted_signal)
        output_signal += weighted_signal
        step_records.append({
            "source": connection.source,
            "target": connection.target,
            "weight": connection.weight,
            "signal": float(source_value),
            "weighted_signal": float(weighted_signal),
            "phase": layer_index.get(OUTPUT_NODE, len(layers) - 1),
            "connection_index": connection_index,
        })
        reached_output = reached_output or abs(weighted_signal) > 1e-9

    final_output = float(output_signal) * OUTPUT_SCALING
    node_values[OUTPUT_NODE] = final_output

    return {
        "input": float(input_signal),
        "activation_function": activation_name,
        "layers": layers,
        "neurons": [_serialize_neuron(neuron) for neuron in network],
        "connections": [_serialize_connection(connection) for connection in connections],
        "steps": step_records,
        "final_output": final_output,
        "output_reached": reached_output,
    }


def process_sensors(sensors, activation_name="TANH"):
    """Encode the organism sensors through the existing neural network."""
    encoded_input = (
        float(sensors.get("food_dx", 0.0)) * 0.55
        + float(sensors.get("food_dy", 0.0)) * 0.35
        + float(sensors.get("distance", 0.0)) * -0.15
        + float(sensors.get("energy", 0.0)) * 0.1
    )
    result = send_signal(encoded_input, activation_name)
    neuron_values = {neuron["id"]: neuron["activation"] for neuron in result["neurons"]}
    if result["output_reached"]:
        output_x = float(np.tanh(result["final_output"] * 1.15))
        output_y = float(np.tanh(neuron_values.get("N3", 0.0) * 6.0))
    else:
        output_x = 0.0
        output_y = 0.0
    result["sensors"] = {key: float(value) for key, value in sensors.items()}
    result["encoded_input"] = encoded_input
    result["outputs"] = {"move_x": output_x, "move_y": output_y}
    result["animation_ms"] = int(np.random.uniform(90.0, 260.0))
    result["learning_rate"] = LEARNING_RATE
    return result
