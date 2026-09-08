from flask import Flask, jsonify, render_template, request

import life
import proc

app = Flask(__name__)


@app.route("/")
def index():
    neuron_count = proc.get_neuron_count()
    return render_template("index.html", count=neuron_count)


@app.route("/lab")
def lab():
    return render_template("lab.html")


@app.route("/config", methods=["GET"])
def config():
    return jsonify(proc.get_config())


@app.route("/signal", methods=["POST"])
def signal():
    payload = request.get_json(silent=True) or {}
    try:
        input_signal = float(payload.get("input", 1.0))
        activation_name = str(payload.get("activation", "LINEAR"))
        return jsonify(proc.send_signal(input_signal, activation_name))
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.route("/api/lab/state", methods=["GET"])
def lab_state():
    return jsonify(life.simulation.snapshot())


@app.route("/api/lab/start", methods=["POST"])
def lab_start():
    life.simulation.start()
    return jsonify(life.simulation.snapshot())


@app.route("/api/lab/stop", methods=["POST"])
def lab_stop():
    life.simulation.stop()
    return jsonify(life.simulation.snapshot())


@app.route("/api/lab/reset", methods=["POST"])
def lab_reset():
    life.simulation.reset()
    return jsonify(life.simulation.snapshot())


@app.route("/api/lab/generation", methods=["POST"])
def lab_generation():
    life.simulation.new_generation()
    return jsonify(life.simulation.snapshot())


@app.route("/api/brain/state", methods=["GET"])
def brain_state():
    state = life.simulation.snapshot()
    return jsonify({
        "tick": state["tick"],
        "generation": state["generation"],
        "running": state["running"],
        "organism": state["organism"],
        "food_eaten": state["food_eaten"],
        "brain": state["brain"],
        "reward": state["reward"],
        "last_action": state["last_action"],
        "learning": state["learning"],
        "memory": state["memory"],
    })


@app.route("/api/brain/memory", methods=["GET"])
def brain_memory():
    memory = life.simulation.snapshot()["memory"]
    return jsonify(memory)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4444, debug=True)
