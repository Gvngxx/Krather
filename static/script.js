const SVG_NS = "http://www.w3.org/2000/svg";
let networkConfig = null;
let isAnimating = false;
let lastBrainTick = 0;
let brainQueue = [];
let animationToken = 0;
let scheduledPulseTimeouts = [];
let lastOrganismAlive = true;

document.addEventListener("DOMContentLoaded", initNetwork);

async function initNetwork() {
  try {
    const response = await fetch("/config");
    if (!response.ok) {
      throw new Error("No se pudo cargar la configuración de la red");
    }

    networkConfig = await response.json();
    renderGraph(networkConfig);

    document.getElementById("startButton").addEventListener("click", () => controlLab("start"));
    document.getElementById("stopButton").addEventListener("click", () => controlLab("stop"));
    document.getElementById("resetButton").addEventListener("click", () => controlLab("reset"));
    document.getElementById("generationButton").addEventListener("click", () => controlLab("generation"));
    document.getElementById("autoGeneration").addEventListener("click", toggleAuto);
    pollBrainState();
    setStatus("RED LISTA");
  } catch (error) {
    setStatus("ERROR DE RED");
    appendLog(error.message);
  }
}

async function toggleAuto() {
  const button = document.getElementById("autoGeneration");
  try {
    const response = await fetch("/api/lab/auto", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: button.getAttribute("aria-pressed") !== "true" }),
    });
    if (!response.ok) throw new Error("No se pudo cambiar AUTO");
    updateMonitor(await response.json());
  } catch (error) {
    appendLog(error.message);
  }
}

function buildActivationOptions() {
  const select = document.getElementById("activationSelect");
  const options = networkConfig.activation_functions || ["LINEAR"];

  select.replaceChildren(
    ...options.map((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      return option;
    })
  );

  select.value = networkConfig.default_activation || "LINEAR";
}

function renderGraph(config) {
  const canvas = document.getElementById("neuralCanvas");
  const allNodes = new Set([
    ...(config.neurons || []).map((neuron) => neuron.id),
    ...(config.connections || []).flatMap((connection) => [connection.source, connection.target]),
    config.input_node,
    config.output_node,
  ]);

  const layers = (config.layers && config.layers.length ? config.layers : buildFallbackLayers(config, [...allNodes]));
  const width = Math.max(900, layers.length * 180 + 100);
  const height = Math.max(520, Math.max(...layers.map((layer) => layer.length)) * 150 + 120);
  const positions = getLayerPositions(layers, width, height);
  const savedPositions = JSON.parse(localStorage.getItem("brain-monitor-positions") || "{}");
  Object.entries(savedPositions).forEach(([id, position]) => {
    if (positions[id] && Number.isFinite(position.x) && Number.isFinite(position.y)) {
      positions[id] = { x: position.x, y: position.y };
    }
  });
  const interactionState = {
    dragId: null,
    pan: JSON.parse(localStorage.getItem("brain-monitor-camera") || "{\"x\":0,\"y\":0,\"scale\":1}"),
    panId: null,
    panStart: null,
  };

  window.__networkPositions = positions;
  window.__graphInteractionState = interactionState;

  const svg = createSvg("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    preserveAspectRatio: "xMidYMid meet",
  });

  const defs = createSvg("defs");
  const marker = createSvg("marker", {
    id: "arrow",
    markerWidth: "8",
    markerHeight: "8",
    refX: "7",
    refY: "4",
    orient: "auto",
  });
  marker.appendChild(createSvg("path", { d: "M0,0 L8,4 L0,8 Z", fill: "#52627a" }));
  defs.appendChild(marker);
  svg.appendChild(defs);

  const graphLayer = createSvg("g", { class: "graph-layer" });
  svg.appendChild(graphLayer);

  const applyCamera = () => {
    const camera = window.__graphInteractionState.pan;
    graphLayer.setAttribute("transform", `translate(${camera.x} ${camera.y}) scale(${camera.scale})`);
    localStorage.setItem("brain-monitor-camera", JSON.stringify(camera));
  };
  window.__applyGraphCamera = applyCamera;
  applyCamera();

  config.connections.forEach((connection, connectionIndex) => {
    const start = positions[connection.source];
    const end = positions[connection.target];

    if (!start || !end) return;

    const group = createSvg("g", {
      class: "connection-group",
      id: connectionId({ ...connection, connection_index: connectionIndex }),
      "data-source": connection.source,
      "data-target": connection.target,
      "data-connection-index": connectionIndex,
    });

    const line = createSvg("line", {
      class: "connection-line",
      x1: start.x,
      y1: start.y,
      x2: end.x,
      y2: end.y,
      "marker-end": "url(#arrow)",
    });

    const label = createSvg("text", {
      class: "weight-label",
      x: (start.x + end.x) / 2,
      y: (start.y + end.y) / 2 - 10,
    });
    label.textContent = `w=${format(connection.weight)}`;

    const pulse = createSvg("circle", {
      class: "signal-pulse",
      r: "6",
      cx: start.x,
      cy: start.y,
      opacity: "0",
    });

    group.append(line, label, pulse);
    graphLayer.appendChild(group);
  });

  Object.entries(positions).forEach(([id, position]) => {
    const isMemoryNeuron = ["N6", "N7", "N8", "N9"].includes(id);
    const nodeGroup = createSvg("g", {
      class: `svg-node${isMemoryNeuron ? " memory-neuron" : ""}`,
      id: nodeId(id),
      transform: `translate(${position.x} ${position.y})`,
      "data-node-id": id,
    });

    const shape = isMemoryNeuron
      ? createSvg("rect", { x: "-39", y: "-39", width: "78", height: "78", rx: "4" })
      : createSvg("circle", { r: id === "INPUT" || id === "OUTPUT" ? "34" : "39" });
    const label = createSvg("text", { class: "node-label", y: "5" });
    label.textContent = id;

    nodeGroup.append(shape, label);
    graphLayer.appendChild(nodeGroup);

    nodeGroup.addEventListener("pointerdown", (event) => {
      if (isAnimating) return;
      const state = window.__graphInteractionState;
      const point = svgPointToGraph(event, svg, state.pan);
      state.dragId = id;
      state.offsetX = point.x - position.x;
      state.offsetY = point.y - position.y;
      nodeGroup.setPointerCapture(event.pointerId);
      nodeGroup.classList.add("dragging");
      event.stopPropagation();
    });

    nodeGroup.addEventListener("pointermove", (event) => {
      const state = window.__graphInteractionState;
      if (!state.dragId || state.dragId !== id) return;

      const point = svgPointToGraph(event, svg, state.pan);
      const nextX = point.x - state.offsetX;
      const nextY = point.y - state.offsetY;
      window.__networkPositions[id] = { x: Math.max(40, Math.min(width - 40, nextX)), y: Math.max(40, Math.min(height - 40, nextY)) };
      nodeGroup.setAttribute("transform", `translate(${window.__networkPositions[id].x} ${window.__networkPositions[id].y})`);
      localStorage.setItem("brain-monitor-positions", JSON.stringify(window.__networkPositions));
      refreshConnectionPositions();
    });

    nodeGroup.addEventListener("pointerup", () => {
      const state = window.__graphInteractionState;
      state.dragId = null;
      nodeGroup.classList.remove("dragging");
    });

    nodeGroup.addEventListener("pointerleave", () => {
      nodeGroup.classList.remove("hovered");
    });

    nodeGroup.addEventListener("pointerenter", () => {
      if (isAnimating) return;
      nodeGroup.classList.add("hovered");
    });
  });

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const state = window.__graphInteractionState;
    const before = svgPointToGraph(event, svg, state.pan);
    const nextScale = Math.max(0.55, Math.min(2.5, state.pan.scale * (event.deltaY < 0 ? 1.1 : 0.9)));
    state.pan.scale = nextScale;
    const after = svgPointToGraph(event, svg, state.pan);
    state.pan.x += (after.x - before.x) * nextScale;
    state.pan.y += (after.y - before.y) * nextScale;
    applyCamera();
  }, { passive: false });

  svg.addEventListener("pointerdown", (event) => {
    if (event.target !== svg) return;
    const state = window.__graphInteractionState;
    state.panId = event.pointerId;
    state.panStart = { x: event.clientX, y: event.clientY, panX: state.pan.x, panY: state.pan.y };
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener("pointermove", (event) => {
    const state = window.__graphInteractionState;
    if (state.panId !== event.pointerId || !state.panStart) return;
    state.pan.x = state.panStart.panX + event.clientX - state.panStart.x;
    state.pan.y = state.panStart.panY + event.clientY - state.panStart.y;
    applyCamera();
  });
  svg.addEventListener("pointerup", () => {
    const state = window.__graphInteractionState;
    state.panId = null;
    state.panStart = null;
  });

  const refreshConnectionPositions = () => {
    const allConnections = [...svg.querySelectorAll(".connection-group")];
    allConnections.forEach((connectionGroup) => {
      const sourceId = connectionGroup.dataset.source;
      const targetId = connectionGroup.dataset.target;
      const start = window.__networkPositions[sourceId];
      const end = window.__networkPositions[targetId];
      if (!start || !end) return;

      const line = connectionGroup.querySelector("line");
      const label = connectionGroup.querySelector("text");
      const pulse = connectionGroup.querySelector("circle.signal-pulse");

      if (line) {
        line.setAttribute("x1", start.x);
        line.setAttribute("y1", start.y);
        line.setAttribute("x2", end.x);
        line.setAttribute("y2", end.y);
      }

      if (label) {
        label.setAttribute("x", (start.x + end.x) / 2);
        label.setAttribute("y", (start.y + end.y) / 2 - 10);
      }

      if (pulse) {
        pulse.setAttribute("cx", start.x);
        pulse.setAttribute("cy", start.y);
      }
    });
  };

  canvas.replaceChildren(svg);
  document.getElementById("graphMeta").textContent = `${config.neurons.length + 2} nodos / ${config.connections.length} conexiones`;
}

function buildFallbackLayers(config, explicitNodes) {
  const nodes = explicitNodes || ["INPUT", ...(config.neurons || []).map((neuron) => neuron.id), "OUTPUT"];
  const incoming = Object.fromEntries(nodes.map((node) => [node, 0]));
  const outgoing = Object.fromEntries(nodes.map((node) => [node, []]));

  (config.connections || []).forEach((connection) => {
    incoming[connection.target] = (incoming[connection.target] || 0) + 1;
    outgoing[connection.source] = outgoing[connection.source] || [];
    outgoing[connection.source].push(connection.target);
  });

  const layers = [];
  const remaining = new Set(nodes);

  while (remaining.size > 0) {
    const current = [...remaining].filter((node) => incoming[node] === 0).sort();
    if (current.length === 0) {
      throw new Error("La topología contiene un ciclo");
    }

    layers.push(current);
    current.forEach((node) => {
      remaining.delete(node);
      (outgoing[node] || []).forEach((target) => {
        incoming[target] -= 1;
      });
    });
  }

  return layers;
}

function getLayerPositions(layers, width, height) {
  const positions = {};

  layers.forEach((layer, layerIndex) => {
    const x = 90 + layerIndex * ((width - 180) / Math.max(1, layers.length - 1));
    const verticalSpacing = height / (layer.length + 1);

    layer.forEach((id, index) => {
      positions[id] = {
        x,
        y: verticalSpacing * (index + 1),
      };
    });
  });

  return positions;
}

async function triggerPulse() {
  return controlLab("start");
}

async function controlLab(action) {
  try {
    if (action === "stop" || action === "reset" || action === "generation") {
      stopBrainAnimation();
    }
    const response = await fetch(`/api/lab/${action}`, { method: "POST" });
    const state = await response.json();
    updateMonitor(state);
    if (action === "reset" || action === "generation") {
      brainQueue = [];
      lastBrainTick = state.tick;
      resetView();
    }
    if (action === "stop") resetView();
  } catch (error) {
    appendLog(error.message);
    setStatus("ERROR DE LAB");
  }
}

async function pollBrainState() {
  try {
    const response = await fetch("/api/brain/state");
    const state = await response.json();
    updateMonitor(state);
    if (state.tick > lastBrainTick && state.brain && state.brain.steps) {
      lastBrainTick = state.tick;
      brainQueue.push(state.brain);
      drainBrainQueue();
    }
  } catch (error) {
    setStatus("SIN CONEXIÓN");
  } finally {
    window.setTimeout(pollBrainState, 120);
  }
}

async function drainBrainQueue() {
  if (isAnimating || brainQueue.length === 0) return;
  isAnimating = true;
  const brain = brainQueue.shift();
  const token = animationToken;
  await animateFlow(brain, token);
  isAnimating = false;
  if (token !== animationToken) return;
  drainBrainQueue();
}

function updateMonitor(state) {
  const organism = state.organism || {};
  const action = state.last_action || {};
  const actionName = formatAction(action);
  const running = Boolean(state.running);
  const organismAlive = organism.alive !== false;

  if (lastOrganismAlive && !organismAlive) {
    triggerDeathBurst();
  }
  lastOrganismAlive = organismAlive;

  document.getElementById("labStatus").textContent = state.status === "GENERATION_COMPLETE"
    ? `GENERATION COMPLETE / ${Number(state.next_generation_in || 0).toFixed(1)}s`
    : running ? "RUNNING" : "STOPPED";
  document.getElementById("organismStatus").textContent = organismAlive ? "ALIVE" : "DEAD";
  document.getElementById("generationValue").textContent = state.generation ?? 1;
  document.getElementById("stepValue").textContent = state.tick ?? 0;
  document.getElementById("energyValue").textContent = format(organism.energy ?? 0, 1);
  document.getElementById("actionValue").textContent = actionName;
  document.getElementById("rewardValue").textContent = format(state.reward ?? 0, 1);
  document.getElementById("averageReward").textContent = format(state.learning?.reward_average ?? 0, 2);
  document.getElementById("foodEaten").textContent = state.food_eaten ?? 0;
  document.getElementById("learnedTicks").textContent = state.learning?.total_ticks ?? 0;
  document.getElementById("weightUpdates").textContent = state.learning?.updates ?? 0;
  document.getElementById("activeConnections").textContent = state.learning?.active_connections ?? 0;
  document.getElementById("nearDeadConnections").textContent = state.learning?.near_dead_connections ?? 0;
  document.getElementById("averageSignal").textContent = format(state.learning?.average_signal ?? 0, 2);
  document.getElementById("averageWeightChange").textContent = format(state.learning?.average_weight_change ?? 0, 4);
  const autoButton = document.getElementById("autoGeneration");
  autoButton.textContent = state.auto_generation ? "ON" : "OFF";
  autoButton.setAttribute("aria-pressed", String(Boolean(state.auto_generation)));
  document.getElementById("brainMemoryCount").textContent = `${state.memory?.count ?? 0} / ${state.memory?.capacity ?? 0}`;
  document.getElementById("brainMemoryAverage").textContent = format(state.memory?.average_recent_reward ?? 0, 2);
  const memorySignal = Number(state.memory?.signal ?? 0);
  document.getElementById("brainMemorySignal").textContent = format(memorySignal, 2);
  document.getElementById("graphMeta").textContent = `${networkConfig.neurons.length + 2} nodos / ${networkConfig.connections.length} conexiones · G${state.generation ?? 1} · R${format(state.reward ?? 0, 2)} · M${state.memory?.count ?? 0}`;
  updateMemoryVisualization(state.memory, state.brain);
  updateMemory(state.memory);
  updateConnectionWeights(state.brain?.connections || []);
  setStatus(running ? "CEREBRO EJECUTANDO" : organismAlive ? "RED LISTA" : "ORGANISMO MUERTO");
}

function updateMemoryVisualization(memory, brain) {
  const signal = Math.max(0, Math.min(1, Number(memory?.signal ?? 0)));
  const indicator = document.getElementById("memoryIndicator");
  if (indicator) {
    indicator.textContent = signal > 0 ? `MEMORY ${format(signal, 2)}` : "MEMORY OFF";
    indicator.classList.toggle("is-active", signal > 0);
  }

  const neuronValues = memory?.neurons || brain?.memory_neurons || {};
  ["N6", "N7", "N8", "N9"].forEach((id) => {
    const node = document.getElementById(nodeId(id));
    const intensity = signal > 0 ? Math.max(0, Math.min(1, Math.abs(Number(neuronValues[id] || 0)))) : 0;
    if (!node) return;
    node.classList.toggle("memory-active", intensity > 0);
    node.style.setProperty("--memory-intensity", intensity.toFixed(3));
  });

  document.querySelectorAll(".connection-group").forEach((group) => {
    const isMemoryRoute = ["N6", "N7", "N8", "N9"].includes(group.dataset.target);
    group.classList.toggle("memory-active", isMemoryRoute && signal > 0);
    group.style.setProperty("--memory-intensity", signal.toFixed(3));
  });
}

function updateMemory(memory) {
  if (!memory) return;
  document.getElementById("memoryCount").textContent = `${memory.count ?? 0} / ${memory.capacity ?? 0}`;
  const last = memory.last;
  document.getElementById("memoryReward").textContent = last ? format(last.reward, 2) : "0.0";
  document.getElementById("memoryAction").textContent = last ? formatAction(last.action) : "IDLE";
  document.getElementById("memoryResult").textContent = last ? formatMemoryResult(last.result) : "--";
  const recent = document.getElementById("memoryRecent");
  recent.replaceChildren(...(memory.recent || []).slice().reverse().map((entry) => {
    const item = document.createElement("li");
    item.textContent = `t${entry.timestamp}  ${format(entry.reward, 1)}  ${formatMemoryResult(entry.result)}`;
    return item;
  }));
}

function formatMemoryResult(result) {
  if (!result) return "--";
  if (result.ate_food) return "FOOD";
  if (result.alive === false) return "DEAD";
  return result.moved > 0.01 ? "MOVED" : "STILL";
}

function updateConnectionWeights(connections) {
  document.querySelectorAll(".connection-group").forEach((group) => {
    const index = Number(group.dataset.connectionIndex);
    const connection = connections[index];
    const label = group.querySelector(".weight-label");
    if (label && connection) {
      label.textContent = `w=${format(connection.weight)}`;
    }
  });
}

function formatAction(action) {
  const moveX = Number(action.move_x) || 0;
  const moveY = Number(action.move_y) || 0;
  const directions = [];

  if (moveX > 0.2) directions.push("RIGHT");
  if (moveX < -0.2) directions.push("LEFT");
  if (moveY > 0.2) directions.push("DOWN");
  if (moveY < -0.2) directions.push("UP");

  const name = directions.length ? `MOVE_${directions.join("_")}` : "IDLE";
  return `${name} (${moveX.toFixed(2)}, ${moveY.toFixed(2)})`;
}

async function animateFlow(data, token) {
  const animationMs = Number(data.animation_ms) || 130;
  const travelWindow = Math.max(260, animationMs * 1.9);
  activateNode("INPUT", "active");
  appendLog(`INPUT recibe ${format(data.input)}.`);

  const phases = [...new Set(data.steps.map((step) => step.phase))].sort((a, b) => a - b);

  for (const [phaseIndex, phase] of phases.entries()) {
    if (token !== animationToken) return;
    const phaseSteps = data.steps.filter((step) => step.phase === phase);

    phaseSteps.forEach((step) => {
      const group = document.getElementById(connectionId(step));
      if (group) {
        group.classList.remove("fading");
        group.classList.add("active");
      }

      const sourceNode = document.getElementById(nodeId(step.source));
      if (sourceNode) {
        sourceNode.classList.remove("fading");
        sourceNode.classList.add("active");
      }

      const targetNode = document.getElementById(nodeId(step.target));
      if (targetNode) {
        targetNode.classList.remove("fading");
        const targetMode = isIsolatedNode(step.target) ? "isolated" : "active";
        activateNode(step.target, targetMode);
      }
    });

    if (window.__networkPositions) {
      const sources = [...new Set(phaseSteps.map((step) => step.source))];
      sources.forEach((source) => {
        const node = document.getElementById(nodeId(source));
        if (node) {
          node.classList.add("pulse");
          scheduledPulseTimeouts.push(setTimeout(() => node.classList.remove("pulse"), Math.max(180, animationMs * 0.8)));
        }
      });
    }

    phaseSteps.forEach((step) => {
      animateConnection(step, animationMs, token);
      showConnection(step);
      appendLog(`${step.source} → ${step.target}: signal=${format(step.signal)}, weight=${format(step.weight)}, weighted=${format(step.weighted_signal)}`);
    });

    await wait(animationMs);
    if (token !== animationToken) return;

    [...new Set(phaseSteps.map((step) => step.target))].forEach((target) => {
      const neuron = data.neurons.find((item) => item.id === target);
      if (neuron) {
        showNeuron(neuron);
      }
    });
  }

  if (data.output_reached) {
    activateNode("OUTPUT", "active");
    setTimeout(() => activateNode("OUTPUT"), Math.max(200, animationMs * 0.9));
  }
  const outputNeuron = data.neurons.find((item) => item.id === "N4");
  if (outputNeuron) {
    showNeuron(outputNeuron);
  }
}

function animateConnection(step, duration, token) {
  const group = document.getElementById(connectionId(step));
  if (!group) return;

  const line = group.querySelector("line");
  const pulse = group.querySelector("circle");
  if (!line || !pulse) return;

  const start = { x: Number(line.getAttribute("x1")), y: Number(line.getAttribute("y1")) };
  const end = { x: Number(line.getAttribute("x2")), y: Number(line.getAttribute("y2")) };
  const started = performance.now();

  pulse.setAttribute("opacity", "1");

  function move(now) {
    if (token !== animationToken) {
      pulse.setAttribute("opacity", "0");
      group.classList.remove("active");
      return;
    }
    const progress = Math.min(1, (now - started) / duration);
    const x = start.x + (end.x - start.x) * progress;
    const y = start.y + (end.y - start.y) * progress;

    pulse.setAttribute("cx", x);
    pulse.setAttribute("cy", y);

    if (progress < 1) {
      requestAnimationFrame(move);
    } else {
      pulse.setAttribute("opacity", "0");
      group.classList.remove("active");
    }
  }

  requestAnimationFrame(move);
}

function showNeuron(neuron) {
  const info = document.getElementById("neuronInfo");
  if (!info) return;
  info.innerHTML = `
    <strong>Neuron: ${neuron.id}</strong>
    <dl>
      <dt>Activation</dt><dd>${format(neuron.activation)}</dd>
      <dt>Bias</dt><dd>${format(neuron.bias)}</dd>
      <dt>Received</dt><dd>${format(neuron.received)}</dd>
      <dt>Weighted sum</dt><dd>${format(neuron.weighted_sum)}</dd>
    </dl>
  `;
}

function showConnection(step) {
  const info = document.getElementById("connectionInfo");
  if (!info) return;
  info.innerHTML = `
    <strong>${step.source} → ${step.target}</strong>
    <dl>
      <dt>Signal</dt><dd>${format(step.signal)}</dd>
      <dt>Weight</dt><dd>${format(step.weight)}</dd>
      <dt>Weighted</dt><dd>${format(step.weighted_signal)}</dd>
    </dl>
  `;
}

function resetView() {
  clearScheduledPulseTimers();
  document.querySelectorAll(".svg-node, .connection-group").forEach((element) => {
    element.classList.remove("active", "isolated", "pulse", "hovered", "dragging", "fading", "death-output");
  });

  document.querySelectorAll(".signal-pulse").forEach((pulse) => {
    pulse.setAttribute("opacity", "0");
  });

  const neuronInfo = document.getElementById("neuronInfo");
  const connectionInfo = document.getElementById("connectionInfo");
  if (neuronInfo) neuronInfo.innerHTML = '<span class="muted">Ninguna neurona activa</span>';
  if (connectionInfo) connectionInfo.innerHTML = '<span class="muted">Esperando una conexión</span>';
}

function clearScheduledPulseTimers() {
  scheduledPulseTimeouts.forEach((timeoutId) => clearTimeout(timeoutId));
  scheduledPulseTimeouts = [];
}

function triggerDeathBurst() {
  animationToken += 1;
  isAnimating = false;
  brainQueue = [];
  clearScheduledPulseTimers();

  const nodes = [...document.querySelectorAll(".svg-node")];
  const outputNode = document.getElementById(nodeId("OUTPUT"));
  const inputNode = document.getElementById(nodeId("INPUT"));

  nodes.forEach((node) => {
    const isInput = node.id === nodeId("INPUT");
    const isOutput = node.id === nodeId("OUTPUT");
    node.classList.remove("fading", "isolated", "death-output");
    if (!isInput && !isOutput) {
      node.classList.add("active");
    }
  });

  document.querySelectorAll(".connection-group").forEach((group) => {
    group.classList.remove("fading");
    group.classList.add("active");
  });

  if (inputNode) {
    inputNode.classList.remove("active", "fading", "isolated", "death-output");
  }

  if (outputNode) {
    outputNode.classList.remove("active", "isolated", "fading");
    outputNode.classList.add("death-output");
  }
}

function stopBrainAnimation() {
  animationToken += 1;
  isAnimating = false;
  brainQueue = [];
  clearScheduledPulseTimers();
  resetView();
}

function activateNode(id, state = null) {
  const node = document.getElementById(nodeId(id));
  if (!node) return;

  const resolvedState = state || (isIsolatedNode(id) ? "isolated" : "active");
  node.classList.remove("active", "isolated");

  if (resolvedState === "isolated") {
    node.classList.add("isolated");
  } else if (resolvedState === "active") {
    node.classList.add("active");
  }
}

function isIsolatedNode(id) {
  if (!networkConfig || !id || id === networkConfig.input_node || id === networkConfig.output_node) {
    return false;
  }
  return !(networkConfig.connections || []).some((connection) => connection.source === id);
}

function appendLog(message) {
  const log = document.getElementById("logOutput");
  const line = document.createElement("div");
  line.textContent = `> ${message}`;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}

function setStatus(message) {
  document.getElementById("networkStatus").textContent = message;
  document.getElementById("logState").textContent = message;
}

function createSvg(tag, attributes = {}) {
  const element = document.createElementNS(SVG_NS, tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function connectionId(connection) {
  const source = safeId(connection.source);
  const target = safeId(connection.target);
  const index = connection.connection_index ?? connection.index ?? "";
  return `connection-${source}-${target}-${index}`;
}

function nodeId(id) {
  return `node-${safeId(id)}`;
}

function safeId(id) {
  return String(id).replace(/[^a-zA-Z0-9_-]/g, "-");
}

function format(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function wait(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
