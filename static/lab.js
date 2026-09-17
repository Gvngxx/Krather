let previousAlive = true;

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("labStart").addEventListener("click", () => sendControl("start"));
  document.getElementById("labStop").addEventListener("click", () => sendControl("stop"));
  document.getElementById("labReset").addEventListener("click", () => sendControl("reset"));
  document.getElementById("labGeneration").addEventListener("click", () => sendControl("generation"));
  document.getElementById("labAuto").addEventListener("click", toggleAuto);
  pollState();
});

async function sendControl(action) {
  const response = await fetch(`/api/lab/${action}`, { method: "POST" });
  if (!response.ok) return;
  const state = await response.json();
  updateWorld(state);
}

async function toggleAuto() {
  const button = document.getElementById("labAuto");
  const response = await fetch("/api/lab/auto", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled: button.getAttribute("aria-pressed") !== "true" }),
  });
  if (response.ok) updateWorld(await response.json());
}

async function pollState() {
  try {
    const response = await fetch("/api/lab/state");
    if (response.ok) updateWorld(await response.json());
  } finally {
    window.setTimeout(pollState, 120);
  }
}

function updateWorld(state) {
  const organism = state.organism;
  const food = state.food;
  const world = document.getElementById("lifeWorld");
  const organismElement = document.getElementById("organism");
  const foodElement = document.getElementById("food");
  const dead = organism.alive === false;
  organismElement.style.left = `${(organism.x / 860) * 100}%`;
  organismElement.style.top = `${(organism.y / 520) * 100}%`;
  organismElement.style.width = `${organism.size}px`;
  organismElement.style.height = `${organism.size}px`;
  foodElement.style.left = `${(food.x / 860) * 100}%`;
  foodElement.style.top = `${(food.y / 520) * 100}%`;
  world.classList.toggle("is-dead", dead);

  if (previousAlive && dead) {
    const burst = document.getElementById("deathBurst");
    burst.style.left = `${(organism.x / 860) * 100}%`;
    burst.style.top = `${(organism.y / 520) * 100}%`;
    burst.classList.remove("explode");
    void burst.offsetWidth;
    burst.classList.add("explode");
  }
  previousAlive = !dead;

  document.querySelectorAll(".organism-child").forEach((element) => element.remove());
  (state.organisms || []).slice(1).forEach((child) => {
    const childElement = document.createElement("div");
    childElement.className = `organism organism-child${child.alive ? "" : " is-dead"}`;
    childElement.style.left = `${(child.x / 860) * 100}%`;
    childElement.style.top = `${(child.y / 520) * 100}%`;
    childElement.style.width = `${child.size}px`;
    childElement.style.height = `${child.size}px`;
    world.appendChild(childElement);
  });

  document.getElementById("lifeState").textContent = state.status === "GENERATION_COMPLETE"
    ? `GENERATION COMPLETE / ${Number(state.next_generation_in || 0).toFixed(1)}s`
    : state.running ? "RUNNING" : dead ? "DEAD" : "STOPPED";
  document.getElementById("lifeOrganism").textContent = dead ? "DEAD" : "ALIVE";
  document.getElementById("lifeGeneration").textContent = state.generation;
  document.getElementById("lifeStep").textContent = state.tick;
  document.getElementById("lifeEnergy").textContent = Number(organism.energy).toFixed(1);
  document.getElementById("lifeBody").textContent = `${Number(organism.size).toFixed(1)} / ${Number(organism.weight).toFixed(1)}`;
  document.getElementById("lifeHunger").textContent = organism.no_food_ticks;
  document.getElementById("lifeOrganisms").textContent = state.organisms?.length || 1;
  document.getElementById("lifeReward").textContent = Number(state.reward).toFixed(1);
  document.getElementById("lifeTotalReward").textContent = Number(state.total_reward).toFixed(1);
  document.getElementById("lifeFoodEaten").textContent = state.food_eaten;
  document.getElementById("lifeWeightUpdates").textContent = state.learning.updates || 0;
  document.getElementById("lifeLearnedTicks").textContent = state.learning.total_ticks;
  const autoButton = document.getElementById("labAuto");
  autoButton.textContent = state.auto_generation ? "ON" : "OFF";
  autoButton.setAttribute("aria-pressed", String(Boolean(state.auto_generation)));
  const memory = state.memory || {};
  const latest = memory.last;
  const recalledAction = memory.latest_action || (latest && latest.action);
  const recalledResult = memory.latest_result || (latest && latest.result);
  document.getElementById("lifeMemoryCount").textContent = `${memory.count || 0} / ${memory.capacity || 0}`;
  document.getElementById("lifeMemoryUsage").textContent = `${((memory.used_ratio || 0) * 100).toFixed(0)}%`;
  document.getElementById("lifeMemoryReward").textContent = Number(memory.last_reward || 0).toFixed(2);
  document.getElementById("lifeMemoryAverage").textContent = Number(memory.average_recent_reward || 0).toFixed(2);
  document.getElementById("lifeMemorySignal").textContent = Number(memory.signal || 0).toFixed(2);
  document.getElementById("lifeMemoryRecall").textContent = Number(memory.recall_strength || 0).toFixed(2);
  document.getElementById("lifeMemoryValence").textContent = memory.valence || "NEUTRAL";
  document.getElementById("lifeMemoryAction").textContent = recalledAction ? formatAction(recalledAction) : "IDLE";
  document.getElementById("lifeMemoryResult").textContent = recalledResult ? formatMemoryResult(recalledResult) : "--";
  document.getElementById("lifeMemoryRecent").replaceChildren(...(memory.recent || []).slice().reverse().map((entry) => {
    const item = document.createElement("li");
    item.textContent = `G${entry.generation ?? state.generation}  ${Number(entry.reward || 0).toFixed(1)}  ${formatMemoryResult(entry.result)}`;
    return item;
  }));
}

function formatAction(action) {
  if (!action) return "IDLE";
  return `(${Number(action.move_x || 0).toFixed(2)}, ${Number(action.move_y || 0).toFixed(2)})`;
}

function formatMemoryResult(result) {
  if (!result) return "--";
  if (result.ate_food) return "FOOD";
  if (result.died) return "DEAD";
  return result.moved > 0.01 ? "MOVED" : "STILL";
}
