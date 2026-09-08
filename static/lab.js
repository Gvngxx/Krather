let previousAlive = true;

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("labStart").addEventListener("click", () => sendControl("start"));
  document.getElementById("labStop").addEventListener("click", () => sendControl("stop"));
  document.getElementById("labReset").addEventListener("click", () => sendControl("reset"));
  document.getElementById("labGeneration").addEventListener("click", () => sendControl("generation"));
  pollState();
});

async function sendControl(action) {
  const response = await fetch(`/api/lab/${action}`, { method: "POST" });
  if (!response.ok) return;
  const state = await response.json();
  updateWorld(state);
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

  document.getElementById("lifeState").textContent = state.running ? "RUNNING" : dead ? "DEAD" : "STOPPED";
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
}
