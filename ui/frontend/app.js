"use strict";

const app = document.getElementById("app");
const crumb = document.getElementById("crumb");
const conn = document.getElementById("conn");
let stopView = null;  // cleanup for the current view

const ICONS = {
  dishwashing: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">' +
    '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/></svg>',
};

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw Object.assign(new Error(body.error || response.statusText), { body });
  return body;
}

function show(templateId) {
  app.replaceChildren(document.getElementById(templateId).content.cloneNode(true));
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// ---- Home ------------------------------------------------------------------
async function renderHome() {
  show("view-home");
  crumb.textContent = "";
  const list = document.getElementById("usecases");
  try {
    for (const useCase of await api("/api/usecases")) {
      const card = el("button", "usecase");
      card.type = "button";
      const icon = el("span", "usecase-icon");
      icon.innerHTML = ICONS[useCase.id] || "";
      card.append(icon, el("h3", "", useCase.name), el("p", "", useCase.description), el("span", "open", "Open →"));
      card.addEventListener("click", () => { location.hash = `#/${useCase.id}`; });
      list.append(card);
    }
    conn.hidden = true;
  } catch (error) {
    conn.hidden = false;
  }
}

// ---- Dishwashing -------------------------------------------------------------
const STATUS_TEXT = {
  idle: "Idle", starting: "Starting", running: "Running", paused: "Paused",
  stopping: "Stopping", stopped: "Stopped", finished: "Finished", error: "Error",
};

function renderDishwashing() {
  show("view-dishwashing");
  crumb.textContent = "Dishwashing";
  const $ = (id) => document.getElementById(id);
  const stream = $("stream");
  const camera = stream.parentElement;
  stream.src = `/api/camera/stream.mjpg?t=${Date.now()}`;
  let latestState = null;
  let busy = false;

  async function command(action) {
    if (busy) return;
    busy = true;
    try {
      update(await api(`/api/dishwashing/${action}`, { method: "POST" }));
    } catch (error) {
      if (error.body && error.body.state) update(error.body.state);
      showMessage(error.message, false);
    } finally {
      busy = false;
    }
  }

  function showMessage(text, warn) {
    const box = $("message");
    box.hidden = !text;
    box.textContent = text || "";
    box.classList.toggle("warn", Boolean(warn));
  }

  function update(state) {
    latestState = state;
    const active = ["starting", "running", "paused", "stopping"].includes(state.status);
    camera.classList.toggle("online", state.camera_online);
    $("camera-offline").textContent = state.camera_online ? "" : "Waiting for camera…";

    const status = $("status");
    status.dataset.status = state.status;
    status.textContent = STATUS_TEXT[state.status] || state.status;
    $("step").textContent = state.step;
    $("cycle").textContent = state.cycle ? `Plate ${state.cycle} of ${state.cycles_total}` : `${state.cycles_total} plates per run`;

    $("btn-start").disabled = active;
    const pause = $("btn-pause");
    pause.disabled = !["running", "paused"].includes(state.status);
    pause.querySelector("span").textContent = state.status === "paused" ? "Resume" : "Pause";
    $("btn-stop").disabled = !["starting", "running", "paused"].includes(state.status);
    $("btn-reset").disabled = active;
    if (state.status === "error") showMessage(state.message, false);
    else showMessage(state.message, true);

    const latest = state.latest;
    const verdict = $("verdict");
    verdict.dataset.label = latest ? latest.label : "";
    verdict.textContent = latest ? (latest.label === "clean" ? "Clean" : "Dirty") : "No plate yet";
    $("latest-meta").textContent = latest ? `${latest.time} · ${latest.pick}` : "";
    const fill = $("meter-fill");
    fill.style.width = latest ? `${(latest.score * 100).toFixed(1)}%` : "0";
    fill.className = `meter-fill ${latest ? latest.label : ""}`;
    $("meter-cut").style.left = `${state.threshold * 100}%`;
    $("score").textContent = latest ? `Dirty score ${latest.score.toFixed(3)}` : "Dirty score —";

    const { clean, dirty } = state.counts;
    $("n-clean").textContent = clean;
    $("n-dirty").textContent = dirty;
    $("n-total").textContent = clean + dirty;
    $("ratio-clean").style.flexGrow = clean;
    $("ratio-dirty").style.flexGrow = dirty;

    const history = $("history");
    history.replaceChildren(...state.history.map((item) => {
      const row = el("li");
      row.append(el("span", `tag ${item.label}`, item.label === "clean" ? "Clean" : "Dirty"),
        el("span", "where", `${item.time} · ${item.pick}`), el("span", "val", item.score.toFixed(3)));
      return row;
    }));
    $("history-empty").hidden = state.history.length > 0;

    const sequence = $("sequence");
    if (!sequence.childElementCount) {
      const s = state.sequence;
      const steps = [s.observe, s.picks.join(" / "), s.lift, s.via, s.drop, s.via, s.observe];
      steps.forEach((name, i) => {
        if (i) sequence.append(el("li", "sep", "→"));
        sequence.append(el("li", "", name));
      });
    }
  }

  $("btn-start").addEventListener("click", () => command("start"));
  $("btn-pause").addEventListener("click", () => command(latestState && latestState.status === "paused" ? "resume" : "pause"));
  $("btn-stop").addEventListener("click", () => command("stop"));
  $("btn-reset").addEventListener("click", () => command("reset"));

  async function poll() {
    try {
      update(await api("/api/dishwashing/state"));
      conn.hidden = true;
    } catch (error) {
      conn.hidden = false;
    }
  }
  poll();
  const timer = setInterval(poll, 500);
  return () => {
    clearInterval(timer);
    stream.removeAttribute("src");  // closes the camera stream connection
  };
}

// ---- Router -------------------------------------------------------------------
function route() {
  if (stopView) stopView();
  stopView = null;
  const page = location.hash.replace(/^#\/?/, "");
  if (page === "dishwashing") stopView = renderDishwashing();
  else renderHome();
}

window.addEventListener("hashchange", route);
route();
