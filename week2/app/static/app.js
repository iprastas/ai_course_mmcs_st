/* Городская справочная: вход, SSE-стрим мыслей, карта, контекст, поиск */

const $ = (id) => document.getElementById(id);
let streaming = false;
let abortCtl = null;
let seenCalls = new Set();
let ctxLines = [];
let chatLog = [];            // для экспорта диалога
let answerEl = null;         // текущий пузырь стримящегося ответа
let answerRaw = "";          // его сырой markdown (для chatLog/final)
let lastFinalAnswer = "";    // последний финал — для follow-up вопросов

// экранирование HTML: спецсимволы заменяются на числовые сущности (код символа)
const esc = (s) => String(s ?? "")
  .replace(/[&<>"']/g, (c) => "&#" + c.charCodeAt(0) + ";");

/* ─────────────── markdown: жирный, код, списки, ТАБЛИЦЫ ─────────────── */

function inlineMd(s) {
  return esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function renderMd(text) {
  const lines = String(text ?? "").split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*\|/.test(line)) {                      // таблица: подряд идущие строки с |
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
      const cells = rows
        .filter((r) => !/^\s*\|[\s:|-]+\|\s*$/.test(r))     // пропуск |---|---|
        .map((r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
      if (cells.length) {
        const [head, ...body] = cells;
        out.push("<table><thead><tr>" +
          head.map((c) => `<th>${inlineMd(c)}</th>`).join("") +
          "</tr></thead><tbody>" +
          body.map((r) => "<tr>" + r.map((c) => `<td>${inlineMd(c)}</td>`).join("") + "</tr>").join("") +
          "</tbody></table>");
      }
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {                 // маркированный список
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i++].replace(/^\s*[-*]\s+/, ""));
      }
      out.push("<ul>" + items.map((t) => `<li>${inlineMd(t)}</li>`).join("") + "</ul>");
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {                // нумерованный список
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i++].replace(/^\s*\d+\.\s+/, ""));
      }
      out.push("<ol>" + items.map((t) => `<li>${inlineMd(t)}</li>`).join("") + "</ol>");
      continue;
    }
    if (line.trim() === "") { i++; continue; }
    const para = [];                                // абзац до пустой строки/таблицы/списка
    while (i < lines.length && lines[i].trim() !== "" &&
           !/^\s*\|/.test(lines[i]) && !/^\s*[-*]\s+/.test(lines[i]) &&
           !/^\s*\d+\.\s+/.test(lines[i])) {
      para.push(lines[i++]);
    }
    out.push(`<p>${para.map(inlineMd).join("<br>")}</p>`);
  }
  return out.join("");
}

/* ─────────────── тема ─────────────── */

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = $("btn-theme");
  if (btn) btn.textContent = theme === "dark" ? "☀️" : "🌙";
  localStorage.setItem("theme", theme);
}

/* ─────────────── модальные окна (вместо alert/confirm/prompt браузера) ─────────────── */

function openModal({ title, text, input, okText = "OK", cancelText = "Отмена",
                     danger = false, alertMode = false }) {
  return new Promise((resolve) => {
    const dlg = $("modal");
    $("modal-title").textContent = title || "";
    $("modal-text").textContent = text || "";
    const inp = $("modal-input");
    const hasInput = input !== undefined && input !== null;
    inp.classList.toggle("hidden", !hasInput);
    inp.value = hasInput ? String(input) : "";
    const ok = $("modal-ok");
    const cancel = $("modal-cancel");
    ok.textContent = okText;
    cancel.textContent = cancelText;
    cancel.classList.toggle("hidden", !!alertMode);
    ok.classList.toggle("danger", !!danger);
    let settled = false;
    const done = (val) => {
      if (settled) return;
      settled = true;
      dlg.close();
      resolve(val);
    };
    ok.onclick = () => done(hasInput ? inp.value : true);
    cancel.onclick = () => done(null);
    dlg.onclose = () => done(null);          // Esc / клик по backdrop
    dlg.showModal();
    if (hasInput) { inp.focus(); inp.select(); }
    else ok.focus();
  });
}

function confirmDialog(text, { title = "Подтверждение", okText = "OK", danger = false } = {}) {
  return openModal({ title, text, okText, danger }).then((v) => v === true);
}

function promptDialog(text, initial = "") {
  return openModal({ title: text, input: initial, okText: "Сохранить" });
}

function alertDialog(text, { title = "Внимание" } = {}) {
  return openModal({ title, text, alertMode: true, okText: "Понятно" });
}

/* ─────────────── экраны ─────────────── */

function showLogin(msg) {
  $("app").classList.add("hidden");
  $("login-screen").classList.remove("hidden");
  const err = $("login-error");
  if (msg) { err.textContent = msg; err.classList.remove("hidden"); }
}

function showApp(login) {
  $("login-screen").classList.add("hidden");
  $("app").classList.remove("hidden");
  if (login) $("who").textContent = "👤 " + login;
  initMap();
}

/* ─────────────── чат и лента ─────────────── */

function appendChat(kind, html, rawText) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + kind;
  wrap.innerHTML = `<div class="bubble">${html}</div>`;
  if (kind === "assistant" && rawText) {
    wrap.dataset.raw = rawText;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn";
    btn.textContent = "⧉";
    btn.title = "Копировать ответ";
    wrap.appendChild(btn);
  }
  $("chat").appendChild(wrap);
  $("chat").scrollTop = $("chat").scrollHeight;
  if (rawText !== undefined) {
    chatLog.push({ role: kind === "user" ? "user" : "assistant", text: rawText });
  }
  return wrap;
}

// копирование ответа (кнопка на пузыре, raw-text в dataset)
$("chat").addEventListener("click", async (e) => {
  const btn = e.target.closest(".copy-btn");
  if (!btn) return;
  const wrap = btn.closest(".msg");
  try {
    await navigator.clipboard.writeText(wrap.dataset.raw || "");
    btn.textContent = "✓";
    setTimeout(() => { btn.textContent = "⧉"; }, 1200);
  } catch (_) {
    btn.textContent = "…";
  }
});

function setTyping(on) {
  let el = $("typing");
  if (on) {
    if (!el) {
      el = document.createElement("div");
      el.id = "typing";
      el.className = "msg assistant";
      el.innerHTML = `<div class="bubble typing"><span class="dots"><i>.</i><i>.</i><i>.</i></span></div>`;
      $("chat").appendChild(el);
    }
    $("chat").scrollTop = $("chat").scrollHeight;
  } else if (el) {
    el.remove();
  }
}

function clearStream() {
  $("stream").innerHTML =
    '<div class="muted small">здесь появятся шаги: мысли → вызовы инструментов → результаты</div>';
  seenCalls = new Set();
  lastThink = null;
  answerEl = null;
  answerRaw = "";
}

function agentBadge(agent) {
  if (!agent) return "";
  const icon = agent === "researcher" ? "🔬" : "🎯";
  return ` <span class="agent-badge" title="${esc(agent)}">${icon}</span>`;
}

function addStep(cls, tag, bodyHtml, preText, tSec, agent) {
  const div = document.createElement("div");
  div.className = "step " + cls;
  const time = tSec !== undefined ? ` <span class="t">+${tSec}с</span>` : "";
  div.innerHTML = `<span class="tag">${tag}${agentBadge(agent)}${time}</span>` +
    `<div class="body">${bodyHtml}</div>` +
    (preText ? `<pre>${esc(preText)}</pre><span class="hint">нажми, чтобы развернуть</span>` : "");
  div.addEventListener("click", () => div.classList.toggle("open"));
  $("stream").appendChild(div);
  $("stream").scrollTop = $("stream").scrollHeight;
  $("stream-count").textContent = "· шагов: " + document.querySelectorAll("#stream .step").length;
  return div;
}

let lastThink = null;

function pushThinking(text, tSec, agent) {
  // новый агент думает — начинаем новый блок, чтобы бейдж был честным
  if (!lastThink || !lastThink.isConnected ||
      (agent && lastThink.dataset.agent !== agent)) {
    lastThink = addStep("think", "🧠 мысль", "", null, tSec, agent);
    lastThink.dataset.agent = agent || "";
  }
  lastThink.querySelector(".body").textContent += text;
  $("stream").scrollTop = $("stream").scrollHeight;
}

function renderEvent(ev) {
  switch (ev.type) {
    case "wait":
      addStep("think", "⏳ в очереди",
        `свободных прогонов нет — вы ${ev.active + 1}-й из ~${ev.limit + ev.active}`,
        null, ev.t);
      break;
    case "thinking":
      pushThinking(ev.text, ev.t, ev.agent);
      break;
    case "delegate": {
      lastThink = null;
      addStep("call", "⇄ делегирование",
        `<code>${esc(ev.from)}</code> → <code>${esc(ev.to)}</code>`,
        null, ev.t, ev.from);
      break;
    }
    case "tool_call": {
      if (ev.id && seenCalls.has(ev.id)) return;
      if (ev.id) seenCalls.add(ev.id);
      lastThink = null;
      const args = JSON.stringify(ev.args ?? {}, null, 2);
      addStep("call", "🔧 вызов инструмента",
        `<code>${esc(ev.name)}</code> <span class="hint">${esc(shortArgs(ev.args))}</span>`,
        args, ev.t, ev.agent);
      break;
    }
    case "tool_result": {
      lastThink = null;
      const first = String(ev.result ?? "").split("\n")[0];
      addStep("result", "↩ результат " + esc(ev.name),
        `<code>${esc(first.slice(0, 120))}</code>`, String(ev.result ?? ""),
        ev.t, ev.agent);
      break;
    }
    case "map_points":
      addMapPoints(ev.points || []);
      break;
    case "answer_delta": {
      lastThink = null;
      answerRaw += ev.text || "";
      if (!answerEl || !answerEl.isConnected) {
        answerEl = appendChat("assistant", "", answerRaw);
        answerRaw = answerEl.querySelector(".bubble") ? "" : answerRaw;
        // appendChat уже пушит rawText в chatLog — перезапишем ниже при final
      }
      const bubble = answerEl.querySelector(".bubble");
      if (bubble) bubble.innerHTML = renderMd(answerRaw);
      $("chat").scrollTop = $("chat").scrollHeight;
      break;
    }
    case "final": {
      lastThink = null;
      const answer = String(ev.answer ?? "");
      if (answerEl && answerEl.isConnected && answer) {
        // дописываем/перезаписываем стримящийся пузырь итоговым текстом
        answerEl.querySelector(".bubble").innerHTML = renderMd(answer);
        answerEl.dataset.raw = answer;
        // в chatLog последняя запись assistant пузыря могла быть пустой — чиним
        for (let i = chatLog.length - 1; i >= 0; i--) {
          if (chatLog[i].role === "assistant") { chatLog[i].text = answer; break; }
          if (chatLog[i].role === "user") break;
        }
        answerEl = null;
        answerRaw = "";
      } else {
        appendChat("assistant", renderMd(answer), answer);
      }
      lastFinalAnswer = answer;
      pushCtx("Ответ: " + answer.replace(/\s+/g, " ").slice(0, 70) + "…");
      break;
    }
    case "usage": {
      const b = $("usage-badge");
      b.textContent = ` шагов ${ev.steps} · ${ev.input_tokens + ev.output_tokens} токенов · ${ev.cost_rub} ₽ · ${ev.seconds} с`;
      b.classList.remove("hidden");
      break;
    }
    case "title": {
      // LLM-название чата (короткий смысл первого вопроса) — обновляем сайдбар
      const c = chats.find((x) => x.id === activeChatId);
      if (c && ev.title) { c.title = ev.title; renderChats(); }
      else if (ev.title) loadChats();   // чат родился из черновика в этом прогоне
      break;
    }
    case "error":
      lastThink = null;
      addStep("error", "⚠ ошибка", esc(ev.message), null, ev.t);
      break;
  }
}

function shortArgs(args) {
  if (!args) return "";
  const s = Object.entries(args).map(([k, v]) => `${k}=${v}`).join(", ");
  return s.length > 70 ? s.slice(0, 70) + "…" : s;
}

function pushCtx(text) {
  ctxLines.push(text);
  $("ctx-count").textContent = ctxLines.length;
  const box = $("context");
  box.classList.remove("muted");
  box.innerHTML = ctxLines.map((l) => `<div class="line">${esc(l)}</div>`).join("");
  box.scrollTop = box.scrollHeight;
}

/* ─────────────── карта (Leaflet) ─────────────── */

let mapLayers = new Map();   // id → layer (не дублируем точки)

// базовые слои карты: открытый OSM (основной), бесплатный спутник Esri,
// топокарта OpenTopoMap — все работают из России без ключей
const TILE_PROVIDERS = {
  streets: () => L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a>',
    maxZoom: 19,
  }),
  satellite: () => L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      attribution: 'Спутник &copy; Esri, Maxar, Earthstar Geographics',
      maxZoom: 19,
    }),
  topo: () => L.tileLayer("https://tile.opentopomap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a>, SRTM | &copy; OpenTopoMap (CC&nbsp;BY&nbsp;SA)',
    maxZoom: 17,
  }),
};
let activeTileLayer = null;
let activeTileKey = "streets";
let fallbackTileLayer = null;

function setMapLayer(key) {
  if (!window._map || !TILE_PROVIDERS[key]) return;
  if (activeTileLayer) activeTileLayer.remove();
  if (fallbackTileLayer) { fallbackTileLayer.remove(); fallbackTileLayer = null; }
  activeTileLayer = TILE_PROVIDERS[key]().addTo(window._map);
  activeTileKey = key;
  document.querySelectorAll("#map-layers .chip-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.layer === key));
}

function initMap() {
  if (window._map || typeof L === "undefined") return;
  window._map = L.map("map", { scrollWheelZoom: false }).setView([47.2357, 39.7015], 12);
  activeTileLayer = TILE_PROVIDERS.streets().addTo(window._map);
  activeTileKey = "streets";
  // резерв основного слоя: серия ошибок загрузки тайлов → зеркало OSM
  let tileErrors = 0;
  window._map.on("tileerror", () => {
    if (activeTileKey !== "streets" || fallbackTileLayer) return;
    tileErrors += 1;
    if (tileErrors < 6) return;
    activeTileLayer.remove();
    fallbackTileLayer = L.tileLayer("https://tile.openstreetmap.de/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a>',
      maxZoom: 18,
    }).addTo(window._map);
  });
  window._map.invalidateSize();
}

function addMapPoints(points) {
  if (!window._map) return;
  const bounds = [];
  for (const p of points) {
    if (mapLayers.has(p.id)) continue;
    const m = L.circleMarker([p.lat, p.lon], {
      radius: 8, color: "#4f8cff", weight: 2, fillColor: "#7c5cff", fillOpacity: 0.85,
    }).addTo(window._map);
    m.bindPopup(`<strong>${esc(p.name)}</strong><br><span class="muted">${esc(p.kind)} · id ${p.id}</span>`);
    mapLayers.set(p.id, m);
    bounds.push([p.lat, p.lon]);
  }
  $("map-count").textContent = mapLayers.size ? `· ${mapLayers.size}` : "";
  if (bounds.length) window._map.fitBounds(bounds, { padding: [30, 30], maxZoom: 15 });
}

function clearMap() {
  if (window._map) {
    for (const m of mapLayers.values()) m.remove();
  }
  mapLayers.clear();
  $("map-count").textContent = "";
}

/* ─────────────── история (контекст после F5 и «Новый диалог») ─────────────── */

function restore(events) {
  ctxLines = []; $("ctx-count").textContent = "0";
  $("context").innerHTML = "история пуста";
  $("context").classList.add("muted");
  clearStream();
  clearMap();
  chatLog = [];
  // убираем все сообщения, кроме приветствия (сброс без перезагрузки страницы)
  for (const el of document.querySelectorAll("#chat .msg:not(:first-child)")) el.remove();
  let pendingAnswer = false;
  const markUnanswered = () => {
    if (!pendingAnswer) return;
    pendingAnswer = false;
    const wrap = document.createElement("div");
    wrap.className = "msg assistant";
    wrap.innerHTML = '<div class="bubble muted small">ответ не был сохранён — прогон прерван</div>';
    $("chat").appendChild(wrap);
  };
  for (const ev of events) {
    if (ev.type === "question") {
      markUnanswered();                  // предыдущий вопрос остался без ответа
      appendChat("user", esc(ev.text), ev.text);
      pushCtx("Вопрос: " + ev.text.slice(0, 70));
      pendingAnswer = true;
    } else if (ev.type === "final") {
      appendChat("assistant", renderMd(ev.answer), ev.answer);
      pushCtx("Ответ: " + ev.answer.replace(/\s+/g, " ").slice(0, 70) + "…");
      pendingAnswer = false;
    } else if (ev.type === "map_points") {
      addMapPoints(ev.points || []);
    } else {
      lastThink = null;
      renderEvent(ev);
    }
  }
  markUnanswered();
}

/* ─────────────── запрос к агенту (SSE через fetch + СТОП) ─────────────── */

function setStreaming(on) {
  streaming = on;
  const btn = $("btn-ask");
  btn.textContent = on ? "■ Стоп" : "Спросить";
  btn.classList.toggle("danger", on);
  $("status").classList.toggle("hidden", !on);
  setTyping(on);
}

async function ask(question) {
  abortCtl = new AbortController();
  setStreaming(true);
  appendChat("user", esc(question), question);
  pushCtx("Вопрос: " + question.slice(0, 70));
  clearStream();
  setStreaming(true);   // ещё раз: clearStream мог снять индикаторы
  answerRaw = "";
  lastFinalAnswer = "";
  $("followups").classList.add("hidden");
  $("followups").innerHTML = "";
  let completed = false;

  try {
    const resp = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: abortCtl.signal,
    });
    if (resp.status === 401) { showLogin("Войдите заново"); return; }
    if (resp.status === 422) {
      const err = await resp.json().catch(() => ({}));
      addStep("error", "⚠ вопрос не принят",
        esc(err.error || "слишком длинный или пустой вопрос (макс. 2000 символов)"));
      return;
    }
    if (!resp.ok) throw new Error("HTTP " + resp.status);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of chunk.split("\n")) {
          if (line.startsWith("data: ")) {
            try { renderEvent(JSON.parse(line.slice(6))); } catch (_) { /* игнор */ }
          }
        }
      }
    }
    completed = true;
  } catch (e) {
    if (e.name === "AbortError") {
      addStep("error", "⏹ прогон остановлен", "пользователь прервал агента");
    } else {
      addStep("error", "⚠ обрыв соединения", esc(String(e)));
    }
  } finally {
    abortCtl = null;
    setStreaming(false);
    loadChats();                     // автотайтл и время чата обновились
  }
  // продолжения диалога: fire-and-forget, при ошибке просто не покажутся
  if (completed && lastFinalAnswer) {
    requestFollowups(question, lastFinalAnswer);
  }
}

/* ─────────────── follow-up чипы после ответа ─────────────── */

function requestFollowups(question, answer) {
  fetch("/api/followups", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, answer }),
  })
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => {
      if (!d || !Array.isArray(d.suggestions) || !d.suggestions.length) return;
      if (streaming) return;          // начался новый запрос — не мешаем
      const box = $("followups");
      box.innerHTML = d.suggestions.map(
        (s) => `<button type="button" class="chip-btn">${esc(s)}</button>`
      ).join("");
      box.classList.remove("hidden");
      box.querySelectorAll(".chip-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          if (streaming) return;
          $("question").value = btn.textContent.trim();
          $("ask-form").requestSubmit();
        });
      });
    })
    .catch(() => { /* тихо */ });
}

/* ─────────────── быстрый семантический поиск ─────────────── */

let searchTimer = null;
let searchCtl = null;       // отмена предыдущего запроса при новом вводе

async function quickSearch(q) {
  const box = $("search-results");
  if (q.trim().length < 3) {
    if (searchCtl) searchCtl.abort();
    box.className = "search-results muted";
    box.textContent = "введите фразу — вернётся топ по косинусу";
    return;
  }
  if (searchCtl) searchCtl.abort();
  searchCtl = new AbortController();
  box.textContent = "ищем…";
  try {
    const r = await fetch("/api/search?q=" + encodeURIComponent(q),
                          { signal: searchCtl.signal });
    if (r.status === 401) return;
    const data = await r.json();
    if (data.error) { box.textContent = "ошибка: " + data.error; return; }
    if (!data.results.length) {
      box.className = "search-results muted";
      box.textContent = "ничего похожего не нашлось";
      return;
    }
    box.className = "search-results";
    box.innerHTML = data.results.map((h) => `
      <div class="hit">
        <div><strong>${esc(h.name || "без названия")}</strong>
          <div class="sub">${esc(h.kind)}${h.cuisine ? " · " + esc(h.cuisine) : ""}${h.addr ? " · " + esc(h.addr) : ""}</div>
        </div>
        <div class="score" title="косинусное сходство с запросом">${(h.score * 100).toFixed(0)}%</div>
      </div>`).join("");
  } catch (e) {
    if (e.name === "AbortError") return;   // запрос отменён — не мешаем новому
    box.textContent = "ошибка поиска";
  } finally {
    searchCtl = null;
  }
}

/* ─────────────── мультичаты (сайдбар диалогов) ─────────────── */

let chats = [];
let activeChatId = null;

async function loadChats() {
  try {
    const r = await fetch("/api/chats");
    if (!r.ok) return;
    const d = await r.json();
    chats = d.chats || [];
    activeChatId = d.active || null;
    renderChats();
  } catch (_) { /* сайдбар не критичен */ }
}

function fmtChatDate(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleDateString("ru-RU", { day: "numeric", month: "short" }) +
    " " + d.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });
}

function renderChats() {
  const list = $("chat-list");
  list.innerHTML = chats.length
    ? chats.map((c) => `
      <div class="chat-item${c.id === activeChatId ? " active" : ""}" data-id="${esc(c.id)}">
        <span class="chat-title">${esc(c.title)}</span>
        <span class="chat-date muted">${fmtChatDate(c.updated_at)}</span>
        <span class="chat-actions">
          <button type="button" class="chat-btn chat-rename" title="Переименовать">✎</button>
          <button type="button" class="chat-btn chat-delete" title="Удалить">🗑</button>
        </span>
      </div>`).join("")
    : '<div class="muted small">чатов пока нет</div>';
}

async function reloadHistory() {
  const h = await fetch("/api/history");
  if (!h.ok) {
    if (h.status === 401) showLogin("Войдите заново");
    return;
  }
  restore((await h.json()).events || []);
}

async function newChat() {
  if (streaming) return;
  // новый чат — черновик без сессии: контекст создастся при первом вопросе,
  // а пустой чат не появится в списке (в отличие от старого поведения)
  await fetch("/api/chats", { method: "POST" });
  activeChatId = null;
  renderChats();
  restore([]);
  $("question").focus();
}

async function activateChat(id) {
  if (streaming || id === activeChatId) return;
  const r = await fetch(`/api/chats/${encodeURIComponent(id)}/activate`,
                        { method: "POST" });
  if (!r.ok) { await alertDialog("Не удалось открыть чат", { title: "Ошибка" }); return; }
  activeChatId = id;
  renderChats();
  await reloadHistory();
}

$("chat-list").addEventListener("click", async (e) => {
  const item = e.target.closest(".chat-item");
  if (!item || streaming) return;
  const id = item.dataset.id;

  if (e.target.closest(".chat-rename")) {
    const cur = chats.find((c) => c.id === id);
    const title = await promptDialog("Название чата:", cur ? cur.title : "");
    if (!title || !title.trim()) return;
    const r = await fetch(`/api/chats/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: title.trim().slice(0, 80) }),
    });
    if (r.ok) await loadChats();
    else await alertDialog("Не удалось переименовать чат", { title: "Ошибка" });
    return;
  }

  if (e.target.closest(".chat-delete")) {
    const ok = await confirmDialog(
      "Диалог и вся его история будут удалены безвозвратно.",
      { title: "Удаление чата", okText: "Удалить", danger: true });
    if (!ok) return;
    const r = await fetch(`/api/chats/${encodeURIComponent(id)}`,
                          { method: "DELETE" });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      await alertDialog(err.error || "Не удалось удалить чат", { title: "Ошибка" });
      return;
    }
    const wasActive = id === activeChatId;
    if (wasActive) {                 // удалили открытый чат — показываем черновик
      activeChatId = null;
      await reloadHistory();
    }
    await loadChats();
    return;
  }

  await activateChat(id);
});

$("btn-new-chat").addEventListener("click", newChat);

// переключатель базовых слоёв карты: улицы / спутник / топо
$("map-layers").addEventListener("click", (e) => {
  const btn = e.target.closest(".chip-btn");
  if (btn && btn.dataset.layer) setMapLayer(btn.dataset.layer);
});

/* ─────────────── экспорт диалога ─────────────── */

function exportDialog() {
  if (!chatLog.length) {
    alertDialog("Диалог пуст — нечего экспортировать");
    return;
  }
  const head = `# Диалог — Городская справочная Ростова\n\n${new Date().toLocaleString("ru-RU")}\n\n`;
  const body = chatLog.map((e) =>
    e.role === "user" ? `**Вопрос:** ${e.text}` : `**Ответ:**\n\n${e.text}`
  ).join("\n\n---\n\n");
  const blob = new Blob([head + body], { type: "text/markdown;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `dialog-${new Date().toISOString().slice(0, 10)}.md`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ─────────────── события страницы ─────────────── */

/* ─────────────── вход / регистрация (форма + демо-вход) ─────────────── */

async function enterApp(login) {
  showApp(login);
  const h = await fetch("/api/history");
  restore((await h.json()).events || []);
  loadChats();
}

let registerMode = false;

function applyAuthMode() {
  $("login-confirm-row").classList.toggle("hidden", !registerMode);
  $("login-confirm").required = registerMode;
  $("login-submit").textContent = registerMode ? "Зарегистрироваться" : "Войти";
  $("btn-toggle-register").textContent = registerMode
    ? "Уже есть аккаунт? Войти"
    : "Нет аккаунта? Регистрация";
  $("login-pass").autocomplete = registerMode ? "new-password" : "current-password";
  $("login-error").classList.add("hidden");
}

$("btn-toggle-register").addEventListener("click", () => {
  registerMode = !registerMode;
  applyAuthMode();
});

$("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const login = $("login-name").value.trim();
  const password = $("login-pass").value;
  if (registerMode) {
    if (password.length < 6) {
      showLogin("Пароль — минимум 6 символов");
      return;
    }
    if (password !== $("login-confirm").value) {
      showLogin("Пароли не совпадают");
      return;
    }
    const r = await fetch("/api/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ login, password }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      showLogin(err.error || "Не удалось зарегистрироваться");
      return;
    }
    await enterApp((await r.json()).login);
    return;
  }
  const r = await fetch("/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ login, password }),
  });
  if (!r.ok) {
    // 401 — неверные данные; 429 — rate-limit: показываем реальную причину
    const err = await r.json().catch(() => ({}));
    showLogin(err.error || "Неверный логин или пароль");
    return;
  }
  const data = await r.json();
  await enterApp(data.login);
});

$("btn-demo").addEventListener("click", async () => {
  const r = await fetch("/login/demo", { method: "POST" });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    showLogin(err.error || "Демо-вход недоступен");
    return;
  }
  const data = await r.json();
  await enterApp(data.login);
});

$("btn-logout").addEventListener("click", async () => {
  await fetch("/logout", { method: "POST" });
  location.reload();
});

$("btn-reset").addEventListener("click", newChat);   // «Новый диалог» = новый чат

$("btn-export").addEventListener("click", exportDialog);
$("btn-theme").addEventListener("click", () =>
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  if (streaming) {                       // кнопка в режиме «Стоп»
    if (abortCtl) abortCtl.abort();
    return;
  }
  const q = $("question").value.trim();
  if (!q) return;
  $("question").value = "";
  $("question").style.height = "auto";
  localStorage.removeItem("draft");      // вопрос отправлен — черновик чист
  ask(q);
});

$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("ask-form").requestSubmit();
  }
});
$("question").addEventListener("input", (e) => {   // автогроу + черновик
  e.target.style.height = "auto";
  e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
  localStorage.setItem("draft", e.target.value);
});

// чипы примеров: клик → в поле и сразу отправка
document.querySelectorAll("#chips .chip-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (streaming) return;
    $("question").value = btn.textContent.trim();
    $("ask-form").requestSubmit();
  });
});

$("search-q").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  const q = e.target.value;
  searchTimer = setTimeout(() => quickSearch(q), 400);
});
$("search-q").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { clearTimeout(searchTimer); quickSearch(e.target.value); }
});

/* ─────────────── старт ─────────────── */

applyTheme(localStorage.getItem("theme") || "dark");
applyAuthMode();

(async function boot() {
  const r = await fetch("/api/history");
  if (r.status === 401) { showLogin(""); return; }
  const data = await r.json();
  showApp(data.login || "");
  restore(data.events || []);
  loadChats();
  // черновик неотправленного вопроса пережил F5
  const draft = localStorage.getItem("draft");
  if (draft) {
    $("question").value = draft;
    $("question").style.height = "auto";
    $("question").style.height = Math.min($("question").scrollHeight, 120) + "px";
  }
})();
