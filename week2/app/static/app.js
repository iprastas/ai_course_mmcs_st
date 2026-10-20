/* Городская справочная: вход, SSE-стрим мыслей агента, контекст, поиск */

const $ = (id) => document.getElementById(id);
let streaming = false;
let seenCalls = new Set();
let ctxLines = [];

const esc = (s) => String(s ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

function renderMd(text) {
  let h = esc(text);
  h = h.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  h = h.replace(/`([^`]+)`/g, "<code>$1</code>");
  return h.replace(/\n/g, "<br>");
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
}

/* ─────────────── чат и лента ─────────────── */

function appendChat(kind, html) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + kind;
  wrap.innerHTML = `<div class="bubble">${html}</div>`;
  $("chat").appendChild(wrap);
  $("chat").scrollTop = $("chat").scrollHeight;
  return wrap;
}

function clearStream() {
  $("stream").innerHTML = "";
  seenCalls = new Set();
}

function addStep(cls, tag, bodyHtml, preText) {
  const div = document.createElement("div");
  div.className = "step " + cls;
  div.innerHTML = `<span class="tag">${tag}</span><div class="body">${bodyHtml}</div>` +
    (preText ? `<pre>${esc(preText)}</pre><span class="hint">нажми, чтобы развернуть</span>` : "");
  if (preText) div.addEventListener("click", () => div.classList.toggle("open"));
  $("stream").appendChild(div);
  $("stream").scrollTop = $("stream").scrollHeight;
  $("stream-count").textContent = "· шагов: " + document.querySelectorAll("#stream .step").length;
  return div;
}

let lastThink = null;

function pushThinking(text) {
  if (!lastThink || !lastThink.isConnected) {
    lastThink = addStep("think", "🧠 мысль", "");
  }
  lastThink.querySelector(".body").textContent += text;
  $("stream").scrollTop = $("stream").scrollHeight;
}

function renderEvent(ev) {
  switch (ev.type) {
    case "thinking":
      pushThinking(ev.text);
      break;
    case "tool_call": {
      if (ev.id && seenCalls.has(ev.id)) return;
      if (ev.id) seenCalls.add(ev.id);
      lastThink = null;
      const args = JSON.stringify(ev.args ?? {}, null, 2);
      addStep("call", "🔧 вызов инструмента",
        `<code>${esc(ev.name)}</code> <span class="hint">${esc(shortArgs(ev.args))}</span>`, args);
      break;
    }
    case "tool_result": {
      lastThink = null;
      const first = String(ev.result ?? "").split("\n")[0];
      addStep("result", "↩ результат " + esc(ev.name),
        `<code>${esc(first.slice(0, 120))}</code>`, String(ev.result ?? ""));
      break;
    }
    case "final":
      lastThink = null;
      appendChat("assistant", renderMd(ev.answer));
      pushCtx("Ответ: " + String(ev.answer).replace(/\s+/g, " ").slice(0, 70) + "…");
      break;
    case "usage": {
      const b = $("usage-badge");
      b.textContent = ` шагов ${ev.steps} · ${ev.input_tokens + ev.output_tokens} токенов · ${ev.cost_rub} ₽ · ${ev.seconds} с`;
      b.classList.remove("hidden");
      break;
    }
    case "error":
      lastThink = null;
      addStep("error", "⚠ ошибка", esc(ev.message));
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

/* ─────────────── история (контекст после F5) ─────────────── */

function restore(events) {
  ctxLines = []; $("ctx-count").textContent = "0";
  $("context").innerHTML = "история пуста";
  $("context").classList.add("muted");
  clearStream();
  for (const ev of events) {
    if (ev.type === "question") {
      appendChat("user", esc(ev.text));
      pushCtx("Вопрос: " + ev.text.slice(0, 70));
    } else if (ev.type === "final") {
      appendChat("assistant", renderMd(ev.answer));
      pushCtx("Ответ: " + ev.answer.replace(/\s+/g, " ").slice(0, 70) + "…");
    } else {
      const saved = lastThink;
      lastThink = null;
      renderEvent(ev);
      lastThink = saved;
    }
  }
}

/* ─────────────── запрос к агенту (SSE через fetch) ─────────────── */

async function ask(question) {
  streaming = true;
  $("btn-ask").disabled = true;
  $("status").classList.remove("hidden");
  appendChat("user", esc(question));
  pushCtx("Вопрос: " + question.slice(0, 70));
  clearStream();

  try {
    const resp = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (resp.status === 401) { showLogin("Войдите заново"); return; }
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
  } catch (e) {
    addStep("error", "⚠ обрыв соединения", esc(String(e)));
  } finally {
    streaming = false;
    $("btn-ask").disabled = false;
    $("status").classList.add("hidden");
  }
}

/* ─────────────── быстрый семантический поиск ─────────────── */

let searchTimer = null;

async function quickSearch(q) {
  const box = $("search-results");
  if (q.trim().length < 3) {
    box.className = "search-results muted";
    box.textContent = "введите фразу — вернётся топ по косинусу";
    return;
  }
  box.textContent = "ищем…";
  try {
    const r = await fetch("/api/search?q=" + encodeURIComponent(q));
    if (r.status === 401) return;
    const data = await r.json();
    if (data.error) { box.textContent = "ошибка: " + data.error; return; }
    if (!data.results.length) { box.textContent = "ничего не нашлось"; return; }
    box.className = "search-results";
    box.innerHTML = data.results.map((h) => `
      <div class="hit">
        <div><strong>${esc(h.name || "без названия")}</strong>
          <div class="sub">${esc(h.kind)}${h.cuisine ? " · " + esc(h.cuisine) : ""}${h.addr ? " · " + esc(h.addr) : ""}</div>
        </div>
        <div class="score">${(h.score * 100).toFixed(0)}%</div>
      </div>`).join("");
  } catch (e) {
    box.textContent = "ошибка поиска";
  }
}

/* ─────────────── события страницы ─────────────── */

$("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const r = await fetch("/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ login: $("login-name").value, password: $("login-pass").value }),
  });
  if (!r.ok) { showLogin("Неверный логин или пароль"); return; }
  const data = await r.json();
  showApp(data.login);
  const h = await fetch("/api/history");
  restore((await h.json()).events || []);
});

$("btn-logout").addEventListener("click", async () => {
  await fetch("/logout", { method: "POST" });
  location.reload();
});

$("btn-reset").addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  const h = await fetch("/api/history");
  restore((await h.json()).events || []);
});

$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  if (streaming) return;
  const q = $("question").value.trim();
  if (!q) return;
  $("question").value = "";
  ask(q);
});

$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("ask-form").requestSubmit();
  }
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

(async function boot() {
  const r = await fetch("/api/history");
  if (r.status === 401) { showLogin(""); return; }
  const data = await r.json();
  showApp(data.login || "");
  restore(data.events || []);
})();
