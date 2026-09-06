// Talks to Python only via `pywebview.api.*` (see api.py). No server, no
// fetch() calls -- pywebview bridges these calls in-process.

let typeOptions = [];   // [{key, label}], loaded once at boot
let aiAvailable = false;

let currentReviewItems = [];
let currentReviewIndex = -1;

let currentScannedType = null;
let currentScannedItems = [];
let currentScannedIndex = -1;

let currentDiscardedItems = [];
let currentDiscardedIndex = -1;

function $(id) { return document.getElementById(id); }

function whenReady(fn) {
  if (window.pywebview && window.pywebview.api) fn();
  else window.addEventListener("pywebviewready", fn);
}

function fillTypeSelect(select) {
  select.innerHTML = "";
  for (const t of typeOptions) {
    const opt = document.createElement("option");
    opt.value = t.key;
    opt.textContent = t.label;
    select.appendChild(opt);
  }
}

// ---------- tabs ----------

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    tab.classList.add("active");
    $("view-" + tab.dataset.view).classList.add("active");
    if (tab.dataset.view === "review") { loadReviewQueue(); pollAiStatus(); }
    if (tab.dataset.view === "run") { loadMonthList(); refreshStatus(); }
    if (tab.dataset.view === "scanned") loadScannedTypeCounts();
    if (tab.dataset.view === "discarded") loadDiscardedQueue();
    if (tab.dataset.view === "anomalies") loadAnomalyGroups();
  });
});

// ---------- Run view ----------

let selectedMonth = null;  // {path, name} or null -- see api.py's select_month_folder docstring for why no default

$("refresh-months").addEventListener("click", loadMonthList);

async function loadMonthList() {
  const months = await pywebview.api.list_month_folders();
  const list = $("month-list");
  list.innerHTML = "";
  for (const m of months) {
    const el = document.createElement("div");
    el.className = "review-list-item" + (selectedMonth && selectedMonth.path === m.path ? " active" : "");
    const doneNote = m.done > 0 ? `, ${m.done} done` : "";
    el.textContent = `${m.name} (${m.pending} pending${doneNote})`;
    el.addEventListener("click", () => selectMonth(m));
    list.appendChild(el);
  }
  if (months.length === 0) {
    list.innerHTML = '<p class="muted">No month folders found under incoming/peshawer/.</p>';
  }
}

async function selectMonth(m) {
  const result = await pywebview.api.select_month_folder(m.path);
  if (result.error) {
    $("client-status").textContent = "Error: " + result.error;
    return;
  }
  selectedMonth = m;
  await loadMonthList();  // re-render so the picked one shows as active
  await loadInputInfo();
  await refreshStatus();
}

async function loadInputInfo() {
  if (!selectedMonth) {
    $("input-path").textContent = "Pick a month above to begin.";
    $("run-button").disabled = true;
    return;
  }
  const files = await pywebview.api.list_input_files();
  const totalPages = files.reduce((sum, f) => sum + f.pages, 0);
  $("input-path").textContent = `${selectedMonth.name}: ${files.length} file(s), ${totalPages} page(s) still pending`;
  $("run-button").disabled = files.length === 0;
}

async function loadDefaultWorkers() {
  const n = await pywebview.api.default_workers();
  $("workers-input").value = n;
  $("workers-hint").textContent = `(this machine: ${n})`;
}

// Ground truth recomputed from disk on every call -- see pipeline_bridge's
// status() docstring for why this never trusts anything cached on screen,
// which is exactly what makes reopening the app (or just switching back to
// this tab) show real numbers instead of the tiles' zeroed HTML default.
//
// Skipped while a run is actively in progress: results.csv/discarded.csv
// (what status()'s discarded/error counts come from) are only written when
// a run finishes or stops, not per page, so refreshing from disk mid-run
// would show those two stuck at a stale number while onProgress's own live
// per-page counts are the accurate ones during that window. Once the run
// ends, onProgress's own "done"/"stopped" handlers call this again, so the
// dashboard still ends up correct either way.
let runInProgress = false;

async function refreshStatus() {
  if (runInProgress) return;
  const status = await pywebview.api.get_status();
  for (const k of ["scanned", "discarded", "review", "error"]) {
    $("count-" + k).textContent = status[k === "error" ? "errors" : k] ?? 0;
  }
  if (!selectedMonth) {
    $("client-status").textContent = "";
  } else if (status.total_files === 0) {
    $("client-status").textContent = "No files found.";
  } else if (status.fully_done) {
    $("client-status").textContent =
      `Fully processed -- ${status.scanned} filed, ${status.review} in review, ${status.discarded} discarded.`;
  } else if (status.started) {
    const remaining = status.total_files - status.done_files;
    $("client-status").textContent =
      `${status.done_files}/${status.total_files} file(s) done -- ${status.scanned} filed, ` +
      `${status.review} in review, ${status.discarded} discarded so far. ${remaining} file(s) not started yet.`;
  } else {
    $("client-status").textContent = `Not started -- ${status.total_files} file(s).`;
  }
}

// Keeps the Run tab live even without switching away and back -- only
// while it's the visible tab, and skipped mid-run for the same reason
// refreshStatus() itself skips (see above).
setInterval(() => {
  if (document.querySelector(".view.active")?.id === "view-run") refreshStatus();
}, 8000);

$("run-button").addEventListener("click", async () => {
  const restart = $("restart-checkbox").checked;
  const workers = parseInt($("workers-input").value, 10) || 1;

  $("run-button").disabled = true;
  $("run-status").textContent = "Starting...";
  $("log").innerHTML = "";
  ["scanned", "discarded", "review", "error"].forEach((k) => ($("count-" + k).textContent = "0"));
  $("progress-fill").style.width = "0%";
  $("progress-label").textContent = "Starting...";

  const result = await pywebview.api.start_run(restart, workers);
  if (result.error) {
    $("run-status").textContent = "Error: " + result.error;
    $("run-button").disabled = false;
  } else {
    runInProgress = true;
    $("pause-button").classList.remove("hidden");
    $("stop-button").classList.remove("hidden");
    $("pause-button").disabled = false;
    $("stop-button").disabled = false;
  }
});

// Pause and Stop both just interrupt the run cleanly (the pool is
// terminated, not drained) -- see pipeline_bridge.run_batch's docstring
// for why "resume" is simply clicking Run again rather than a separate
// mechanism: the per-file .done marker already makes that safe.
async function stopRun(label) {
  $("pause-button").disabled = true;
  $("stop-button").disabled = true;
  $("run-status").textContent = `${label}...`;
  await pywebview.api.stop_run();
}
$("pause-button").addEventListener("click", () => stopRun("Pausing"));
$("stop-button").addEventListener("click", () => stopRun("Stopping"));

// Called from Python (api.py._push) via window.evaluate_js -- must stay a
// global function for that to find it.
function onProgress(payload) {
  if (payload.phase === "start") {
    const workerNote = payload.workers > 1 ? ` across ${payload.workers} workers` : "";
    $("run-status").textContent = `Processing -- ${payload.total} page(s)${workerNote}`;
    if (payload.total === 0) {
      $("progress-label").textContent = "Nothing to do -- already fully processed.";
      $("run-button").disabled = false;
    }
    return;
  }

  if (payload.phase === "page") {
    const pct = payload.total ? Math.round((payload.index / payload.total) * 100) : 0;
    $("progress-fill").style.width = pct + "%";
    $("progress-label").textContent = `${payload.index} / ${payload.total} (${pct}%)`;
    for (const k of ["scanned", "discarded", "review", "error"]) {
      $("count-" + k).textContent = payload.counts[k] ?? 0;
    }
    const line = document.createElement("div");
    line.className = "status-" + (payload.status === "scanned" ? "renamed" : payload.status);
    const time = payload.time ? `[${payload.time.split(" ")[1] || payload.time}]  ` : "";
    const typeNote = payload.type && payload.type !== "other" ? ` (${payload.type})` : "";
    line.textContent = `${time}p${payload.page}  ${payload.file}  ->  ${payload.status}${typeNote}  ${payload.destination}`;
    const log = $("log");
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
    return;
  }

  if (payload.phase === "done") {
    $("progress-fill").style.width = "100%";
    $("progress-label").textContent = "Done.";
    $("run-status").textContent = `Finished -- ${payload.total} page(s) in ${Math.round(payload.elapsed)}s`;
    $("run-button").disabled = false;
    $("pause-button").classList.add("hidden");
    $("stop-button").classList.add("hidden");
    runInProgress = false;
    refreshStatus();
    loadMonthList();
    loadInputInfo();
    return;
  }

  if (payload.phase === "stopped") {
    $("progress-label").textContent = "Stopped.";
    $("run-status").textContent =
      `Stopped after ${payload.counts.scanned + payload.counts.discarded + payload.counts.review + payload.counts.error} ` +
      `of ${payload.total} page(s) -- click Run to pick up where this left off.`;
    $("run-button").disabled = false;
    $("pause-button").classList.add("hidden");
    $("stop-button").classList.add("hidden");
    runInProgress = false;
    refreshStatus();
    loadMonthList();
    loadInputInfo();
    return;
  }

  if (payload.phase === "error") {
    const line = document.createElement("div");
    line.className = "status-error";
    line.textContent = `[${new Date().toLocaleTimeString()}]  ERROR: ${payload.message}`;
    $("log").appendChild(line);
    $("run-status").textContent = "Stopped on an error -- see log.";
    $("run-button").disabled = false;
    $("pause-button").classList.add("hidden");
    $("stop-button").classList.add("hidden");
    runInProgress = false;
  }
}

// ---------- Review view ----------

$("refresh-review").addEventListener("click", loadReviewQueue);

async function loadReviewQueue() {
  currentReviewItems = await pywebview.api.list_review_items();
  currentReviewIndex = -1;
  renderReviewList();
  $("review-count").textContent = `${currentReviewItems.length} page(s) waiting`;
  $("review-item").classList.add("hidden");
  $("review-empty").classList.remove("hidden");
}

function renderReviewList() {
  const list = $("review-list");
  list.innerHTML = "";
  currentReviewItems.forEach((item, i) => {
    const el = document.createElement("div");
    el.className = "review-list-item" + (i === currentReviewIndex ? " active" : "");
    el.textContent = item.filename;
    if (item.suggested_number) el.classList.add("has-suggestion");
    el.addEventListener("click", () => openReviewItem(i));
    list.appendChild(el);
  });
}

async function openReviewItem(index) {
  currentReviewIndex = index;
  renderReviewList();
  const item = currentReviewItems[index];
  $("review-empty").classList.add("hidden");
  $("review-item").classList.remove("hidden");
  $("review-filename").textContent = item.filename;
  fillTypeSelect($("review-type"));
  // Default the book type to what the pipeline actually detected. Without
  // this the dropdown showed whichever option is first (Gas Cylinder), so
  // confirming a Delivery Challan without noticing filed it into the wrong
  // book -- silently, since every value is "valid".
  if (item.detected_type) $("review-type").value = item.detected_type;
  // Pre-fill the AI's read when it wasn't confirmed by a second reader.
  // The page still needs a human decision -- this just turns typing a
  // number into glancing at one and pressing Enter.
  if (item.suggested_number) {
    $("review-number").value = item.suggested_number;
    if (item.suggested_type) $("review-type").value = item.suggested_type;
    $("review-feedback").textContent = "AI suggests " + item.suggested_number + " -- check it, then Confirm.";
    $("review-feedback").className = "suggestion";
  } else {
    $("review-number").value = "";
    $("review-feedback").textContent = "";
    $("review-feedback").className = "";
  }
  $("review-image").src = "";
  $("review-image").src = await pywebview.api.get_page_image(item.path);
  $("review-number").focus();
  $("review-number").select();
}

// A reviewer meeting a page that isn't an ECR/DC document at all needs a
// real option -- otherwise they can only Skip it (leaving it in the queue
// forever) or invent a number for a page that has none. Reversible from the
// Discarded view. See review_bridge.discard_item.
$("review-discard").addEventListener("click", async () => {
  const item = currentReviewItems[currentReviewIndex];
  if (!item) return;
  $("review-discard").disabled = true;
  const result = await pywebview.api.discard_item(item.path);
  $("review-discard").disabled = false;
  if (result.error) {
    $("review-feedback").textContent = "Error: " + result.error;
    $("review-feedback").className = "err";
    return;
  }
  refreshStatus();
  currentReviewItems.splice(currentReviewIndex, 1);
  renderReviewList();
  $("review-count").textContent = `${currentReviewItems.length} page(s) waiting`;
  if (currentReviewItems.length === 0) {
    $("review-item").classList.add("hidden");
    $("review-empty").classList.remove("hidden");
    $("review-empty").textContent = "Review queue is empty.";
  } else {
    openReviewItem(Math.min(currentReviewIndex, currentReviewItems.length - 1));
  }
});

$("review-skip").addEventListener("click", () => {
  if (currentReviewIndex < currentReviewItems.length - 1) openReviewItem(currentReviewIndex + 1);
});

async function rotate(imgEl, path, degrees, rotateLeftBtn, rotateRightBtn) {
  rotateLeftBtn.disabled = true;
  rotateRightBtn.disabled = true;
  const result = await pywebview.api.rotate_page(path, degrees);
  rotateLeftBtn.disabled = false;
  rotateRightBtn.disabled = false;
  if (result.error) return result;
  imgEl.src = result.image;
  return result;
}

$("rotate-left").addEventListener("click", () => {
  const item = currentReviewItems[currentReviewIndex];
  if (item) rotate($("review-image"), item.path, 270, $("rotate-left"), $("rotate-right"));
});
$("rotate-right").addEventListener("click", () => {
  const item = currentReviewItems[currentReviewIndex];
  if (item) rotate($("review-image"), item.path, 90, $("rotate-left"), $("rotate-right"));
});

$("review-submit").addEventListener("click", submitReview);
$("review-number").addEventListener("keydown", (e) => { if (e.key === "Enter") submitReview(); });
$("scanned-number").addEventListener("keydown", (e) => { if (e.key === "Enter") $("scanned-save").click(); });
$("discarded-number").addEventListener("keydown", (e) => { if (e.key === "Enter") $("discarded-restore").click(); });

async function submitReview() {
  const item = currentReviewItems[currentReviewIndex];
  if (!item) return;
  const number = $("review-number").value.trim();
  if (!/^\d+$/.test(number)) {
    $("review-feedback").textContent = "Enter digits only.";
    $("review-feedback").className = "err";
    return;
  }
  const typeKey = $("review-type").value;
  $("review-submit").disabled = true;
  const result = await pywebview.api.submit_correction(item.path, number, typeKey);
  $("review-submit").disabled = false;
  if (result.error) {
    $("review-feedback").textContent = "Error: " + result.error;
    $("review-feedback").className = "err";
    return;
  }
  $("review-feedback").textContent = "Filed as " + result.destination;
  $("review-feedback").className = "ok";
  refreshStatus();
  currentReviewItems.splice(currentReviewIndex, 1);
  renderReviewList();
  $("review-count").textContent = `${currentReviewItems.length} page(s) waiting`;
  if (currentReviewItems.length === 0) {
    $("review-item").classList.add("hidden");
    $("review-empty").classList.remove("hidden");
    $("review-empty").textContent = "Review queue is empty.";
  } else {
    openReviewItem(Math.min(currentReviewIndex, currentReviewItems.length - 1));
  }
}

// AI auto-fill: kicked off by a button (not automatically) since it moves
// pages on its own without a per-page confirm click -- worth a deliberate
// action, not something that fires just from opening the tab.
$("run-ai-review").addEventListener("click", async () => {
  $("run-ai-review").disabled = true;
  await pywebview.api.run_ai_on_review_queue();
  pollAiStatus();
});

let _aiPollTimer = null;

async function pollAiStatus() {
  if (!aiAvailable) return;
  const s = await pywebview.api.ai_review_status();
  if (!s.available) return;
  $("ai-status").classList.remove("hidden");
  const usageNote = ` (${s.used}/${s.limit} used today -- Google's free tier allows ${s.official_limit}/day)`;
  const capReached = s.used >= s.limit;
  $("run-ai-review").disabled = capReached;
  if (s.total === 0) {
    $("ai-status").textContent = capReached ? "Daily AI limit reached." + usageNote : "";
    return;
  }
  $("ai-status").textContent = `AI: ${s.done}/${s.total} checked, ${s.filed} filed to Scanned` + usageNote;
  if (s.done < s.total && !capReached) {
    $("run-ai-review").disabled = true;
    clearTimeout(_aiPollTimer);
    _aiPollTimer = setTimeout(async () => {
      pollAiStatus();
      // A page the AI filed disappears from the queue -- refresh the list
      // periodically while the pass runs so it doesn't sit there stale.
      loadReviewQueue();
    }, 2000);
  } else {
    $("run-ai-review").disabled = capReached;
    loadReviewQueue();
  }
}

// ---------- Scanned view ----------

$("refresh-scanned").addEventListener("click", loadScannedTypeCounts);

async function loadScannedTypeCounts() {
  const counts = await pywebview.api.scanned_type_counts();
  renderScannedTypeList(counts);
  if (currentScannedType) await loadScannedItems(currentScannedType);
}

function renderScannedTypeList(counts) {
  const list = $("scanned-type-list");
  list.innerHTML = "";
  for (const c of counts) {
    const el = document.createElement("div");
    el.className = "review-list-item" + (c.type === currentScannedType ? " active" : "");
    el.textContent = `${c.label} (${c.count})`;
    el.addEventListener("click", async () => {
      currentScannedType = c.type;
      renderScannedTypeList(counts);   // re-render just to update the active highlight
      await loadScannedItems(c.type);
    });
    list.appendChild(el);
  }
}

async function loadScannedItems(typeKey) {
  currentScannedItems = await pywebview.api.list_scanned_items(typeKey);
  currentScannedIndex = -1;
  renderScannedList();
  $("scanned-count").textContent = `${currentScannedItems.length} page(s)`;
  $("scanned-item").classList.add("hidden");
  $("scanned-empty").classList.remove("hidden");
}

function renderScannedList() {
  const list = $("scanned-list");
  list.innerHTML = "";
  currentScannedItems.forEach((item, i) => {
    const el = document.createElement("div");
    el.className = "review-list-item" + (i === currentScannedIndex ? " active" : "");
    el.textContent = item.filename;
    el.addEventListener("click", () => openScannedItem(i));
    list.appendChild(el);
  });
}

async function openScannedItem(index) {
  currentScannedIndex = index;
  renderScannedList();
  const item = currentScannedItems[index];
  $("scanned-empty").classList.add("hidden");
  $("scanned-item").classList.remove("hidden");
  $("scanned-filename").textContent = `${item.type_label} / ${item.filename}`;
  $("scanned-number").value = item.filename.replace(/\.pdf$/i, "");
  fillTypeSelect($("scanned-type"));
  $("scanned-type").value = item.type;
  $("scanned-feedback").textContent = "";
  $("scanned-image").src = "";
  $("scanned-image").src = await pywebview.api.get_page_image(item.path);
  $("scanned-number").focus();
  $("scanned-number").select();
}

// Same escape hatch on the Scanned side: a page filed under a number that
// isn't actually an ECR/DC document should be removable, not just renamed.
$("scanned-discard").addEventListener("click", async () => {
  const item = currentScannedItems[currentScannedIndex];
  if (!item) return;
  $("scanned-discard").disabled = true;
  const result = await pywebview.api.discard_item(item.path);
  $("scanned-discard").disabled = false;
  if (result.error) {
    $("scanned-feedback").textContent = "Error: " + result.error;
    $("scanned-feedback").className = "err";
    return;
  }
  currentScannedItems.splice(currentScannedIndex, 1);
  loadScannedTypeCounts();
  renderScannedList();
  if (currentScannedItems.length === 0) {
    $("scanned-item").classList.add("hidden");
    $("scanned-empty").classList.remove("hidden");
  } else {
    openScannedItem(Math.min(currentScannedIndex, currentScannedItems.length - 1));
  }
});

$("scanned-skip").addEventListener("click", () => {
  if (currentScannedIndex < currentScannedItems.length - 1) openScannedItem(currentScannedIndex + 1);
});

$("scanned-rotate-left").addEventListener("click", () => {
  const item = currentScannedItems[currentScannedIndex];
  if (item) rotate($("scanned-image"), item.path, 270, $("scanned-rotate-left"), $("scanned-rotate-right"));
});
$("scanned-rotate-right").addEventListener("click", () => {
  const item = currentScannedItems[currentScannedIndex];
  if (item) rotate($("scanned-image"), item.path, 90, $("scanned-rotate-left"), $("scanned-rotate-right"));
});

$("scanned-save").addEventListener("click", async () => {
  const item = currentScannedItems[currentScannedIndex];
  if (!item) return;
  const number = $("scanned-number").value.trim();
  if (!/^\d+$/.test(number)) {
    $("scanned-feedback").textContent = "Enter digits only.";
    $("scanned-feedback").className = "err";
    return;
  }
  const typeKey = $("scanned-type").value;
  $("scanned-save").disabled = true;
  const result = await pywebview.api.rename_scanned_item(item.path, number, typeKey);
  $("scanned-save").disabled = false;
  if (result.error) {
    $("scanned-feedback").textContent = "Error: " + result.error;
    $("scanned-feedback").className = "err";
    return;
  }
  $("scanned-feedback").textContent = "Saved as " + result.destination;
  $("scanned-feedback").className = "ok";

  // Advance to the next page instead of reloading the list, which threw the
  // selection away and dropped the user back on "pick a page" after every
  // single correction -- unworkable when walking a whole book in sequence.
  // The corrected page has been renamed (and possibly moved to another
  // book's folder), so it is dropped from this in-memory list; it reappears
  // in its right place on the next Refresh.
  // Refresh the per-book counts ONLY. loadScannedTypeCounts() also calls
  // loadScannedItems(), which rebuilds the list and clears the selection --
  // un-awaited, it resolves a moment later and silently undoes the advance
  // below, dropping the user back on "Pick a book type".
  pywebview.api.scanned_type_counts().then(renderScannedTypeList);
  currentScannedItems.splice(currentScannedIndex, 1);
  renderScannedList();
  $("scanned-count").textContent = `${currentScannedItems.length} page(s)`;
  if (currentScannedItems.length === 0) {
    $("scanned-item").classList.add("hidden");
    $("scanned-empty").classList.remove("hidden");
    $("scanned-empty").textContent = "No more pages in this book type.";
  } else {
    openScannedItem(Math.min(currentScannedIndex, currentScannedItems.length - 1));
  }
});

// ---------- Discarded view ----------

$("refresh-discarded").addEventListener("click", loadDiscardedQueue);

async function loadDiscardedQueue() {
  currentDiscardedItems = await pywebview.api.list_discarded_items();
  currentDiscardedIndex = -1;
  renderDiscardedList();
  $("discarded-count").textContent = `${currentDiscardedItems.length} page(s)`;
  $("discarded-item").classList.add("hidden");
  $("discarded-empty").classList.remove("hidden");
}

function renderDiscardedList() {
  const list = $("discarded-list");
  list.innerHTML = "";
  currentDiscardedItems.forEach((item, i) => {
    const el = document.createElement("div");
    el.className = "review-list-item" + (i === currentDiscardedIndex ? " active" : "");
    el.textContent = item.filename + (item.reason ? `  [${item.reason}]` : "");
    el.addEventListener("click", () => openDiscardedItem(i));
    list.appendChild(el);
  });
}

async function openDiscardedItem(index) {
  currentDiscardedIndex = index;
  renderDiscardedList();
  const item = currentDiscardedItems[index];
  $("discarded-empty").classList.add("hidden");
  $("discarded-item").classList.remove("hidden");
  $("discarded-filename").textContent = item.filename;
  $("discarded-reason").textContent = item.reason ? `Discarded as: ${item.reason}` : "";
  $("discarded-number").value = "";
  fillTypeSelect($("discarded-type"));
  $("discarded-feedback").textContent = "";
  $("discarded-image").src = "";
  $("discarded-image").src = await pywebview.api.get_page_image(item.path);
}

$("discarded-skip").addEventListener("click", () => {
  if (currentDiscardedIndex < currentDiscardedItems.length - 1) openDiscardedItem(currentDiscardedIndex + 1);
});

$("discarded-rotate-left").addEventListener("click", () => {
  const item = currentDiscardedItems[currentDiscardedIndex];
  if (item) rotate($("discarded-image"), item.path, 270, $("discarded-rotate-left"), $("discarded-rotate-right"));
});
$("discarded-rotate-right").addEventListener("click", () => {
  const item = currentDiscardedItems[currentDiscardedIndex];
  if (item) rotate($("discarded-image"), item.path, 90, $("discarded-rotate-left"), $("discarded-rotate-right"));
});

$("discarded-restore").addEventListener("click", async () => {
  const item = currentDiscardedItems[currentDiscardedIndex];
  if (!item) return;
  const number = $("discarded-number").value.trim();
  if (!/^\d+$/.test(number)) {
    $("discarded-feedback").textContent = "Enter digits only.";
    $("discarded-feedback").className = "err";
    return;
  }
  const typeKey = $("discarded-type").value;
  $("discarded-restore").disabled = true;
  const result = await pywebview.api.restore_discarded_item(item.path, number, typeKey);
  $("discarded-restore").disabled = false;
  if (result.error) {
    $("discarded-feedback").textContent = "Error: " + result.error;
    $("discarded-feedback").className = "err";
    return;
  }
  $("discarded-feedback").textContent = "Restored to " + result.destination;
  $("discarded-feedback").className = "ok";
  currentDiscardedItems.splice(currentDiscardedIndex, 1);
  renderDiscardedList();
  $("discarded-count").textContent = `${currentDiscardedItems.length} page(s)`;
  if (currentDiscardedItems.length === 0) {
    $("discarded-item").classList.add("hidden");
    $("discarded-empty").classList.remove("hidden");
    $("discarded-empty").textContent = "Discarded pile is empty.";
  } else {
    openDiscardedItem(Math.min(currentDiscardedIndex, currentDiscardedItems.length - 1));
  }
});

// ---------- keyboard navigation ----------

// Down/Up move to the next/previous page in whichever of Review/Scanned/
// Discarded is currently the active tab -- lets a human flip through a
// queue fast without reaching for the mouse each time. Skipped when a
// <select> has focus so the type dropdown's own native up/down cycling
// still works normally; a focused text input (the number field) has no
// native meaning for these keys, so it's safe to hijack there too.
document.addEventListener("keydown", (e) => {
  if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
  if (document.activeElement && document.activeElement.tagName === "SELECT") return;

  const activeView = document.querySelector(".view.active");
  if (!activeView) return;
  const delta = e.key === "ArrowDown" ? 1 : -1;

  if (activeView.id === "view-review") {
    const next = currentReviewIndex + delta;
    if (next >= 0 && next < currentReviewItems.length) {
      e.preventDefault();
      openReviewItem(next);
    }
  } else if (activeView.id === "view-scanned") {
    const next = currentScannedIndex + delta;
    if (next >= 0 && next < currentScannedItems.length) {
      e.preventDefault();
      openScannedItem(next);
    }
  } else if (activeView.id === "view-discarded") {
    const next = currentDiscardedIndex + delta;
    if (next >= 0 && next < currentDiscardedItems.length) {
      e.preventDefault();
      openDiscardedItem(next);
    }
  }
});

// ---------- boot ----------

whenReady(async () => {
  // Data-entry build: opens straight into Review (no Run tab here -- see
  // index.html's comment on #view-run for why its markup still loads).
  typeOptions = await pywebview.api.type_options();
  loadReviewQueue();
  pollAiStatus();
  aiAvailable = await pywebview.api.ai_available();
  if (aiAvailable) $("run-ai-review").classList.remove("hidden");
});

// ---------- Duplicates & outliers ----------
//
// A pre-printed book number should be near-unique within a year and inside
// the book's own range. A value filed many times, or far outside that range,
// is almost always the wrong FIELD being read rather than a real document --
// "2232" and "7940" turned out to be VEHICLE numbers, filed 38 and 17 times.
// This view exists to find and fix those in bulk.

let currentAnomalyGroups = [];
let currentAnomalyGroupIndex = -1;
let currentAnomalyItems = [];
let currentAnomalyIndex = -1;

$("refresh-anomalies").addEventListener("click", loadAnomalyGroups);

async function loadAnomalyGroups() {
  currentAnomalyGroups = await pywebview.api.list_anomaly_groups();
  const dups = currentAnomalyGroups.filter((g) => g.kind === "duplicate").length;
  const outs = currentAnomalyGroups.length - dups;
  $("anomaly-count").textContent = `${dups} repeated number(s), ${outs} outlier(s)`;
  renderAnomalyGroupList();
  $("anomaly-item-list").innerHTML = "";
  $("anomaly-item-count").textContent = "";
  $("anomaly-item").classList.add("hidden");
  $("anomaly-empty").classList.remove("hidden");
}

function renderAnomalyGroupList() {
  const list = $("anomaly-group-list");
  list.innerHTML = "";
  currentAnomalyGroups.forEach((g, i) => {
    const el = document.createElement("div");
    el.className = "review-list-item" + (i === currentAnomalyGroupIndex ? " active" : "");
    const badge = g.kind === "duplicate"
      ? `<span class="badge dup">x${g.count}</span>`
      : `<span class="badge out">outlier</span>`;
    el.innerHTML = `${badge}${g.type_label} &mdash; ${g.number}`;
    el.addEventListener("click", () => openAnomalyGroup(i));
    list.appendChild(el);
  });
}

async function openAnomalyGroup(index) {
  currentAnomalyGroupIndex = index;
  renderAnomalyGroupList();
  const g = currentAnomalyGroups[index];
  currentAnomalyItems = await pywebview.api.list_duplicate_items(g.type, g.number);
  const why = g.kind === "outlier"
    ? `outside this book's range (${g.core || "?"})`
    : `filed ${g.count} times`;
  $("anomaly-item-count").textContent = `${g.number}: ${currentAnomalyItems.length} page(s) -- ${why}`;
  renderAnomalyItemList();
  if (currentAnomalyItems.length) openAnomalyItem(0);
}

function renderAnomalyItemList() {
  const list = $("anomaly-item-list");
  list.innerHTML = "";
  currentAnomalyItems.forEach((item, i) => {
    const el = document.createElement("div");
    el.className = "review-list-item" + (i === currentAnomalyIndex ? " active" : "");
    el.textContent = item.filename;
    el.addEventListener("click", () => openAnomalyItem(i));
    list.appendChild(el);
  });
}

async function openAnomalyItem(index) {
  currentAnomalyIndex = index;
  renderAnomalyItemList();
  const item = currentAnomalyItems[index];
  const g = currentAnomalyGroups[currentAnomalyGroupIndex];
  $("anomaly-empty").classList.add("hidden");
  $("anomaly-item").classList.remove("hidden");
  $("anomaly-filename").textContent = `${item.type_label} / ${item.filename}`;
  $("anomaly-why").textContent = g && g.kind === "outlier"
    ? `Outlier: this book's numbers run ${g.core}.`
    : `This number is filed on ${g ? g.count : "?"} pages.`;
  $("anomaly-number").value = g ? g.number : "";
  fillTypeSelect($("anomaly-type"));
  $("anomaly-type").value = item.type;
  $("anomaly-feedback").textContent = "";
  $("anomaly-image").src = "";
  $("anomaly-image").src = await pywebview.api.get_page_image(item.path);
  $("anomaly-number").focus();
  $("anomaly-number").select();
}

function advanceAnomaly() {
  currentAnomalyItems.splice(currentAnomalyIndex, 1);
  renderAnomalyItemList();
  if (currentAnomalyItems.length === 0) {
    $("anomaly-item").classList.add("hidden");
    $("anomaly-empty").classList.remove("hidden");
    $("anomaly-empty").textContent = "Done with this number -- pick another.";
  } else {
    openAnomalyItem(Math.min(currentAnomalyIndex, currentAnomalyItems.length - 1));
  }
}

$("anomaly-save").addEventListener("click", async () => {
  const item = currentAnomalyItems[currentAnomalyIndex];
  if (!item) return;
  const number = $("anomaly-number").value.trim();
  if (!/^\d+$/.test(number)) {
    $("anomaly-feedback").textContent = "Enter digits only.";
    $("anomaly-feedback").className = "err";
    return;
  }
  $("anomaly-save").disabled = true;
  const result = await pywebview.api.rename_scanned_item(item.path, number, $("anomaly-type").value);
  $("anomaly-save").disabled = false;
  if (result.error) {
    $("anomaly-feedback").textContent = "Error: " + result.error;
    $("anomaly-feedback").className = "err";
    return;
  }
  advanceAnomaly();
});

$("anomaly-discard").addEventListener("click", async () => {
  const item = currentAnomalyItems[currentAnomalyIndex];
  if (!item) return;
  $("anomaly-discard").disabled = true;
  const result = await pywebview.api.discard_item(item.path);
  $("anomaly-discard").disabled = false;
  if (result.error) {
    $("anomaly-feedback").textContent = "Error: " + result.error;
    $("anomaly-feedback").className = "err";
    return;
  }
  advanceAnomaly();
});

$("anomaly-skip").addEventListener("click", () => {
  if (currentAnomalyIndex < currentAnomalyItems.length - 1) openAnomalyItem(currentAnomalyIndex + 1);
});

$("anomaly-number").addEventListener("keydown", (e) => { if (e.key === "Enter") $("anomaly-save").click(); });

$("anomaly-rotate-left").addEventListener("click", () =>
  rotate($("anomaly-image"), currentAnomalyItems[currentAnomalyIndex].path, -90,
         $("anomaly-rotate-left"), $("anomaly-rotate-right")));
$("anomaly-rotate-right").addEventListener("click", () =>
  rotate($("anomaly-image"), currentAnomalyItems[currentAnomalyIndex].path, 90,
         $("anomaly-rotate-left"), $("anomaly-rotate-right")));
