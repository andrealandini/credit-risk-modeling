/* ── Utilities ──────────────────────────────────────────────────── */

function toast(msg, type = "success", duration = 3500) {
  const stack = document.getElementById("toast-stack");
  if (!stack) return;
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  stack.appendChild(el);
  setTimeout(() => el.remove(), duration);
}

function showSpinner(id, show) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle("hidden", !show);
}

function renderPlotly(divId, figJson) {
  if (!figJson || !window.Plotly) return;
  const layout = {
    ...figJson.layout,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(15,23,42,0)",
    font: { color: "#cbd5e1", family: "Inter, system-ui, sans-serif", size: 12 },
    xaxis: { ...(figJson.layout?.xaxis || {}), gridcolor: "#1e293b", linecolor: "#334155" },
    yaxis: { ...(figJson.layout?.yaxis || {}), gridcolor: "#1e293b", linecolor: "#334155" },
  };
  Plotly.react(divId, figJson.data, layout, { responsive: true, displayModeBar: false });
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

function setLog(terminalId, lines) {
  const el = document.getElementById(terminalId);
  if (el) { el.textContent = lines.join("\n"); el.scrollTop = el.scrollHeight; }
}

/* ── Param form renderer ─────────────────────────────────────────── */

function buildParamForm(schema, containerId, onChangeCallback) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = "";

  (schema || []).forEach(p => {
    const row = document.createElement("div");
    row.className = "param-row";

    if (p.type === "range") {
      const label = document.createElement("label");
      label.className = "param-label";
      label.innerHTML = `${p.label} <span id="${containerId}_${p.name}_val">${p.default}</span>`;
      const input = document.createElement("input");
      input.type = "range";
      input.className = "param-range";
      input.min = p.min; input.max = p.max; input.step = p.step;
      input.value = p.default;
      input.dataset.name = p.name;
      input.id = `${containerId}_${p.name}`;
      input.addEventListener("input", () => {
        document.getElementById(`${containerId}_${p.name}_val`).textContent = (+input.value).toFixed(
          p.step < 0.01 ? 3 : p.step < 0.1 ? 2 : 1
        );
        onChangeCallback && onChangeCallback();
      });
      row.appendChild(label);
      row.appendChild(input);

    } else if (p.type === "number") {
      const label = document.createElement("label");
      label.className = "param-label";
      label.textContent = p.label;
      const input = document.createElement("input");
      input.type = "number";
      input.className = "param-number";
      input.min = p.min; input.max = p.max; input.step = p.step;
      input.value = p.default;
      input.dataset.name = p.name;
      input.id = `${containerId}_${p.name}`;
      input.addEventListener("change", () => onChangeCallback && onChangeCallback());
      row.appendChild(label);
      row.appendChild(input);

    } else if (p.type === "select") {
      const label = document.createElement("label");
      label.className = "param-label";
      label.textContent = p.label;
      const sel = document.createElement("select");
      sel.className = "param-select";
      sel.dataset.name = p.name;
      sel.id = `${containerId}_${p.name}`;
      (p.options || []).forEach(opt => {
        const o = document.createElement("option");
        o.value = opt.value;
        o.textContent = opt.label;
        if (opt.value === p.default) o.selected = true;
        sel.appendChild(o);
      });
      sel.addEventListener("change", () => onChangeCallback && onChangeCallback());
      row.appendChild(label);
      row.appendChild(sel);
    }

    container.appendChild(row);
  });
}

function collectParams(containerId, schema) {
  const params = {};
  (schema || []).forEach(p => {
    const el = document.getElementById(`${containerId}_${p.name}`);
    if (!el) return;
    if (p.type === "range" || p.type === "number") {
      params[p.name] = parseFloat(el.value);
    } else {
      params[p.name] = el.value;
    }
  });
  return params;
}

/* ── PD Models page ──────────────────────────────────────────────── */

let pdCurrentModel = "logistic_regression";
let pdSchemas = {};

function initPDPage(schemas) {
  pdSchemas = schemas;
  document.querySelectorAll(".pd-model-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      pdCurrentModel = btn.dataset.model;
      document.querySelectorAll(".pd-model-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      buildParamForm(pdSchemas[pdCurrentModel], "pd-param-form", null);
    });
  });
  buildParamForm(pdSchemas[pdCurrentModel], "pd-param-form", null);

  document.getElementById("pd-run-btn").addEventListener("click", runPDModel);
}

async function runPDModel() {
  const btn = document.getElementById("pd-run-btn");
  btn.disabled = true;
  showSpinner("pd-spinner", true);

  try {
    const params = collectParams("pd-param-form", pdSchemas[pdCurrentModel]);
    const data = await postJSON("/pd/api/compute", { model: pdCurrentModel, ...params });

    document.getElementById("pd-result-badge").textContent = data.pd_pct;
    setLog("pd-log", data.log || []);

    if (data.tornado_fig) renderPlotly("pd-tornado-chart", data.tornado_fig);
    if (data.merton_fig) {
      document.getElementById("merton-fig-box").classList.remove("hidden");
      renderPlotly("merton-asset-chart", data.merton_fig);
    } else {
      document.getElementById("merton-fig-box").classList.add("hidden");
    }
    toast("PD computed successfully");
  } catch (e) {
    toast(e.message, "error");
    setLog("pd-log", [`Error: ${e.message}`]);
  } finally {
    btn.disabled = false;
    showSpinner("pd-spinner", false);
  }
}

/* ── LGD Models page ─────────────────────────────────────────────── */

let lgdCurrentModel = "beta_regression";
let lgdSchemas = {};

function initLGDPage(schemas) {
  lgdSchemas = schemas;
  document.querySelectorAll(".lgd-model-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      lgdCurrentModel = btn.dataset.model;
      document.querySelectorAll(".lgd-model-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      buildParamForm(lgdSchemas[lgdCurrentModel], "lgd-param-form", null);
    });
  });
  buildParamForm(lgdSchemas[lgdCurrentModel], "lgd-param-form", null);
  document.getElementById("lgd-run-btn").addEventListener("click", runLGDModel);
}

async function runLGDModel() {
  const btn = document.getElementById("lgd-run-btn");
  btn.disabled = true;
  showSpinner("lgd-spinner", true);
  try {
    const params = collectParams("lgd-param-form", lgdSchemas[lgdCurrentModel]);
    const data = await postJSON("/lgd/api/compute", { model: lgdCurrentModel, ...params });

    document.getElementById("lgd-result-badge").textContent = data.lgd_pct;
    setLog("lgd-log", data.log || []);

    if (data.sensitivity && data.sensitivity.length) {
      renderSensitivityTable("lgd-sens-table", data.sensitivity);
    }
    toast("LGD computed successfully");
  } catch (e) {
    toast(e.message, "error");
    setLog("lgd-log", [`Error: ${e.message}`]);
  } finally {
    btn.disabled = false;
    showSpinner("lgd-spinner", false);
  }
}

function renderSensitivityTable(tableId, rows) {
  const el = document.getElementById(tableId);
  if (!el) return;
  el.innerHTML = `
    <thead><tr>
      <th>Parameter</th><th>Base LGD</th><th>+Shock</th><th>−Shock</th><th>Δ Up</th><th>Δ Down</th>
    </tr></thead>
    <tbody>${rows.map(r => `<tr>
      <td>${r.param}</td>
      <td>${(r.base * 100).toFixed(2)}%</td>
      <td>${(r.up * 100).toFixed(2)}%</td>
      <td>${(r.down * 100).toFixed(2)}%</td>
      <td class="${r.delta_up > 0 ? 'text-red' : 'text-green'}">${r.delta_up > 0 ? '+' : ''}${(r.delta_up * 100).toFixed(2)}pp</td>
      <td class="${r.delta_dn > 0 ? 'text-red' : 'text-green'}">${r.delta_dn > 0 ? '+' : ''}${(r.delta_dn * 100).toFixed(2)}pp</td>
    </tr>`).join("")}</tbody>
  `;
}

/* ── EAD Models page ─────────────────────────────────────────────── */

let eadCurrentModel = "ccf";
let eadSchemas = {};

function initEADPage(schemas) {
  eadSchemas = schemas;
  document.querySelectorAll(".ead-model-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      eadCurrentModel = btn.dataset.model;
      document.querySelectorAll(".ead-model-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      buildParamForm(eadSchemas[eadCurrentModel], "ead-param-form", null);
    });
  });
  buildParamForm(eadSchemas[eadCurrentModel], "ead-param-form", null);
  document.getElementById("ead-run-btn").addEventListener("click", runEADModel);
}

async function runEADModel() {
  const btn = document.getElementById("ead-run-btn");
  btn.disabled = true;
  showSpinner("ead-spinner", true);
  try {
    const params = collectParams("ead-param-form", eadSchemas[eadCurrentModel]);
    const data = await postJSON("/ead/api/compute", { model: eadCurrentModel, ...params });

    document.getElementById("ead-result-badge").textContent = data.ead_fmt;
    setLog("ead-log", data.log || []);
    toast("EAD computed successfully");
  } catch (e) {
    toast(e.message, "error");
    setLog("ead-log", [`Error: ${e.message}`]);
  } finally {
    btn.disabled = false;
    showSpinner("ead-spinner", false);
  }
}

/* ── Comparison page ─────────────────────────────────────────────── */

let compScenarios = [];
let compPDSchemas = {}, compLGDSchemas = {}, compEADSchemas = {};
let compBuiltin = {};
let compResults = [];

function initComparisonPage(pdSchemas, lgdSchemas, eadSchemas, builtin) {
  compPDSchemas = pdSchemas;
  compLGDSchemas = lgdSchemas;
  compEADSchemas = eadSchemas;
  compBuiltin = builtin;

  document.getElementById("run-comparison-btn").addEventListener("click", runComparison);
  document.getElementById("export-csv-btn").addEventListener("click", exportCSV);

  Object.entries(builtin).forEach(([key, macro]) => {
    const btn = document.createElement("button");
    btn.className = "btn btn-outline text-sm";
    btn.textContent = `+ ${key.charAt(0).toUpperCase() + key.slice(1)}`;
    btn.addEventListener("click", () => addScenarioFromBuiltin(key, macro));
    document.getElementById("builtin-btns").appendChild(btn);
  });

  addScenario("ECB Baseline");
  addScenario("Recession Stress");
}

let _scenarioCounter = 0;

function addScenario(name, macroDefaults) {
  const id = ++_scenarioCounter;
  const sc = {
    id,
    name: typeof name === "string" ? name : `Scenario ${id}`,
    pd_model: "logistic_regression",
    lgd_model: "beta_regression",
    ead_model: "ccf",
    macro: macroDefaults || compBuiltin.baseline || {},
  };
  compScenarios.push(sc);
  renderScenarioCard(sc);
}

function addScenarioFromBuiltin(key, macro) {
  const names = { baseline: "ECB Baseline", recession: "Recession Stress",
                  stagflation: "Stagflation", recovery: "Strong Recovery" };
  addScenario(names[key] || key, macro);
}

function removeScenario(id) {
  compScenarios = compScenarios.filter(s => s.id !== id);
  document.getElementById(`scenario-card-${id}`)?.remove();
}

function renderScenarioCard(sc) {
  const container = document.getElementById("scenarios-container");
  const card = document.createElement("div");
  card.className = "scenario-card";
  card.id = `scenario-card-${sc.id}`;

  const macroFields = [
    { key: "gdp_growth", label: "GDP Growth %", min: -10, max: 8, step: 0.1 },
    { key: "unemployment", label: "Unemployment %", min: 3, max: 20, step: 0.1 },
    { key: "inflation", label: "Inflation %", min: -1, max: 12, step: 0.1 },
    { key: "policy_rate", label: "Policy Rate %", min: 0, max: 6, step: 0.05 },
    { key: "credit_growth", label: "Credit Growth %", min: -10, max: 15, step: 0.5 },
    { key: "lending_standards", label: "Lending Stds", min: -40, max: 60, step: 1 },
  ];

  card.innerHTML = `
    <div class="scenario-header">
      <input class="scenario-name-input" type="text" value="${sc.name}" id="sc-name-${sc.id}" />
      <button class="btn btn-outline text-xs" onclick="removeScenario(${sc.id})">✕ Remove</button>
    </div>
    <div class="mb-1">
      <div class="card-title" style="margin-bottom:0.5rem">Model Stack</div>
      <div style="display:flex;flex-direction:column;gap:0.4rem">
        <div class="param-row">
          <label class="param-label">PD Model</label>
          <select class="param-select" id="sc-pd-${sc.id}" onchange="updateScenario(${sc.id})">
            ${Object.entries(compPDSchemas).map(([k]) =>
              `<option value="${k}" ${k===sc.pd_model?'selected':''}>${compPDSchemas[k] ? k : k}</option>`
            ).join("")}
          </select>
        </div>
        <div class="param-row">
          <label class="param-label">LGD Model</label>
          <select class="param-select" id="sc-lgd-${sc.id}" onchange="updateScenario(${sc.id})">
            ${Object.entries(compLGDSchemas).map(([k]) =>
              `<option value="${k}" ${k===sc.lgd_model?'selected':''}>${k}</option>`
            ).join("")}
          </select>
        </div>
        <div class="param-row">
          <label class="param-label">EAD Model</label>
          <select class="param-select" id="sc-ead-${sc.id}" onchange="updateScenario(${sc.id})">
            ${Object.entries(compEADSchemas).map(([k]) =>
              `<option value="${k}" ${k===sc.ead_model?'selected':''}>${k}</option>`
            ).join("")}
          </select>
        </div>
      </div>
    </div>
    <hr class="card-divider"/>
    <div class="card-title" style="margin-bottom:0.5rem">Macro Inputs</div>
    <div class="macro-grid" id="sc-macro-${sc.id}">
      ${macroFields.map(f => `
        <div class="param-row">
          <label class="param-label">${f.label} <span id="sc-${f.key}-val-${sc.id}">${(sc.macro[f.key]||0).toFixed(1)}</span></label>
          <input type="range" class="param-range" min="${f.min}" max="${f.max}" step="${f.step}"
                 value="${sc.macro[f.key] || 0}"
                 id="sc-${f.key}-${sc.id}"
                 oninput="document.getElementById('sc-${f.key}-val-${sc.id}').textContent=parseFloat(this.value).toFixed(1)">
        </div>
      `).join("")}
    </div>
    <div id="sc-result-${sc.id}" class="hidden mt-1"></div>
  `;
  container.appendChild(card);
}

function updateScenario(id) {
  const sc = compScenarios.find(s => s.id === id);
  if (!sc) return;
  sc.pd_model = document.getElementById(`sc-pd-${id}`)?.value;
  sc.lgd_model = document.getElementById(`sc-lgd-${id}`)?.value;
  sc.ead_model = document.getElementById(`sc-ead-${id}`)?.value;
}

function collectScenario(sc) {
  const macro = {};
  ["gdp_growth","unemployment","inflation","policy_rate","credit_growth","lending_standards"].forEach(k => {
    const el = document.getElementById(`sc-${k}-${sc.id}`);
    if (el) macro[k] = parseFloat(el.value);
  });

  const pdSch = compPDSchemas[sc.pd_model] || [];
  const lgdSch = compLGDSchemas[sc.lgd_model] || [];
  const eadSch = compEADSchemas[sc.ead_model] || [];

  const pd_params = {};
  pdSch.forEach(p => { pd_params[p.name] = p.default; });
  Object.assign(pd_params, macro);

  const lgd_params = {};
  lgdSch.forEach(p => { lgd_params[p.name] = p.default; });
  if (macro.gdp_growth !== undefined) lgd_params.gdp_growth = macro.gdp_growth;

  const ead_params = {};
  eadSch.forEach(p => { ead_params[p.name] = p.default; });
  if (macro.gdp_growth !== undefined) ead_params.gdp_growth = macro.gdp_growth;
  if (macro.unemployment !== undefined) ead_params.unemployment = macro.unemployment;

  return {
    name: document.getElementById(`sc-name-${sc.id}`)?.value || sc.name,
    pd_model: sc.pd_model,
    lgd_model: sc.lgd_model,
    ead_model: sc.ead_model,
    macro,
    pd_params,
    lgd_params,
    ead_params,
    mc_params: { n_obligors: 200, n_paths: 3000, horizon_q: 4, rho: 0.15, seed: 42 },
  };
}

async function runComparison() {
  if (compScenarios.length === 0) { toast("Add at least one scenario", "error"); return; }
  const btn = document.getElementById("run-comparison-btn");
  btn.disabled = true;
  showSpinner("comp-spinner", true);
  document.getElementById("comp-results-area").classList.add("hidden");

  try {
    const scenarios = compScenarios.map(collectScenario);
    const data = await postJSON("/comparison/api/run", { scenarios });
    compResults = data.results || [];
    renderComparisonResults(compResults, data.comparison_fig);
    toast(`Compared ${compResults.length} scenarios`);
  } catch (e) {
    toast(e.message, "error");
  } finally {
    btn.disabled = false;
    showSpinner("comp-spinner", false);
  }
}

function renderComparisonResults(results, compFig) {
  const area = document.getElementById("comp-results-area");
  area.classList.remove("hidden");

  // Summary table
  const tbody = document.getElementById("comp-table-body");
  tbody.innerHTML = results.map(r => r.error ? `
    <tr><td>${r.name}</td><td colspan="7" style="color:var(--red)">${r.error}</td></tr>
  ` : `
    <tr>
      <td><strong>${r.name}</strong><br><span class="text-muted text-xs">${r.pd_model} / ${r.lgd_model} / ${r.ead_model}</span></td>
      <td>${r.pd_pct}</td>
      <td>${r.lgd_pct}</td>
      <td>${r.ead_fmt}</td>
      <td style="color:var(--amber)">${r.el_fmt}</td>
      <td>${r.ul_fmt}</td>
      <td style="color:var(--red)">${r.var_99_fmt}</td>
    </tr>
  `).join("");

  if (compFig) renderPlotly("comp-bar-chart", compFig);

  // Individual loss distributions
  const lossContainer = document.getElementById("comp-loss-charts");
  lossContainer.innerHTML = "";
  results.filter(r => !r.error && r.loss_fig).forEach(r => {
    const box = document.createElement("div");
    box.className = "chart-box";
    const divId = `loss-chart-${r.name.replace(/\s+/g, "-")}`;
    box.innerHTML = `<div class="chart-title">${r.name}</div><div id="${divId}"></div>`;
    lossContainer.appendChild(box);
    setTimeout(() => renderPlotly(divId, r.loss_fig), 50);
  });
}

async function exportCSV() {
  if (!compResults.length) { toast("Run comparison first", "error"); return; }
  try {
    const res = await fetch("/comparison/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ results: compResults }),
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "credit_risk_scenarios.csv"; a.click();
    URL.revokeObjectURL(url);
    toast("CSV downloaded");
  } catch (e) {
    toast(e.message, "error");
  }
}

/* ── Dashboard ───────────────────────────────────────────────────── */

function initDashboard(macroFig, lossFig, pdTsFig, corrFig) {
  renderPlotly("dash-macro-chart", macroFig);
  renderPlotly("dash-loss-chart", lossFig);
  renderPlotly("dash-pdts-chart", pdTsFig);
  renderPlotly("dash-corr-chart", corrFig);
}
