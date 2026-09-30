// ── Landing Page ─────────────────────────────────────────────────────────────
function goHome(e) {
  if (e) e.preventDefault();
  returnToLanding();
  window.scrollTo({top: 0, behavior: 'smooth'});
}

function goAbout(e) {
  if (e) e.preventDefault();
  if (document.getElementById('landing-container').style.display === 'none') {
    returnToLanding();
  }
  setTimeout(() => {
    const el = document.querySelector('.landing-middle');
    if (el) {
      const top = el.getBoundingClientRect().top + window.scrollY - 80;
      window.scrollTo({top, behavior: 'smooth'});
    }
  }, 100);
}

function goDevelopers(e) {
  if (e) e.preventDefault();
  if (document.getElementById('landing-container').style.display === 'none') {
    returnToLanding();
  }
  setTimeout(() => {
    const el = document.querySelector('.landing-bottom');
    if (el) {
      const top = el.getBoundingClientRect().top + window.scrollY - 80;
      window.scrollTo({top, behavior: 'smooth'});
    }
  }, 100);
}

function goTool(e) {
  if (e) e.preventDefault();
  startApp();
}

// Side nav observer
document.addEventListener("DOMContentLoaded", () => {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        document.querySelectorAll('.side-dot').forEach(d => d.classList.remove('active'));
        if (entry.target.classList.contains('landing-top')) {
          document.querySelector('.side-dot[data-target="home"]')?.classList.add('active');
        } else if (entry.target.classList.contains('landing-middle')) {
          document.querySelector('.side-dot[data-target="about"]')?.classList.add('active');
        } else if (entry.target.classList.contains('landing-bottom')) {
          document.querySelector('.side-dot[data-target="devs"]')?.classList.add('active');
        }
      }
    });
  }, { threshold: 0.3 });
  
  document.querySelectorAll('.landing-top, .landing-middle, .landing-bottom').forEach(section => {
    observer.observe(section);
  });
});

function startApp() {
  document.getElementById('landing-container').style.display = 'none';
  document.getElementById('app-container').style.display = 'flex';
  document.getElementById('navTitle').style.display = 'block';
  document.getElementById('sideNav').style.display = 'none';
  // Fade in
  setTimeout(() => {
    document.getElementById('app-container').style.opacity = '1';
  }, 10);
  window.scrollTo({top: 0, behavior: 'smooth'});
}

function returnToLanding() {
  document.getElementById('app-container').style.opacity = '0';
  document.getElementById('navTitle').style.display = 'none';
  document.getElementById('sideNav').style.display = 'flex';
  setTimeout(() => {
    document.getElementById('app-container').style.display = 'none';
    document.getElementById('landing-container').style.display = 'flex';
  }, 500);
}

function copyEmail() {
  const email = 'hybridknapsack@gmail.com';
  navigator.clipboard.writeText(email).then(() => {
    const toast = document.getElementById('emailToast');
    toast.style.display = 'block';
    setTimeout(() => {
      toast.style.display = 'none';
    }, 3000);
  }).catch(err => {
    console.error('Failed to copy email: ', err);
  });
}

// ── Page Navigation ────────────────────────────────────────────────────────
function switchPage(pageId) {
  // Update Tabs
  document.querySelectorAll('.tab-link').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.page === pageId);
  });
  
  // Update Views
  document.querySelectorAll('.page-view').forEach(view => {
    view.style.display = 'none';
    view.classList.remove('active');
  });
  
  const activeView = document.getElementById('page-' + pageId);
  if (activeView) {
    activeView.style.display = 'block';
    // Small delay to allow display:block to apply before adding opacity class
    setTimeout(() => activeView.classList.add('active'), 10);
  }
  
  // Toggle ActionBar visibility
  const actionBar = document.querySelector('.actionbar');
  if (actionBar) {
    actionBar.style.display = pageId === 'execution' ? 'block' : 'none';
  }

  // Toggle footer text visibility
  const footerTxt = document.getElementById('footerTxt');
  if (footerTxt) {
    footerTxt.style.display = pageId === 'execution' ? 'block' : 'none';
  }

  // Each phase starts at the top rather than mid-scroll from the last one.
  window.scrollTo({ top: 0, behavior: 'smooth' });
  updatePhaseNav();
  if (pageId === 'review') loadReview();
}

/* Keeps the sequential "continue" buttons in step with the current state:
   the Setup phase cannot be left until projects have actually been chosen,
   and the reason is stated rather than the click silently failing. */
function updatePhaseNav() {
  const selCount = CHECKED.size;

  const toExec = document.getElementById('toExecutionBtn');
  const hint   = document.getElementById('setupHint');
  if (toExec) {
    toExec.disabled = selCount === 0;
  }
  if (hint) {
    hint.textContent = selCount === 0
      ? 'Select at least one project above to continue.'
      : `${selCount.toLocaleString()} project${selCount === 1 ? '' : 's'} selected.`;
    hint.classList.toggle('warn', selCount === 0);
  }

  const execHint = document.getElementById('execHint');
  if (execHint) {
    const hasResults = !!document.querySelector('#results .compare-grid');
    execHint.textContent = hasResults
      ? 'Results recorded. Continue to view the history and statistical report.'
      : 'Run the algorithms above to record results.';
    execHint.classList.toggle('warn', !hasResults);
  }
}

// ── State ──────────────────────────────────────────────────────────────────
let DS       = 'pasig';
let PROJECTS = [];   // full dataset for current DS
let FILTERED = [];   // after sector + search filter
let CHECKED  = new Set();
let BUDGET   = 5_000_000_000;
let SECTOR   = 'All';
let PAGE     = 0;
const PAGE_SIZE = 50;

let currentTab  = 2;
const _expandedNodes = new Set();
// DP is a correctness baseline (a third exact method that should reach the
// same optimum), not one of the two variations compared in SOP 3, so it
// starts disabled and is opted into deliberately.
let activeAlgos = {dp: false, bnb: true, ga: true};

function toggleAlgo(name, el) {
  // Prevent unchecking the last remaining algorithm
  const others = Object.keys(activeAlgos).filter(k => k !== name);
  const anyOtherActive = others.some(k => activeAlgos[k]);
  if (!el.checked && !anyOtherActive) {
    el.checked = true;
    return;
  }
  activeAlgos[name] = el.checked;
  document.getElementById('lbl-'+name).classList.toggle('checked', el.checked);
  updateActionBar();
}

// ── Init ───────────────────────────────────────────────────────────────────
async function init() {
  await loadDs('pasig');
  updateMeta();
  updateActionBar();
}

async function loadDs(ds) {
  DS = ds;
  const resp = await fetch(`/api/projects?ds=${ds}`);
  const data = await resp.json();
  PROJECTS = data.projects;
  CHECKED.clear();
  SECTOR = 'All';
  PAGE   = 0;
  document.getElementById('searchBox').value = '';
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.toggle('on', b.dataset.sector === 'All'));

  // Update switcher
  document.getElementById('btnPasig').className = 'ds-card' + (ds==='pasig' ? ' active' : '');
  document.getElementById('btnQC').className    = 'ds-card' + (ds==='qc'    ? ' active' : '');
  const bar = document.getElementById('infoBar');
  bar.className = 'ds-infobar ' + ds;
  const info = {
    pasig: '<b>Pasig City APP FY 2025 (General Fund)</b> — Source: City Government of Pasig, LGU Transparency Portal. Used as the <b>small dataset</b> to establish a standard of optimality and verify algorithm accuracy.',
    qc:    '<b>Quezon City APP FY 2025 (4th Quarter)</b> — Source: QC Bids and Awards Committee, LGU Transparency Portal. Used as the <b>large dataset</b> to stress-test scalability and efficiency of the hybrid algorithm.',
  };
  document.getElementById('infoText').innerHTML = info[ds];
  document.getElementById('footerTxt').textContent = ds==='pasig'
    ? 'Pasig City · Annual Procurement Plan FY 2025 (General Fund) | Knapsack DP · Branch & Bound · Genetic Algorithm'
    : 'Quezon City · Annual Procurement Plan FY 2025 (4th Quarter) | Knapsack DP · Branch & Bound · Genetic Algorithm';

  filterProjects();
  preselectTop();
  // Restore the empty state rather than blanking the panel. Blanking left
  // step 5 of the Execution tab visibly empty with no explanation of what
  // to do next.
  resetResultsPanel();
}

function resetResultsPanel() {
  document.getElementById('results').innerHTML = `
    <div class="card">
      <div class="empty-state">
        <p>No results yet.</p>
        <p class="es-sub">Choose your projects and algorithms, then run. Each algorithm's
        runtime, execution time and pruning rate will be compared here.</p>
      </div>
    </div>`;
}

function switchDs(ds) {
  if (ds === DS) return;
  loadDs(ds);
  const bp = document.getElementById('btnPasig');
  const bq = document.getElementById('btnQC');
  if (bp) bp.setAttribute('aria-pressed', ds === 'pasig');
  if (bq) bq.setAttribute('aria-pressed', ds === 'qc');
}

function updateMeta() {
  fetch('/api/meta').then(r=>r.json()).then(d=>{
    document.getElementById('pasigMeta').textContent =
      `${d.pasig.count.toLocaleString()} Projects`;
    document.getElementById('qcMeta').textContent =
      `${d.qc.count.toLocaleString()} Projects`;
  });
}

// ── Budget ─────────────────────────────────────────────────────────────────
const BUDGET_MAX = 50000000000;

function updateBudget(v, source) {
  let n;
  if (source === 'input') {
    // Strip everything except digits (allow the user to type commas freely)
    n = parseInt(String(v).replace(/[^\d]/g, ''), 10);
    if (isNaN(n)) n = 0;
  } else {
    n = parseInt(v, 10);
    if (isNaN(n)) n = 0;
  }
  // Clamp to [0, BUDGET_MAX]
  if (n < 0) n = 0;
  if (n > BUDGET_MAX) n = BUDGET_MAX;
  BUDGET = n;

  // Sync the slider (always reflects the clamped value)
  document.getElementById('budgetSlider').value = n;

  // Sync the text input. While the user is actively typing in it, don't
  // fight their cursor by reformatting mid-edit; only push the formatted
  // value when the change came from the slider.
  if (source !== 'input') {
    document.getElementById('budgetInput').value = n.toLocaleString();
  }
  updateActionBar();
}

function normalizeBudgetInput() {
  // On blur, snap the text field to the clean formatted, clamped value.
  document.getElementById('budgetInput').value = BUDGET.toLocaleString();
}

function fmt(n) {
  return '₱' + Math.round(n).toLocaleString();
}
function fmtB(n) {
  if (n >= 1e9) return (n/1e9).toFixed(2)+'B';
  if (n >= 1e6) return (n/1e6).toFixed(1)+'M';
  return Math.round(n).toLocaleString();
}

// ── Filter + render ────────────────────────────────────────────────────────
function setSector(btn) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
  SECTOR = btn.dataset.sector;
  PAGE = 0;
  filterProjects();
}

function filterProjects() {
  const q = document.getElementById('searchBox').value.toLowerCase();
  FILTERED = PROJECTS.filter(p => {
    if (SECTOR !== 'All' && p.sector !== SECTOR) return false;
    if (q && !p.name.toLowerCase().includes(q) && !p.pmo.toLowerCase().includes(q)) return false;
    return true;
  });
  PAGE = 0;
  renderTable();
}

function renderTable() {
  const start = PAGE * PAGE_SIZE;
  const end   = Math.min(start + PAGE_SIZE, FILTERED.length);
  const page_items = FILTERED.slice(start, end);

  const tbody = document.getElementById('projTbody');
  if (!FILTERED.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty">No projects match the filter.</td></tr>';
    renderPagination();
    updateSelCount();
    return;
  }

  tbody.innerHTML = page_items.map((p, li) => {
    const pi = p._idx;  // index in PROJECTS
    const chk = CHECKED.has(pi);
    return `<tr>
      <td><input type="checkbox" class="chk" ${chk?'checked':''} onchange="toggleCheck(${pi},this.checked)"></td>
      <td style="max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escHtml(p.name)}">${escHtml(p.name)}</td>
      <td><span class="s-badge s-${p.sector.replace(/ /g,'-')}">${p.sector}</span></td>
      <td style="font-size:11px;color:var(--tx-secondary);max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escHtml(p.pmo)}</td>
      <td class="cost-col">${fmtB(p.cost)}</td>
      <td class="benefit-col"><span class="b-pill" style="background:${bcolor(p.benefit)}">${p.benefit.toFixed(1)}</span></td>
    </tr>`;
  }).join('');

  // Update pool stats
  document.getElementById('totalCount').textContent  = PROJECTS.length.toLocaleString();
  document.getElementById('poolLabel').textContent   = DS==='pasig'
    ? 'Pasig City APP FY 2025' : 'Quezon City APP FY 2025';
  const tot = PROJECTS.reduce((s,p)=>s+p.cost,0);
  document.getElementById('totalBudgetLbl').textContent = fmtB(tot);

  renderPagination();
  updateSelCount();
  syncChkAll();
}

function renderPagination() {
  const total_pages = Math.ceil(FILTERED.length / PAGE_SIZE);
  const start = PAGE * PAGE_SIZE + 1;
  const end   = Math.min((PAGE+1)*PAGE_SIZE, FILTERED.length);
  document.getElementById('pageInfo').textContent = `Showing ${start.toLocaleString()}–${end.toLocaleString()} of ${FILTERED.length.toLocaleString()}`;

  const cont = document.getElementById('pageBtns');
  if (total_pages <= 1) { cont.innerHTML=''; return; }

  let btns = `<button class="page-btn" onclick="goPage(${PAGE-1})" ${PAGE===0?'disabled':''}>←</button>`;
  // Show at most 7 page buttons around current
  const lo = Math.max(0, PAGE-3), hi = Math.min(total_pages-1, PAGE+3);
  if (lo > 0) btns += `<button class="page-btn" onclick="goPage(0)">1</button>${lo>1?'<span style="color:var(--tx-muted);padding:0 4px">…</span>':''}`;
  for (let i=lo; i<=hi; i++) btns += `<button class="page-btn ${i===PAGE?'active':''}" onclick="goPage(${i})">${i+1}</button>`;
  if (hi < total_pages-1) btns += `${hi<total_pages-2?'<span style="color:var(--tx-muted);padding:0 4px">…</span>':''}<button class="page-btn" onclick="goPage(${total_pages-1})">${total_pages}</button>`;
  btns += `<button class="page-btn" onclick="goPage(${PAGE+1})" ${PAGE===total_pages-1?'disabled':''}>→</button>`;
  cont.innerHTML = btns;
}

function goPage(p) { PAGE = p; renderTable(); }

function toggleCheck(pi, checked) {
  if (checked) CHECKED.add(pi); else CHECKED.delete(pi);
  updateSelCount();
}

function toggleAll(checked) {
  // Select / deselect ALL filtered projects across every page
  FILTERED.forEach(p => {
    if (checked) CHECKED.add(p._idx); else CHECKED.delete(p._idx);
  });
  renderTable();
}

function syncChkAll() {
  const el = document.getElementById('chkAll');
  if (!el) return;
  const total    = FILTERED.length;
  const selected = FILTERED.filter(p => CHECKED.has(p._idx)).length;
  el.checked       = total > 0 && selected === total;
  el.indeterminate = selected > 0 && selected < total;
}

function updateSelCount() {
  const sel   = [...CHECKED];
  const total = sel.reduce((s,i) => s + PROJECTS[i].cost, 0);
  document.getElementById('selPill').textContent = `${sel.length.toLocaleString()} selected`;
  document.getElementById('selCost').textContent  = `${fmt(total)} total cost`;
  updateActionBar();
}

function preselectTop() {
  CHECKED.clear();
  // Select top-50 by benefit score
  [...PROJECTS]
    .map((p,i)=>({i, b:p.benefit, c:p.cost}))
    .sort((a,b)=>b.b-a.b||a.c-b.c)
    .slice(0,50)
    .forEach(x=>CHECKED.add(x.i));
  renderTable();
}

function bcolor(b) {
  if (b >= 9)  return '#16A34A';
  if (b >= 7)  return '#0284C7';
  if (b >= 5)  return '#B45309';
  return '#9CA3AF';
}
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}


// ── Run algorithms (streamed) ─────────────────────────────────────────────
// Progress arrives over Server-Sent Events instead of waiting for one big
// response, so every algorithm's trial counter, latest timing and running
// mean update live, and each algorithm's result renders the moment it
// finishes its own trials.
let LIVE = {};   // algo -> live tallies for the race panel

async function runAlgos() {
  const selected_indices = [...CHECKED];
  if (!selected_indices.length) {
    showMsg('warn', 'Select at least one project in step 3 before running.');
    document.getElementById('stepProjects').scrollIntoView({block:'center'});
    return;
  }
  const algos = Object.keys(activeAlgos).filter(k => activeAlgos[k]);
  if (!algos.length) {
    showMsg('warn', 'Select at least one algorithm in step 4 before running.');
    document.getElementById('setup').scrollIntoView({block:'center'});
    return;
  }
  clearMsg();

  const trials = getTrials();
  const btn = document.getElementById('runBtn');
  btn.innerHTML = '<span class="spinner"></span>Running…';
  btn.disabled = true;

  // The old single bar is replaced by the per-algorithm race panel.
  document.getElementById('progressWrap').style.display = 'none';

  LIVE = {};
  const finished = {};
  let doneData = null;

  try {
    const resp = await fetch('/api/run/stream', {
      method:  'POST',
      headers: {'Content-Type':'application/json'},
      body:    JSON.stringify({
        ds: DS, budget: BUDGET, selected: selected_indices,
        algos, trials, pop_size: 60, gens: 100,  // mutation defaults to 1/n server-side
      }),
    });

    if (!resp.ok) {
      let msg = 'Run failed.';
      try { msg = (await resp.json()).error || msg; } catch (e) {}
      throw new Error(msg);
    }

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      const frames = buffer.split('\n\n');
      buffer = frames.pop();

      for (const frame of frames) {
        const line = frame.split('\n').find(l => l.startsWith('data: '));
        if (!line) continue;
        const ev = JSON.parse(line.slice(6));

        if (ev.type === 'start') {
          ev.algos.forEach(a => {
            LIVE[a] = { label: ev.labels[a], trial: 0, trials: ev.trials,
                        runtime: null, mean_runtime: null, mean_exec: null,
                        mean_pruning: null, done: false };
          });
          renderRacePanel();

        } else if (ev.type === 'progress') {
          const L = LIVE[ev.algo];
          if (L) {
            L.trial        = ev.trial;
            L.trials       = ev.trials;
            L.runtime      = ev.runtime_ms;
            L.mean_runtime = ev.mean_runtime;
            L.mean_exec    = ev.mean_exec;
            L.mean_pruning = ev.mean_pruning;
            L.benefit      = ev.total_benefit;
          }
          renderRacePanel();

        } else if (ev.type === 'result') {
          if (LIVE[ev.algo]) LIVE[ev.algo].done = true;
          finished[ev.algo] = ev.result;
          renderRacePanel();

        } else if (ev.type === 'done') {
          doneData = ev;
        }
      }
    }

    if (doneData) {
      currentTab = doneData.results.length - 1;
      renderResults(doneData);
      renderHistory(doneData.history);
      captureAllocation(doneData, selected_indices);
      loadStats();
      updatePhaseNav();
    }

  } catch(err) {
    document.getElementById('results').innerHTML =
      `<div class="banner warn"><span>⚠</span> Error: ${escHtml(err.message)}</div>`;
  } finally {
    btn.disabled = false;
    updateActionBar();
  }
}

/* Live race panel: one lane per algorithm, updated on every trial. */
function renderRacePanel() {
  const algos = Object.keys(LIVE);
  if (!algos.length) return;

  const allDone = algos.every(a => LIVE[a].done);
  const lanes = algos.map(a => {
    const L   = LIVE[a];
    const pct = L.trials ? Math.round((L.trial / L.trials) * 100) : 0;
    return `
      <div class="lane ${L.done ? 'done' : ''}">
        <div class="lane-hdr">
          <span class="lane-name">${escHtml(L.label)}</span>
          <span class="lane-count">${L.done ? 'finished' : `run ${L.trial} / ${L.trials}`}</span>
        </div>
        <div class="lane-bar"><div class="lane-fill" style="width:${pct}%"></div></div>
        <div class="lane-stats">
          <div><span class="ls-l">last runtime</span><span class="ls-v">${L.runtime==null?'—':L.runtime.toFixed(2)+' ms'}</span></div>
          <div><span class="ls-l">mean runtime</span><span class="ls-v">${L.mean_runtime==null?'—':L.mean_runtime.toFixed(2)+' ms'}</span></div>
          <div><span class="ls-l">mean exec.</span><span class="ls-v">${L.mean_exec==null?'—':L.mean_exec.toFixed(2)+' ms'}</span></div>
          <div><span class="ls-l">mean pruning</span><span class="ls-v">${L.mean_pruning==null?'—':L.mean_pruning.toFixed(1)+'%'}</span></div>
        </div>
      </div>`;
  }).join('');

  document.getElementById('results').innerHTML = `
    <div class="card race-card">
      <div class="card-hdr">
        <div class="card-title">${allDone ? 'Run complete' : 'Running…'}</div>
        <div class="card-meta">${allDone ? 'Compiling comparison…' : 'Trials are interleaved, so each timing is measured without contention'}</div>
      </div>
      <div class="race-lanes">${lanes}</div>
    </div>`;
}

function renderResults(data) {
  const results    = data.results;
  const sel_items  = data.selected_items;
  const budget_used = data.budget;

  const benefits = results.map(r=>r.total_benefit);
  const maxBen   = Math.max(...benefits);
  const minBen   = Math.min(...benefits);
  const allMatch = benefits.every(b=>b===maxBen);
  const gapPct   = maxBen>0 ? (maxBen-minBen)/maxBen*100 : 0;

  // A small gap (< 1%) is almost always the standard DP's discretization
  // rounding (it works on a bucketed cost axis), not a GA convergence
  // problem - the exact methods (B&B, B&B+GA) always agree with each other.
  const banner = results.length === 1
    ? `<div class="banner ok"><span>✓</span> ${escHtml(results[0].label)} reached a benefit score of ${maxBen.toFixed(2)}.</div>`
    : allMatch
    ? `<div class="banner ok"><span>✓</span> All algorithms reached the same optimal benefit score (${maxBen.toFixed(2)}) — results are consistent.</div>`
    : gapPct < 1.0
    ? `<div class="banner ok"><span>✓</span> Branch-and-Bound methods agree on the optimum (${maxBen.toFixed(2)}); standard DP is within ${gapPct.toFixed(3)}% due to its discretized cost axis — expected behaviour.</div>`
    : `<div class="banner warn"><span>⚠</span> Benefit scores differ by ${gapPct.toFixed(2)}%. For very large selections, standard DP uses a coarser cost grid; B&B and B&B+GA remain exact.</div>`;

  const barColors = ['#4F6EF7','#0284C7','#7C3AED'];
  const ccards = results.map((r,ri)=>{
    const isBest = r.total_benefit === maxBen;
    const tc = r.selected.reduce((s,i)=>s+sel_items[i].cost,0);
    const pct = maxBen>0 ? Math.round((r.total_benefit/maxBen)*100) : 0;
    const bc = isBest ? '#16A34A' : barColors[ri % barColors.length];
    const hasNodes = r.pruning_rate != null;
    const ntOpen = hasNodes && _expandedNodes.has(ri);
    return `<div class="ccard ${isBest?'best':''}">
      <div class="ccard-eye">${isBest?'<div class="best-tag">✓ Optimal</div>':''}</div>
      <div class="ccard-title">${escHtml(r.label)}</div>
      <div class="ccard-big">${r.total_benefit.toFixed(2)}</div>
      <div class="ccard-sub">total benefit score</div>
      <div class="bar-t"><div class="bar-f" style="width:${pct}%;background:${bc}"></div></div>
      <div class="ccard-div"></div>
      <div class="ccard-row"><span class="lbl">Budget used</span><span class="val ${isBest?'hl-green':'hl'}">${fmtB(tc)}</span></div>
      <div class="ccard-row"><span class="lbl">Projects</span><span class="val">${r.selected.length}</span></div>
      <div class="ccard-row"><span class="lbl">Runtime</span><span class="val hl">${r.runtime_ms!=null ? r.runtime_ms.toFixed(1)+' ms' : 'N/A'}</span></div>
      <div class="ccard-row"><span class="lbl">Execution time</span><span class="val">${r.exec_ms!=null ? r.exec_ms.toFixed(1)+' ms' : 'N/A'}</span></div>
      <div class="ccard-row ${hasNodes?'node-row':''} ${ntOpen?'open':''}" id="nt-btn-${ri}" ${hasNodes?`onclick="toggleNodes(${ri})"`:''}><span class="lbl">Pruning rate${hasNodes?'<span class="nt-caret">▸</span>':''}</span><span class="val ${r.pruning_rate!=null?'hl-green':''}">${r.pruning_rate!=null ? r.pruning_rate.toFixed(2)+'%' : 'N/A'}</span></div>
      <div class="node-detail" id="nt-${ri}" style="display:${ntOpen?'block':'none'}">
        <div class="ccard-row"><span class="lbl">Nodes explored</span><span class="val">${r.nodes_generated!=null ? r.nodes_generated.toLocaleString() : 'N/A'}</span></div>
        <div class="ccard-row"><span class="lbl">Nodes pruned</span><span class="val ${r.nodes_pruned!=null?'hl-green':''}">${r.nodes_pruned!=null ? r.nodes_pruned.toLocaleString() : 'N/A'}</span></div>
      </div>
      <div class="ccard-row"><span class="lbl">Time</span><span class="val">${escHtml(r.time_complexity)}</span></div>
      <div class="ccard-row"><span class="lbl">Space</span><span class="val">${escHtml(r.space_complexity)}</span></div>
    </div>`;
  }).join('');


  const cr   = results[currentTab] || results[results.length-1];
  const cset = new Set(cr.selected);
  const ctc  = cr.selected.reduce((s,i)=>s+sel_items[i].cost,0);

  const tabs = results.map((r,i)=>
    `<div class="algo-tab ${currentTab===i?'on':''}" onclick="switchTab(${i})">${escHtml(r.label)}</div>`
  ).join('');

  const chips = sel_items.map((p,i)=>`
    <div class="res-chip ${cset.has(i)?'sel':'rej'}">
      <span class="s-badge s-${p.sector.replace(/ /g,'-')}">${p.sector}</span>
      <span class="res-name">${escHtml(p.name)}</span>
      <span class="res-cost">${fmt(p.cost)}</span>
      <span class="res-score">${p.benefit.toFixed(1)}</span>
    </div>`).join('');

  document.getElementById('results').innerHTML = `
    <div class="card fade-in">
      <div class="card-hdr"><div class="card-title">Algorithm comparison — ${DS==='pasig'?'Pasig City':'Quezon City'}</div></div>
      ${banner}
      <div class="deflist">
        <div><b>Runtime</b> — time the algorithm spends actively executing, excluding setup.</div>
        <div><b>Execution time</b> — total time to process the data and produce the allocation.</div>
        <div><b>Pruning rate</b> — share of search-tree branches discarded without exploring.</div>
      </div>
      <div class="compare-grid">${ccards}</div>
    </div>
    <div class="card fade-in" style="animation-delay:.1s">
      <div class="card-title" style="margin-bottom:14px">Selected projects</div>
      <div class="algo-tabs" id="resTabs">${tabs}</div>
      <div class="stats-bar">
        <div class="stat"><div class="stat-l">Total benefit</div><div class="stat-v acc">${cr.total_benefit.toFixed(2)}</div><div class="stat-s">utility score</div></div>
        <div class="stat"><div class="stat-l">Budget used</div><div class="stat-v">${fmtB(ctc)}</div><div class="stat-s">${Math.round((ctc/budget_used)*100)}% of cap</div></div>
        <div class="stat"><div class="stat-l">Projects funded</div><div class="stat-v">${cr.selected.length}</div><div class="stat-s">of ${sel_items.length} candidates</div></div>
        <div class="stat"><div class="stat-l">Runtime</div><div class="stat-v">${cr.runtime_ms!=null ? cr.runtime_ms.toFixed(1)+' ms' : 'N/A'}</div><div class="stat-s">active execution only</div></div>
        <div class="stat"><div class="stat-l">Execution time</div><div class="stat-v">${cr.exec_ms!=null ? cr.exec_ms.toFixed(1)+' ms' : 'N/A'}</div><div class="stat-s">${escHtml(cr.time_complexity)}</div></div>
        <div class="stat"><div class="stat-l">Nodes explored</div><div class="stat-v">${cr.nodes_generated!=null ? cr.nodes_generated.toLocaleString() : 'N/A'}</div><div class="stat-s">${cr.nodes_generated!=null ? 'search tree size' : 'no B&B tree'}</div></div>
        <div class="stat"><div class="stat-l">Pruning rate</div><div class="stat-v ${cr.pruning_rate!=null?'acc':''}">${cr.pruning_rate!=null ? cr.pruning_rate.toFixed(2)+'%' : 'N/A'}</div><div class="stat-s">${cr.pruning_rate!=null ? 'branches pruned' : 'no B&B tree'}</div></div>
      </div>
      <div class="res-list">${chips}</div>
    </div>`;

  // Store for tab switching
  window._lastData = data;
}

function switchTab(i) {
  currentTab = i;
  renderResults(window._lastData);
}

function toggleNodes(i) {
  const box = document.getElementById('nt-' + i);
  const btn = document.getElementById('nt-btn-' + i);
  if (!box) return;
  const open = box.style.display === 'none';
  box.style.display = open ? 'block' : 'none';
  if (btn) btn.classList.toggle('open', open);
  if (open) _expandedNodes.add(i); else _expandedNodes.delete(i);
}

/* ── Inline messaging ────────────────────────────────────────────────
   Errors are shown in place rather than through alert(), so the user can
   read the problem and the offending control at the same time. */
function showMsg(kind, text) {
  const el = document.getElementById('globalMsg');
  if (!el) return;
  const icon = kind === 'err' ? '!' : kind === 'warn' ? '!' : 'i';
  el.innerHTML = `<div class="msg ${kind}"><span class="msg-icon" aria-hidden="true">${icon}</span><span>${escHtml(text)}</span></div>`;
}

function clearMsg() {
  const el = document.getElementById('globalMsg');
  if (el) el.innerHTML = '';
}

/* ── Persistent status bar ───────────────────────────────────────────
   Keeps the run configuration visible at all times and blocks the action
   before it can fail, stating the reason instead of reporting it after. */
function updateActionBar() {
  const selCount = CHECKED.size;
  const algos    = Object.keys(activeAlgos).filter(k => activeAlgos[k]);
  const trials   = Math.max(1, Math.min(30, parseInt(document.getElementById('trialsInput').value, 10) || 1));

  const set = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };
  set('abDs',     DS === 'pasig' ? 'Pasig City' : 'Quezon City');
  set('abBudget', fmtB(BUDGET));
  set('abSel',    selCount.toLocaleString());
  set('abAlgos',  algos.length);
  set('abTrials', trials);

  const selEl = document.getElementById('abSel');
  if (selEl) selEl.classList.toggle('warn', selCount === 0);
  const algEl = document.getElementById('abAlgos');
  if (algEl) algEl.classList.toggle('warn', algos.length === 0);

  // Error prevention: say what is missing before the button can be pressed.
  let reason = '';
  if (!selCount && !algos.length) reason = 'Select at least one project and one algorithm to run.';
  else if (!selCount)             reason = 'Select at least one project in step 3.';
  else if (!algos.length)         reason = 'Select at least one algorithm in step 4.';

  const btn = document.getElementById('runBtn');
  if (btn) {
    btn.disabled = !!reason;
    btn.textContent = trials > 1 ? `Run ${trials} times each` : 'Run algorithms';
  }
  const rEl = document.getElementById('abReason');
  if (rEl) rEl.textContent = reason;

  // Reflect progress on the numbered rail.
  const mark = (id, done) => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('done', done);
  };
  mark('stepProjects', selCount > 0);
  mark('setup',        algos.length > 0);

  updatePhaseNav();
}

/* ── Trials control ──────────────────────────────────────────────────── */
function getTrials() {
  const el = document.getElementById('trialsInput');
  let v = parseInt(el.value, 10);
  if (isNaN(v) || v < 1) v = 1;
  if (v > 30) v = 30;
  el.value = v;
  return v;
}

function setTrials(n) {
  document.getElementById('trialsInput').value = n;
  updateActionBar();
}

/* ── Run history / experiment log ────────────────────────────────────── */
let HIST_DATA   = [];     // full combo list incl. per-trial rows
let HIST_OPEN   = null;   // "ds|algo" of the currently expanded combo

async function loadHistory() {
  try {
    const resp = await fetch('/api/history');
    const data = await resp.json();
    renderHistory(data.combos);
  } catch(err) {
    // Non-fatal: the history panel simply stays empty.
  }
}

function histKey(c) { return c.ds + '|' + c.algo; }

function toggleHist(key) {
  HIST_OPEN = (HIST_OPEN === key) ? null : key;
  renderHistory(HIST_DATA);
}

function num(v, dp) {
  return (v === null || v === undefined) ? '—' : Number(v).toFixed(dp);
}

/* Mean of a numeric field across a combo's stored runs. */
function histMean(trials, field) {
  const vals = trials.map(t => t[field]).filter(v => v !== null && v !== undefined);
  if (!vals.length) return null;
  return vals.reduce((a,b) => a+b, 0) / vals.length;
}

function renderHistory(combos) {
  const panel = document.getElementById('historyPanel');
  if (!panel) return;

  HIST_DATA = combos || [];
  const total = HIST_DATA.reduce((s,c) => s + c.count, 0);

  if (!HIST_DATA.length) {
    panel.innerHTML = `
      <div class="card fade-in">
        <div class="card-hdr">
          <div class="card-title">Run history</div>
        </div>
        <div class="hist-empty">
          No runs recorded yet.<br>
          Run an algorithm above — each run's runtime, execution time and
          pruning rate will be logged here
          (last 30 runs kept per dataset + algorithm).
        </div>
      </div>`;
    return;
  }

  // Keep the expanded combo valid; otherwise open the first one by default.
  if (!HIST_DATA.some(c => histKey(c) === HIST_OPEN)) {
    HIST_OPEN = histKey(HIST_DATA[0]);
  }

  const cards = HIST_DATA.map(c => {
    const key  = histKey(c);
    const pct  = Math.min(100, Math.round((c.count / 30) * 100));
    const done = c.complete;
    const on   = HIST_OPEN === key;
    return `
      <div class="hist-card ${done?'done':''} ${on?'on':''}" onclick="toggleHist('${key}')">
        <div class="hist-algo">${escHtml(c.algo_name)}</div>
        <div class="hist-ds">${escHtml(c.ds_name)}</div>
        <span class="hist-count ${done?'done':''}">${c.count}</span>
        <span class="hist-of"> / 30 runs</span>
        <div class="hist-bar"><div class="hist-bar-fill ${done?'done':''}" style="width:${pct}%"></div></div>
      </div>`;
  }).join('');

  // ── Per-run detail table for the expanded combination ────────────────
  const combo  = HIST_DATA.find(c => histKey(c) === HIST_OPEN);
  let detail = '';

  if (combo) {
    const t = combo.trials;
    const rows = t.map((r, i) => `
      <tr>
        <td class="hr-idx">Run ${i + 1}</td>
        <td>${num(r.runtime_ms, 2)}</td>
        <td>${num(r.exec_ms, 2)}</td>
        <td>${r.pruning_rate === null ? '—' : num(r.pruning_rate, 2)}</td>
        <td class="hr-dim">${r.nodes_generated === null ? '—' : r.nodes_generated.toLocaleString()}</td>
        <td class="hr-dim">${r.nodes_pruned === null ? '—' : r.nodes_pruned.toLocaleString()}</td>
        <td class="hr-dim">${num(r.total_benefit, 2)}</td>
        <td class="hr-time">${escHtml(r.timestamp.split(' ')[1] || '')}</td>
      </tr>`).join('');

    const mRun  = histMean(t, 'runtime_ms');
    const mExec = histMean(t, 'exec_ms');
    const mPrun = histMean(t, 'pruning_rate');
    const mBen  = histMean(t, 'total_benefit');

    detail = `
      <div class="hist-detail">
        <div class="hist-detail-hdr">
          <span class="hd-title">${escHtml(combo.algo_name)} · ${escHtml(combo.ds_name)}</span>
          <span class="hd-sub">${combo.count} run${combo.count===1?'':'s'} stored</span>
        </div>
        <div class="hist-table-wrap">
          <table class="hist-table">
            <thead>
              <tr>
                <th>Trial</th>
                <th>Runtime (ms)</th>
                <th>Execution time (ms)</th>
                <th>Pruning rate (%)</th>
                <th>Nodes explored</th>
                <th>Nodes pruned</th>
                <th>Total benefit</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
            <tfoot>
              <tr>
                <td class="hr-idx">Mean</td>
                <td>${num(mRun, 4)}</td>
                <td>${num(mExec, 4)}</td>
                <td>${mPrun === null ? '—' : num(mPrun, 4)}</td>
                <td class="hr-dim">—</td>
                <td class="hr-dim">—</td>
                <td class="hr-dim">${num(mBen, 4)}</td>
                <td class="hr-time"></td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>`;
  }

  panel.innerHTML = `
    <div class="card fade-in">
      <div class="card-hdr">
        <div class="card-title">Run history — ${total} run${total===1?'':'s'} recorded</div>
        <div class="hist-actions">
          <label class="xl-opt" title="Repeats every funded row with a Dataset column so both cities can be pivoted together. Roughly doubles the file size.">
            <input type="checkbox" id="xlCombined" checked> combined sheet
          </label>
          <button class="hist-btn export" onclick="exportExcel()">⬇ Export to Excel</button>
          <button class="hist-btn danger" onclick="clearHistory()">Clear history</button>
        </div>
      </div>
      <div class="hist-grid">${cards}</div>
      ${detail}
    </div>`;
}

function exportExcel() {
  // The combined cross-city sheet repeats every funded row, which roughly
  // doubles the file size and build time, so it is switchable.
  const cb = document.getElementById('xlCombined');
  const combined = cb ? (cb.checked ? '1' : '0') : '1';
  window.location.href = '/api/export?combined=' + combined;
}

function clearHistory() {
  const modal = document.getElementById('clearHistoryModal');
  modal.style.display = 'flex';
  // Small delay to allow display:flex to apply before adding opacity class for transition
  setTimeout(() => {
    modal.classList.add('show');
  }, 10);
}

function closeClearHistoryModal() {
  const modal = document.getElementById('clearHistoryModal');
  modal.classList.remove('show');
  // Wait for transition to finish before hiding
  setTimeout(() => {
    modal.style.display = 'none';
  }, 300);
}

async function confirmClearHistory() {
  closeClearHistoryModal();
  try {
    const resp = await fetch('/api/history/clear', {
      method:  'POST',
      headers: {'Content-Type':'application/json'},
      body:    JSON.stringify({}),
    });
    const data = await resp.json();
    renderHistory(data.combos);
    loadStats();
  } catch(err) {
    showMsg('err', 'Could not clear the history: ' + err.message);
  }
}

/* ── Statistical report ──────────────────────────────────────────────
   Implements Figure 6's "Analysis and Logging Module" output: the paired
   t-test (KB vs KBG) and the independent t-test / efficiency ratio across
   dataset sizes. Computed server-side from the recorded runs. */
async function loadStats() {
  try {
    const resp = await fetch('/api/stats');
    renderStats(await resp.json());
  } catch(err) { /* non-fatal */ }
}

function fx(v, dp) {
  return (v === null || v === undefined) ? '—' : Number(v).toFixed(dp);
}

/* p-values here span many orders of magnitude, so very small ones are shown
   in scientific notation rather than rounding to a misleading 0.0000. */
function fp(v) {
  if (v === null || v === undefined) return null;
  if (v < 0.0001) return v.toExponential(2);
  return v.toFixed(4);
}

function verdict(row) {
  if (row.p === null || row.p === undefined) {
    return '<span class="vd none" title="Metric is constant across runs">not computable</span>';
  }
  return row.significant
    ? '<span class="vd sig">significant</span>'
    : '<span class="vd ns">not significant</span>';
}

function renderStats(rep) {
  const panel = document.getElementById('statsPanel');
  if (!panel) return;

  const hasEff  = rep.efficiency && rep.efficiency.length;
  const hasComp = rep.comparison && rep.comparison.length;

  if (!hasEff && !hasComp) {
    panel.innerHTML = `
      <div class="card">
        <div class="empty-state">
          <p>No statistics yet.</p>
          <p class="es-sub">The efficiency ratio needs runs on <b>both</b> datasets, and the
          paired t-test needs runs of <b>both</b> B&amp;B and B&amp;B+GA. Record at least two runs
          of each, then the report appears here.</p>
        </div>
      </div>`;
    return;
  }

  let html = '';

  if (hasComp) {
    const blocks = rep.comparison.map(g => `
      <div class="stat-block">
        <div class="sb-title">${escHtml(g.ds_name)}</div>
        <div class="stat-table-wrap">
        <table class="hist-table">
          <thead><tr>
            <th>Metric</th><th>Mean (KB)</th><th>Mean (KBG)</th>
            <th>Mean diff.</th><th>SD of diff.</th><th>t</th><th>p</th><th>Result</th>
          </tr></thead>
          <tbody>
            ${g.rows.map(r => `
              <tr>
                <td class="hr-idx">${escHtml(r.metric)}</td>
                <td>${fx(r.mean_a,4)}</td>
                <td>${fx(r.mean_b,4)}</td>
                <td>${fx(r.mean_diff,4)}</td>
                <td>${fx(r.sd_diff,4)}</td>
                <td>${r.t===null||r.t===undefined?'—':fx(r.t,4)}</td>
                <td>${fp(r.p) ?? '—'}</td>
                <td>${verdict(r)}</td>
              </tr>`).join('')}
          </tbody>
        </table></div>
      </div>`).join('');

    html += `
      <div class="card">
        <div class="card-hdr"><div class="card-title">Paired t-test — B&amp;B vs. B&amp;B + GA</div></div>
        <p class="stat-lede">Tests whether the hybrid differs significantly from branch-and-bound alone.
        Runs are paired by problem instance: both solve the identical project set under the identical budget.
        Null hypothesis rejected when p &lt; 0.05.</p>
        ${blocks}
      </div>`;
  }

  if (hasEff) {
    const blocks = rep.efficiency.map(g => `
      <div class="stat-block">
        <div class="sb-title">${escHtml(g.algo_name)}</div>
        <div class="stat-table-wrap">
        <table class="hist-table">
          <thead><tr>
            <th>Metric</th><th>Mean (Pasig)</th><th>Mean (QC)</th>
            <th>Ratio</th><th>t</th><th>df</th><th>p</th><th>Result</th>
          </tr></thead>
          <tbody>
            ${g.rows.map(r => `
              <tr>
                <td class="hr-idx">${escHtml(r.metric)}</td>
                <td>${fx(r.mean_1,4)}</td>
                <td>${fx(r.mean_2,4)}</td>
                <td>${fx(r.ratio,4)}</td>
                <td>${r.t===null||r.t===undefined?'—':fx(r.t,4)}</td>
                <td>${fx(r.df,2)}</td>
                <td>${fp(r.p) ?? '—'}</td>
                <td>${verdict(r)}</td>
              </tr>`).join('')}
          </tbody>
        </table></div>
      </div>`).join('');

    html += `
      <div class="card">
        <div class="card-hdr"><div class="card-title">Independent t-test — efficiency across dataset sizes</div></div>
        <p class="stat-lede">Efficiency is the ratio of each optimality parameter on the small dataset
        relative to the large one. A result that is <i>not</i> significant indicates performance is stable
        across input sizes.</p>
        ${blocks}
      </div>`;
  }

  html += `
    <div class="msg info" style="margin-top:14px">
      <span class="msg-icon" aria-hidden="true">i</span>
      <span>Two-tailed tests at the 0.05 level. "Not computable" means the metric is constant
      across runs, so its standard deviation is zero — expected for the pruning rate of
      branch-and-bound, which is deterministic, and not a data error.</span>
    </div>`;

  panel.innerHTML = html;
}

init();
loadHistory();
loadStats();
/* ── Post-allocation verification and comparative review ──────────────────
   Step 7 of the Data Generation/Gathering Procedure. The manuscript specifies
   a MANUAL verification phase, so this does not replace the reviewer's
   judgement - it supplies the material. Automated checks establish the
   mechanical facts over every funded project; the reviewer records the
   judgement on a purposive sample. */
let REVIEW = null;

async function loadReview() {
  const vp = document.getElementById('verifyPanel');
  if (!LAST_RUN) {
    vp.innerHTML = `<div class="card"><div class="empty-state">
      <p>No allocation to verify yet.</p>
      <p class="es-sub">Run the algorithms first. The verification works from the
      project list the algorithm produced.</p></div></div>`;
    renderCompare();
    return;
  }
  vp.innerHTML = `<div class="card"><div class="empty-state"><p>Checking…</p></div></div>`;
  try {
    const resp = await fetch('/api/review', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        ds: LAST_RUN.ds, budget: LAST_RUN.budget,
        selected: LAST_RUN.selected, funded: LAST_RUN.funded,
      }),
    });
    REVIEW = await resp.json();
    if (REVIEW.error) throw new Error(REVIEW.error);
    renderVerify(); renderCompare(); loadFunded(true);
  } catch (err) {
    vp.innerHTML = `<div class="banner warn"><span>⚠</span> ${escHtml(err.message)}</div>`;
  }
}

function renderVerify() {
  const r = REVIEW;
  const rows = r.checks.map(c => `
    <tr>
      <td class="hr-idx">${escHtml(c.check)}</td>
      <td>${escHtml(c.result)}</td>
      <td>${c.pass ? '<span class="vd sig">pass</span>' : '<span class="vd none">review</span>'}</td>
    </tr>`).join('');

  const sect = r.sectors.map(s => `
    <tr>
      <td class="hr-idx">${escHtml(s.sector)}</td>
      <td>${s.in_plan.toLocaleString()}</td>
      <td>${s.funded.toLocaleString()}</td>
      <td>${s.funded_pct.toFixed(1)}%</td>
      <td>${s.share_plan.toFixed(1)}%</td>
      <td>${s.share_funded.toFixed(1)}%</td>
      <td>${(s.share_funded - s.share_plan >= 0 ? '+' : '') + (s.share_funded - s.share_plan).toFixed(1)} pts</td>
    </tr>`).join('');

  document.getElementById('verifyPanel').innerHTML = `
    <div class="card">
      <div class="card-hdr"><div class="card-title">Automated integrity checks</div>
        <div class="card-meta">${r.funded.toLocaleString()} funded of ${r.candidates.toLocaleString()} candidates</div></div>
      <p class="stat-lede">These checks cover <b>every</b> funded project. They establish mechanical
      facts — traceability to the source plan, budget compliance, arithmetic consistency — and do not
      substitute for the reviewer's judgement below.</p>
      <div class="stat-table-wrap"><table class="hist-table">
        <thead><tr><th>Check</th><th>Result</th><th>Status</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
    </div>

    <div class="card" style="margin-top:14px">
      <div class="card-hdr"><div class="card-title">Sector alignment with the published plan</div></div>
      <p class="stat-lede">Compares each sector's share of the candidate plan with its share of the
      funded set. A small difference indicates the allocation broadly reflects the local government's
      own distribution of projects.</p>
      <div class="stat-table-wrap"><table class="hist-table">
        <thead><tr><th>Sector</th><th>In plan</th><th>Funded</th><th>% of sector</th>
        <th>Share of plan</th><th>Share of funded</th><th>Difference</th></tr></thead>
        <tbody>${sect}</tbody></table></div>
    </div>`;
}

/* Keeps a visible count of saved entries, so it is obvious whether anything
   will reach the export. */
/* Comparative review across the two datasets, built from the allocations
   recorded so far in this session. */
let ALLOC_SUMMARY = {};

function renderCompare() {
  const keys = Object.keys(ALLOC_SUMMARY);
  const el = document.getElementById('comparePanel');
  if (keys.length < 2) {
    el.innerHTML = `<div class="card"><div class="empty-state">
      <p>Both datasets needed.</p>
      <p class="es-sub">Run an allocation on Pasig and on Quezon City at the same budget,
      then the scaling comparison appears here.</p></div></div>`;
    return;
  }
  const row = (k) => {
    const a = ALLOC_SUMMARY[k];
    return `<tr>
      <td class="hr-idx">${escHtml(a.name)}</td>
      <td>${a.candidates.toLocaleString()}</td>
      <td>${a.funded.toLocaleString()}</td>
      <td>${(a.funded/a.candidates*100).toFixed(1)}%</td>
      <td>₱${a.budget.toLocaleString(undefined,{maximumFractionDigits:0})}</td>
      <td>₱${a.cost.toLocaleString(undefined,{maximumFractionDigits:0})}</td>
      <td>${a.benefit.toLocaleString(undefined,{maximumFractionDigits:2})}</td>
      <td>${a.perM.toFixed(2)}</td></tr>`;
  };
  el.innerHTML = `
    <div class="card">
      <div class="card-hdr"><div class="card-title">Small vs. large dataset</div></div>
      <p class="stat-lede">How the allocation behaves as the candidate pool grows. Compare at the
      same budget, or at the same proportion of each city's plan — and state which in your write-up.</p>
      <div class="stat-table-wrap"><table class="hist-table">
        <thead><tr><th>Dataset</th><th>Candidates</th><th>Funded</th><th>% funded</th>
        <th>Budget</th><th>Spent</th><th>Total benefit</th><th>Benefit per ₱1M</th></tr></thead>
        <tbody>${keys.map(row).join('')}</tbody></table></div>
    </div>`;
}


/* Records the most recent allocation so the verification phase has something
   to check, and keeps one summary per dataset for the scaling comparison. */
let LAST_RUN = null;

function captureAllocation(data, candidateIdx) {
  // Prefer the hybrid's allocation when present; both are exact, so the
  // selections agree, but the hybrid is the proposed model.
  const r = data.results.find(x => x.algo === 'ga') || data.results[0];
  if (!r) return;

  LAST_RUN = {
    ds: data.ds, budget: data.budget,
    selected: candidateIdx, funded: r.selected, algo: r.label,
  };

  const items = data.selected_items || [];
  const cost = r.selected.reduce((t, i) => t + (items[i] ? Number(items[i].cost) : 0), 0);
  ALLOC_SUMMARY[data.ds] = {
    name: data.ds === 'pasig' ? 'Pasig City (small)' : 'Quezon City (large)',
    candidates: candidateIdx.length, funded: r.selected.length,
    budget: data.budget, cost, benefit: r.total_benefit,
    perM: cost ? r.total_benefit / (cost / 1e6) : 0,
  };
}

/* ── Funded project browser ───────────────────────────────────────────────
   The complete allocation for each city, paged on the server because a
   Quezon City run can exceed 25,000 projects. This is the same list the
   export contains; the manual review itself is done in Excel, where the
   funded sheets carry Reviewer verdict and note columns. */
let FUNDED_STATE = { ds: null, algo: null, page: 1, q: '', sector: 'All', sort: 'benefit' };
let FUNDED_DATA = null;

async function loadFunded(reset) {
  if (reset) FUNDED_STATE.page = 1;
  const qs = new URLSearchParams({
    ds: FUNDED_STATE.ds || 'pasig', algo: FUNDED_STATE.algo || 'ga',
    page: FUNDED_STATE.page, q: FUNDED_STATE.q,
    sector: FUNDED_STATE.sector, sort: FUNDED_STATE.sort,
  });
  try {
    const resp = await fetch('/api/funded?' + qs.toString());
    FUNDED_DATA = await resp.json();
    if (!FUNDED_STATE.ds && FUNDED_DATA.available.length) {
      FUNDED_STATE.ds = FUNDED_DATA.available[0].ds;
      FUNDED_STATE.algo = FUNDED_DATA.available[0].algo;
    }
    renderFunded();
  } catch (err) { /* non-fatal */ }
}

function fundedPick(ds, algo) {
  FUNDED_STATE.ds = ds; FUNDED_STATE.algo = algo; loadFunded(true);
}
function fundedPage(p) { FUNDED_STATE.page = p; loadFunded(false); }
function fundedSort(v) { FUNDED_STATE.sort = v; loadFunded(true); }
function fundedSector(v) { FUNDED_STATE.sector = v; loadFunded(true); }
function fundedSearch(v) { FUNDED_STATE.q = v; loadFunded(true); }

function renderFunded() {
  const el = document.getElementById('fundedPanel');
  if (!el) return;
  const d = FUNDED_DATA;

  if (!d || !d.available.length) {
    el.innerHTML = `<div class="card"><div class="empty-state">
      <p>No allocation yet.</p>
      <p class="es-sub">Run the algorithms on a dataset and the funded projects appear here.</p>
    </div></div>`;
    return;
  }

  const picks = d.available.map(a => `
    <button class="fd-pick ${a.ds === d.ds && a.algo === d.algo ? 'on' : ''}"
            onclick="fundedPick('${a.ds}','${a.algo}')">
      ${escHtml(a.ds_name)} · ${escHtml(a.algo_name)}
      <span class="fd-n">${a.count.toLocaleString()}</span>
    </button>`).join('');

  const opts = ['All', ...d.sectors].map(s =>
    `<option value="${escHtml(s)}" ${s === FUNDED_STATE.sector ? 'selected' : ''}>${escHtml(s)}</option>`).join('');

  const rows = d.rows.map(r => `
    <tr>
      <td class="hr-idx">${escHtml(r.code)}</td>
      <td style="text-align:left;white-space:normal;max-width:380px">${escHtml(r.name.slice(0,150))}</td>
      <td style="text-align:left">${escHtml(r.pmo)}</td>
      <td>${escHtml(r.sector)}</td>
      <td>₱${r.cost.toLocaleString(undefined,{maximumFractionDigits:0})}</td>
      <td>${r.benefit.toFixed(3)}</td>
      <td>${r.ratio === null ? '—' : r.ratio.toLocaleString(undefined,{maximumFractionDigits:2})}</td>
    </tr>`).join('');

  // Compact pager: first, a window around the current page, last.
  const P = d.pages, cur = d.page;
  const want = new Set([1, P, cur, cur-1, cur+1, cur-2, cur+2]);
  const nums = [...want].filter(n => n >= 1 && n <= P).sort((a,b)=>a-b);
  let pager = '', prev = 0;
  for (const n of nums) {
    if (prev && n - prev > 1) pager += `<span class="pg-gap">…</span>`;
    pager += `<button class="pg-btn ${n===cur?'on':''}" onclick="fundedPage(${n})">${n}</button>`;
    prev = n;
  }

  el.innerHTML = `
    <div class="card">
      <div class="card-hdr">
        <div class="card-title">Funded projects</div>
        <div class="card-meta">budget ₱${d.budget.toLocaleString(undefined,{maximumFractionDigits:0})}</div>
      </div>
      <div class="fd-picks">${picks}</div>

      <div class="fd-controls">
        <input class="fd-search" placeholder="Search project, office or code…"
               value="${escHtml(FUNDED_STATE.q)}"
               oninput="clearTimeout(window._fdT);window._fdT=setTimeout(()=>fundedSearch(this.value),300)">
        <select class="fd-sel" onchange="fundedSector(this.value)">${opts}</select>
        <select class="fd-sel" onchange="fundedSort(this.value)">
          <option value="benefit" ${FUNDED_STATE.sort==='benefit'?'selected':''}>Highest benefit</option>
          <option value="cost"    ${FUNDED_STATE.sort==='cost'?'selected':''}>Highest cost</option>
          <option value="ratio"   ${FUNDED_STATE.sort==='ratio'?'selected':''}>Best benefit per ₱1M</option>
          <option value="name"    ${FUNDED_STATE.sort==='name'?'selected':''}>Project name</option>
        </select>
        <span class="fd-count">${d.total.toLocaleString()} project${d.total===1?'':'s'}
          · ₱${d.sum_cost.toLocaleString(undefined,{maximumFractionDigits:0})}
          · benefit ${d.sum_benefit.toLocaleString(undefined,{maximumFractionDigits:2})}</span>
      </div>

      <div class="stat-table-wrap" style="max-height:520px">
        <table class="hist-table">
          <thead><tr>
            <th>Code</th><th style="text-align:left">Project / program</th>
            <th style="text-align:left">Implementing office</th><th>Sector</th>
            <th>Cost</th><th>Benefit</th><th>per ₱1M</th>
          </tr></thead>
          <tbody>${rows || '<tr><td colspan="7" style="text-align:center;padding:18px">No projects match.</td></tr>'}</tbody>
        </table>
      </div>
      <div class="fd-pager">
        <span class="fd-pageinfo">Page ${cur} of ${P}</span>
        <div class="fd-pagebtns">
          <button class="pg-btn" ${cur<=1?'disabled':''} onclick="fundedPage(${cur-1})">←</button>
          ${pager}
          <button class="pg-btn" ${cur>=P?'disabled':''} onclick="fundedPage(${cur+1})">→</button>
        </div>
      </div>
    </div>`;
}
