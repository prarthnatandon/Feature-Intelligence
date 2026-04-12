/**
 * dashboard.js — state machine, SSE client, all UI event handling
 *
 * States: landing → running → results
 */

'use strict';

let currentRunId = null;
let eventSource = null;
let briefData = null;
let toolCallCount = 0;
let activeAgents = new Set();
let thinkingVisible = true;
let thinkingBuffer = '';

const FREQUENCY_ORDER = { very_high: 4, high: 3, medium: 2, low: 1 };
const EFFORT_LABELS = {
  quick_win: 'Quick win',
  medium_lift: 'Medium lift',
  major_investment: 'Major investment',
};
const EVIDENCE_COLORS = { strong: 'green', moderate: 'blue', emerging: 'orange' };
const UNIQUENESS_COLORS = { high: 'green', medium: 'blue', low: 'orange' };

// ===========================================================================
// Run timer
// ===========================================================================

let timerInterval = null;
let timerStart = null;

function startTimer() {
  timerStart = Date.now();
  const el = document.getElementById('run-timer');
  if (el) el.classList.remove('hidden');
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    const secs = Math.floor((Date.now() - timerStart) / 1000);
    const m = Math.floor(secs / 60);
    const s = String(secs % 60).padStart(2, '0');
    if (el) el.textContent = `${m}:${s}`;
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
  const el = document.getElementById('run-timer');
  if (el) el.classList.add('hidden');
}

// ===========================================================================
// State transitions — animated, never blank
// ===========================================================================

function showLanding() {
  const landing = document.getElementById('state-landing');
  const running = document.getElementById('state-running');
  const results = document.getElementById('state-results');
  landing.classList.remove('exiting');
  landing.style.display = '';
  running.style.display = 'none';
  results.style.display = 'none';
  document.getElementById('app-header').classList.add('hidden');
}

function showRunning() {
  const running = document.getElementById('state-running');
  document.getElementById('state-landing').style.display = 'none';
  document.getElementById('state-results').style.display = 'none';
  running.style.display = 'block';
  running.classList.add('entering');
  running.addEventListener('animationend', () => running.classList.remove('entering'), { once: true });
  document.getElementById('app-header').classList.remove('hidden');
  setStatus('running', 'Running...');
  startTimer();
}

function showResults() {
  const results = document.getElementById('state-results');
  document.getElementById('state-landing').style.display = 'none';
  document.getElementById('state-running').style.display = 'none';
  results.style.display = 'block';
  results.classList.add('entering');
  results.addEventListener('animationend', () => results.classList.remove('entering'), { once: true });
  document.getElementById('app-header').classList.remove('hidden');
  setStatus('complete', 'Complete');
  document.getElementById('download-btn').classList.remove('hidden');
  const shareBtn = document.getElementById('share-btn');
  if (shareBtn) shareBtn.classList.remove('hidden');
  stopTimer();
}

function resetToLanding() {
  if (eventSource) { eventSource.close(); eventSource = null; }
  currentRunId = null;
  briefData = null;
  toolCallCount = 0;
  activeAgents.clear();
  thinkingBuffer = '';
  stopTimer();
  resetRunningUI();
  showLanding();
}

function handlePipelineError(rawError) {
  if (eventSource) { eventSource.close(); eventSource = null; }
  stopTimer();

  // Determine a user-friendly message
  let message = 'Something went wrong during analysis.';
  if (rawError.includes('rate_limit') || rawError.includes('429')) {
    message = 'API rate limit reached. Please wait 60 seconds and try again.';
  } else if (rawError.includes('auth') || rawError.includes('401')) {
    message = 'API key invalid. Check your ANTHROPIC_API_KEY configuration.';
  } else if (rawError.includes('timeout') || rawError.includes('timed out')) {
    message = 'Analysis timed out. Try again — partial results may still load.';
  }

  setStatus('error', 'Error');

  // Show error banner inside running state (don't send user away)
  const running = document.getElementById('state-running');
  let errBanner = document.getElementById('pipeline-error-banner');
  if (!errBanner) {
    errBanner = document.createElement('div');
    errBanner.id = 'pipeline-error-banner';
    errBanner.style.cssText = [
      'margin: 24px auto', 'max-width: 560px', 'padding: 20px 24px',
      'background: #fef2f2', 'border: 1px solid #fecaca', 'border-radius: 12px',
      'display: flex', 'flex-direction: column', 'gap: 12px', 'align-items: flex-start',
    ].join(';');
    running.querySelector('.running-inner')?.appendChild(errBanner) || running.appendChild(errBanner);
  }
  errBanner.style.display = 'flex';
  errBanner.innerHTML = `
    <div style="font-weight:600;color:#991b1b;font-size:15px;">Analysis stopped</div>
    <div style="color:#7f1d1d;font-size:14px;line-height:1.5;">${message}</div>
    <button onclick="dismissErrorAndRetry()" style="
      margin-top:4px;padding:8px 18px;background:#58cc02;color:#fff;
      border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;">
      Try Again
    </button>
  `;
}

function dismissErrorAndRetry() {
  const banner = document.getElementById('pipeline-error-banner');
  if (banner) banner.style.display = 'none';
  resetToLanding();
  const btn = document.getElementById('landing-btn');
  if (btn) { btn.disabled = false; btn.innerHTML = 'Run Analysis →'; }
}

// ===========================================================================
// Company form helpers
// ===========================================================================

const KNOWN_DOMAINS = {
  'duolingo': 'duolingo.com', 'spotify': 'spotify.com', 'notion': 'notion.so',
  'airbnb': 'airbnb.com', 'uber': 'uber.com', 'lyft': 'lyft.com',
  'netflix': 'netflix.com', 'youtube': 'youtube.com', 'instagram': 'instagram.com',
  'twitter': 'twitter.com', 'tiktok': 'tiktok.com', 'snapchat': 'snapchat.com',
  'reddit': 'reddit.com', 'discord': 'discord.com', 'slack': 'slack.com',
  'zoom': 'zoom.us', 'figma': 'figma.com', 'canva': 'canva.com',
  'dropbox': 'dropbox.com', 'evernote': 'evernote.com', 'todoist': 'todoist.com',
  'headspace': 'headspace.com', 'calm': 'calm.com', 'strava': 'strava.com',
  'peloton': 'onepeloton.com', 'robinhood': 'robinhood.com', 'coinbase': 'coinbase.com',
  'babbel': 'babbel.com', 'rosettastone': 'rosettastone.com',
  'linear': 'linear.app', 'vercel': 'vercel.com', 'github': 'github.com',
  'gitlab': 'gitlab.com', 'asana': 'asana.com', 'trello': 'trello.com',
  'shopify': 'shopify.com', 'grammarly': 'grammarly.com', 'obsidian': 'obsidian.md',
  'anki': 'apps.ankiweb.net', 'pinterest': 'pinterest.com', 'linkedin': 'linkedin.com',
};

const KNOWN_DESCRIPTIONS = {
  'duolingo':      'language learning app',
  'spotify':       'music and podcast streaming app',
  'notion':        'all-in-one workspace and note-taking app',
  'slack':         'team messaging and collaboration platform',
  'discord':       'voice, video, and text chat for communities',
  'airbnb':        'home rental and travel booking marketplace',
  'uber':          'ride-hailing and delivery app',
  'lyft':          'ride-hailing app',
  'netflix':       'video streaming subscription service',
  'youtube':       'video sharing and streaming platform',
  'instagram':     'photo and video social media app',
  'tiktok':        'short-form video social media app',
  'snapchat':      'ephemeral photo and messaging app',
  'reddit':        'social news and discussion forum',
  'zoom':          'video conferencing and meetings platform',
  'figma':         'collaborative UI design tool',
  'canva':         'online graphic design platform',
  'shopify':       'e-commerce platform for online stores',
  'grammarly':     'AI writing assistant and grammar checker',
  'headspace':     'meditation and mindfulness app',
  'calm':          'sleep, meditation, and relaxation app',
  'strava':        'fitness tracking app for runners and cyclists',
  'peloton':       'connected fitness platform and equipment',
  'robinhood':     'commission-free stock trading app',
  'coinbase':      'cryptocurrency exchange and wallet',
  'todoist':       'task manager and to-do list app',
  'asana':         'project management and team collaboration tool',
  'trello':        'visual project management with boards and cards',
  'github':        'code hosting and developer collaboration platform',
  'gitlab':        'DevOps and code collaboration platform',
  'linear':        'issue tracking and project management for software teams',
  'vercel':        'frontend cloud deployment platform',
  'obsidian':      'local-first knowledge base and note-taking app',
  'anki':          'spaced repetition flashcard app',
  'babbel':        'language learning app',
  'pinterest':     'visual discovery and image bookmarking platform',
  'linkedin':      'professional networking and job search platform',
  'dropbox':       'cloud storage and file sharing platform',
  'evernote':      'note-taking and organisation app',
  'todoist':       'task manager and to-do list app',
};

function _guessDomain(name) {
  const key = name.trim().toLowerCase().replace(/\s+/g, '');
  return KNOWN_DOMAINS[key] || `${key}.com`;
}

function _guessDescription(name) {
  const key = name.trim().toLowerCase().replace(/\s+/g, '');
  return KNOWN_DESCRIPTIONS[key] || '';
}

function onCompanyInput(value) {
  const nameEl = document.getElementById('hero-company-name');
  if (nameEl) nameEl.textContent = value.trim() || 'Your Company';

  // Auto-fill "What they make" if we recognise the company
  const productEl = document.getElementById('input-product');
  if (productEl) {
    const desc = _guessDescription(value);
    if (desc) {
      productEl.value = desc;
      productEl.style.color = '';
    }
  }

  // Update favicon (shown inline inside the input)
  const faviconEl = document.getElementById('company-favicon');
  const wrapperEl = faviconEl?.parentElement;
  if (faviconEl) {
    if (value.trim().length > 1) {
      const domain = _guessDomain(value.trim());
      faviconEl.src = `https://www.google.com/s2/favicons?domain=${domain}&sz=32`;
      faviconEl.style.display = 'inline-block';
      wrapperEl?.classList.add('has-favicon');
    } else {
      faviconEl.style.display = 'none';
      faviconEl.src = '';
      wrapperEl?.classList.remove('has-favicon');
    }
    // Hide favicon if image fails to load
    faviconEl.onerror = () => {
      faviconEl.style.display = 'none';
      wrapperEl?.classList.remove('has-favicon');
    };
  }
}

function getCompanyPayload() {
  return {
    company_name: (document.getElementById('input-company')?.value || 'Duolingo').trim(),
    product_description: (document.getElementById('input-product')?.value || 'consumer app').trim(),
    subreddit: (document.getElementById('input-subreddit')?.value || '').trim().replace(/^r\//, ''),
    app_store_id: '',
    known_features: (document.getElementById('input-features')?.value || '').trim(),
  };
}

// ===========================================================================
// Start analysis
// ===========================================================================

async function startAnalysis() {
  const payload = getCompanyPayload();
  window._currentPayload = payload;

  // Validate — company name required
  if (!payload.company_name) {
    const input = document.getElementById('input-company');
    if (input) { input.focus(); input.style.borderColor = '#ef4444'; setTimeout(() => input.style.borderColor = '', 2000); }
    return;
  }

  const btn = document.getElementById('landing-btn');
  btn.disabled = true;
  btn.innerHTML = `<span class="btn-spinner"></span>Analyzing ${payload.company_name}...`;

  // Store company name for use in running/results UI
  window._currentCompany = payload.company_name;
  setRunningCompanyName(payload.company_name);

  // ── Optimistic transition — landing fades out immediately ──────────────
  const landing = document.getElementById('state-landing');
  landing.classList.add('exiting');

  await new Promise(r => setTimeout(r, 320));
  showRunning();
  setStatusLine(`Connecting to analysis engine for ${payload.company_name}...`);

  try {
    const res = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || 'Failed to start');
    }
    const data = await res.json();
    currentRunId = data.run_id;
    connectSSE(currentRunId);
  } catch (e) {
    resetToLanding();
    const errEl = document.getElementById('landing-error');
    if (errEl) {
      errEl.textContent = `Could not start analysis: ${e.message}`;
      errEl.classList.remove('hidden');
      setTimeout(() => errEl.classList.add('hidden'), 5000);
    }
    btn.disabled = false;
    btn.innerHTML = 'Run Analysis →';
  }
}

function setRunningCompanyName(name) {
  // Update all dynamic company name placeholders in the running + results UI
  document.querySelectorAll('.dynamic-company-name').forEach(el => {
    el.textContent = name;
  });
}

// ===========================================================================
// SSE connection
// ===========================================================================

function connectSSE(runId) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(`/stream/${runId}`);

  eventSource.onmessage = (e) => {
    let ev;
    try { ev = JSON.parse(e.data); } catch { return; }
    handleEvent(ev);
  };

  eventSource.onerror = () => {
    setStatusLine('Connection lost — check server');
    eventSource.close();
  };
}

// ===========================================================================
// Event router
// ===========================================================================

function handleEvent(ev) {
  switch (ev.type) {
    case 'connected':
      setStatusLine('Connected — fetching data...');
      break;

    case 'phase_start':
      handlePhaseStart(ev);
      break;

    case 'phase_complete':
      handlePhaseComplete(ev);
      break;

    case 'progress':
      setStatusLine(ev.message || '');
      break;

    case 'agent_start':
      startAgent(ev.agent);
      break;

    case 'agent_tool_call':
      recordToolCall(ev.agent, ev.tool, ev.tool_input);
      break;

    case 'agent_complete':
      completeAgent(ev.agent);
      break;

    case 'theme_discovered':
      addLiveTheme(ev.data);
      break;

    case 'thinking_delta':
      appendThinking(ev.text || '');
      break;

    case 'agent_stream':
      // Orchestrator text — not displayed separately
      break;

    case 'brief_ready':
      briefData = ev.data;
      renderResults(briefData);
      break;

    case 'error':
      handlePipelineError(ev.error || 'Analysis failed');
      break;

    case 'done':
      showResults();
      eventSource.close();
      break;

    case 'keepalive':
      break;
  }
}

// ===========================================================================
// Phase / stepper
// ===========================================================================

const STEP_MAP = {
  fetching: 'step-fetching',
  wave1:    'step-theme',    // 02 Wave 1
  wave2:    'step-agents',   // 03 Wave 2
  synthesis: 'step-synthesis',
  pipeline: null,
  theme: null,   // legacy — handled via wave1
  agents: null,  // legacy
};

function handlePhaseStart(ev) {
  const stepId = STEP_MAP[ev.phase];
  if (stepId) setStep(stepId, 'active');
  setStatusLine(ev.message || `Phase: ${ev.phase}`);

  if (ev.phase === 'synthesis') {
    document.getElementById('orchestrator-panel').classList.remove('hidden');
    setStep('step-synthesis', 'active');
  }
}

function handlePhaseComplete(ev) {
  const stepId = STEP_MAP[ev.phase];
  if (stepId) setStep(stepId, 'done');
  if (ev.message) setStatusLine(ev.message);

  if (ev.data?.total) {
    setStatusLine(`Loaded ${ev.data.total} feedback items — Reddit: ${ev.data.reddit}, App Store: ${ev.data.app_store}, Seed: ${ev.data.seed}`);
  }
}

function setStep(id, state) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove('active', 'done');
  if (state) el.classList.add(state);
}

// ===========================================================================
// Agent cards
// ===========================================================================

// Per-agent call counters
const _agentCallCounts = {};

function startAgent(agent) {
  activeAgents.add(agent);
  _agentCallCounts[agent] = 0;
  const card = document.getElementById(`agent-${agent}`);
  const dot  = document.getElementById(`dot-${agent}`);
  if (card) card.classList.add('running');
  if (dot)  { dot.classList.remove('complete'); dot.classList.add('running'); }
  _setAgentStatus(agent, 'Working...');
}

function recordToolCall(agent, tool, input) {
  toolCallCount++;
  _agentCallCounts[agent] = (_agentCallCounts[agent] || 0) + 1;
  const count = _agentCallCounts[agent];

  // Derive a human-readable status from the tool name + first input value
  const TOOL_LABELS = {
    record_theme:        'Found theme',
    rate_ai_feasibility: 'Rated feasibility',
    record_gap:          'Identified gap',
    record_power_quote:  'Saved quote',
    write_brief_section: 'Writing section',
    rank_opportunities:  'Ranking opportunities',
    record_strength:     'Noted strength',
    finalize_brief:      'Finalising brief...',
    emit_theme_summary:  'Summarising themes',
    emit_feasibility_summary: 'Summarising feasibility',
    emit_gap_summary:    'Summarising gaps',
    emit_quote_summary:  'Summarising quotes',
  };

  let label = TOOL_LABELS[tool] || tool.replace(/_/g, ' ');

  // Append the first string value from input as context (e.g. theme name)
  if (input) {
    const firstVal = Object.values(input).find(v => typeof v === 'string' && v.length > 2);
    if (firstVal) label += `: ${firstVal.slice(0, 40)}`;
  }

  _setAgentStatus(agent, `${label} (${count})`);
}

function completeAgent(agent) {
  activeAgents.delete(agent);
  const count = _agentCallCounts[agent] || 0;
  const card  = document.getElementById(`agent-${agent}`);
  const dot   = document.getElementById(`dot-${agent}`);
  if (card) { card.classList.remove('running'); card.classList.add('complete'); }
  if (dot)  { dot.classList.remove('running'); dot.classList.add('complete'); }
  _setAgentStatus(agent, `Done — ${count} findings recorded`);
}

function _setAgentStatus(agent, text) {
  const el = document.getElementById(`status-${agent}`);
  if (el) el.textContent = text;
}

// ===========================================================================
// Live theme discovery wall
// ===========================================================================

function addLiveTheme(theme) {
  if (!theme || !theme.theme_name) return;

  const wall = document.getElementById('discovery-wall');
  const container = document.getElementById('theme-cards-live');
  if (!wall || !container) return;

  wall.classList.remove('hidden');

  const freq = theme.frequency_estimate || 'medium';
  const card = document.createElement('div');
  card.className = 'theme-card-live';
  card.innerHTML = `
    <div class="theme-card-live-name">${esc(theme.theme_name)}</div>
    <span class="theme-freq-badge ${freq}">${freq.replace('_', ' ')}</span>
  `;
  container.appendChild(card);
  container.scrollTop = container.scrollHeight;
}

// ===========================================================================
// Extended thinking
// ===========================================================================

function appendThinking(text) {
  thinkingBuffer += text;
  const el = document.getElementById('thinking-text');
  if (el) {
    el.textContent = thinkingBuffer;
    el.scrollTop = el.scrollHeight;
  }
}

function toggleThinking() {
  thinkingVisible = !thinkingVisible;
  const el = document.getElementById('thinking-text');
  const toggle = document.getElementById('thinking-toggle');
  if (el) el.style.display = thinkingVisible ? '' : 'none';
  if (toggle) toggle.textContent = thinkingVisible ? 'Hide' : 'Show';
}

// ===========================================================================
// Results rendering
// ===========================================================================

function renderResults(data) {
  // Update all dynamic company name elements
  const company = data.company_name || window._currentCompany || 'Company';
  window._currentCompany = company;
  setRunningCompanyName(company);

  // Update page/header titles
  const headerTitle = document.getElementById('header-title');
  if (headerTitle) headerTitle.textContent = `${company} · Feature Intelligence`;
  document.title = `${company} Feature Intelligence`;

  renderHeadline(data);
  renderOpportunities(data);
  renderStrengths(data);
  renderReportSections(data);
  renderEvidence(data);
}

function renderHeadline(data) {
  if (!data.headline_insight) return;
  const card = document.getElementById('headline-card');
  const text = document.getElementById('headline-text');
  card.classList.remove('hidden');
  text.textContent = data.headline_insight;
}

function renderOpportunities(data) {
  const list = document.getElementById('opportunities-list');

  // Add gap analysis disclaimer
  const existingDisclaimer = list.parentNode.querySelector('.gap-disclaimer');
  if (existingDisclaimer) existingDisclaimer.remove();
  const disclaimerEl = document.createElement('div');
  disclaimerEl.className = 'gap-disclaimer';
  const hasUserFeatures = window._currentPayload?.known_features?.trim().length > 0;
  disclaimerEl.innerHTML = hasUserFeatures
    ? '✓ Gap analysis uses your provided feature list as ground truth.'
    : '⚠ Gap analysis uses AI product knowledge (training cutoff may miss recent features). Paste current features in the form for higher accuracy.';
  list.parentNode.insertBefore(disclaimerEl, list);

  const opps = data.ranked_opportunities || [];
  list.innerHTML = opps.map(opp => `
    <div class="opp-card">
      <div class="opp-rank-col">
        <div class="opp-rank">${opp.rank}</div>
      </div>
      <div class="opp-body">
        <div class="opp-name">${esc(opp.feature_name)}</div>
        <div class="opp-liner">${esc(opp.one_liner)}</div>
        <div class="opp-badges">
          <span class="badge ${EVIDENCE_COLORS[opp.evidence_strength] || ''}">Evidence: ${opp.evidence_strength}</span>
          <span class="badge ${UNIQUENESS_COLORS[opp.ai_uniqueness] || ''}">AI fit: ${opp.ai_uniqueness}</span>
          <span class="badge badge-default">${EFFORT_LABELS[opp.effort_estimate] || opp.effort_estimate}</span>
        </div>
        ${opp.supporting_quote ? `<div class="opp-quote">${esc(opp.supporting_quote)}</div>` : ''}
      </div>
    </div>
  `).join('');
}

function renderStrengths(data) {
  // Pull strengths from sections if available
  const sections = data.sections || [];
  const strengthQuotes = [];
  sections.forEach(s => {
    (s.supporting_quotes || []).forEach(q => strengthQuotes.push(q));
  });

  // Try to get from brief data directly — the GapAgent output isn't directly in FeatureBrief
  // but the Orchestrator surfaces strengths in the executive_summary section quotes
  const execSection = sections.find(s => s.section_id === 'executive_summary');
  if (!execSection) return;

  // Only show if there's a top_3_insights
  const insights = data.top_3_insights || [];
  if (!insights.length) return;

  const section = document.getElementById('strengths-section');
  const list = document.getElementById('strengths-list');
  section.classList.remove('hidden');

  list.innerHTML = insights.map(ins => `
    <div class="strength-card">
      <div class="strength-check">✓</div>
      <div>
        <div class="strength-why">${esc(ins)}</div>
      </div>
    </div>
  `).join('');
}

function mdToHtml(text) {
  if (!text) return '';
  const lines = text.split('\n');
  const out = [];
  let inList = false;

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];

    // Headings
    if (/^### /.test(line)) { if (inList) { out.push('</ul>'); inList = false; } out.push(`<h4>${inlineEsc(line.slice(4))}</h4>`); continue; }
    if (/^## /.test(line))  { if (inList) { out.push('</ul>'); inList = false; } out.push(`<h3>${inlineEsc(line.slice(3))}</h3>`); continue; }
    if (/^# /.test(line))   { if (inList) { out.push('</ul>'); inList = false; } out.push(`<h2>${inlineEsc(line.slice(2))}</h2>`); continue; }

    // Horizontal rule
    if (/^---+$/.test(line.trim())) { if (inList) { out.push('</ul>'); inList = false; } out.push('<hr>'); continue; }

    // Bullet list
    if (/^[-*] /.test(line)) {
      if (!inList) { out.push('<ul>'); inList = true; }
      out.push(`<li>${inlineEsc(line.slice(2))}</li>`);
      continue;
    }

    // Numbered list
    if (/^\d+\. /.test(line)) {
      if (inList) { out.push('</ul>'); inList = false; }
      out.push(`<p>${inlineEsc(line)}</p>`);
      continue;
    }

    // Close list before blank line or normal paragraph
    if (inList) { out.push('</ul>'); inList = false; }

    // Blank line
    if (line.trim() === '') { out.push('<br>'); continue; }

    // Normal paragraph
    out.push(`<p>${inlineEsc(line)}</p>`);
  }

  if (inList) out.push('</ul>');
  return out.join('');
}

function inlineEsc(text) {
  // Escape HTML first, then convert inline markdown
  return esc(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,     '<em>$1</em>')
    .replace(/`(.+?)`/g,       '<code>$1</code>');
}

function renderReportSections(data) {
  const container = document.getElementById('report-sections');
  const sections = data.sections || [];
  container.innerHTML = sections.map((s, i) => `
    <div class="report-card">
      <div class="report-card-header" onclick="toggleSection(${i})">
        <span>${esc(s.title)}</span>
        <span class="report-card-chevron" id="chevron-${i}">›</span>
      </div>
      <div class="report-card-body" id="section-body-${i}">
        <div class="md-content">${mdToHtml(s.content)}</div>
        ${(s.supporting_quotes || []).map(q => `<blockquote>"${esc(q)}"</blockquote>`).join('')}
      </div>
    </div>
  `).join('');

  // Open first section
  if (sections.length) toggleSection(0);
}

function toggleSection(i) {
  const body = document.getElementById(`section-body-${i}`);
  const chev = document.getElementById(`chevron-${i}`);
  if (!body) return;
  const open = body.classList.toggle('open');
  if (chev) { chev.textContent = open ? '⌄' : '›'; chev.classList.toggle('open', open); }
}

function renderOpportunityMatrix(data) {
  const opps = data.ranked_opportunities || [];
  if (!opps.length || typeof d3 === 'undefined') return;

  const container = document.getElementById('opportunity-matrix');
  container.innerHTML = '';

  const margin = { top: 24, right: 40, bottom: 60, left: 60 };
  const totalW = container.clientWidth || 760;
  const w = totalW - margin.left - margin.right;
  const h = 280;

  // Ordinal positions
  const effortPos  = { quick_win: 1, medium_lift: 2, major_investment: 3 };
  const impactPos  = { emerging: 1, moderate: 2, strong: 3 };
  const sizeMap    = { high: 18, medium: 13, low: 9 };
  const colorMap   = { high: '#58cc02', medium: '#3b82f6', low: '#f59e0b' };
  const effortLabels = ['Quick win', 'Medium lift', 'Major investment'];
  const impactLabels = ['Emerging', 'Moderate', 'Strong'];

  const xScale = d3.scalePoint()
    .domain(['quick_win', 'medium_lift', 'major_investment'])
    .range([0, w])
    .padding(0.5);

  const yScale = d3.scalePoint()
    .domain(['emerging', 'moderate', 'strong'])
    .range([h, 0])
    .padding(0.4);

  const svg = d3.select(container)
    .append('svg')
    .attr('width', totalW)
    .attr('height', h + margin.top + margin.bottom);

  const g = svg.append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

  // Quadrant shading
  const midX = (xScale('quick_win') + xScale('medium_lift')) / 2;
  const midY = (yScale('emerging') + yScale('moderate')) / 2;
  g.append('rect').attr('x', 0).attr('y', 0).attr('width', midX).attr('height', midY)
    .attr('fill', '#f0fce7').attr('opacity', .5);
  g.append('text').attr('x', midX / 2).attr('y', midY / 2 - 6)
    .attr('class', 'matrix-quadrant-label').attr('text-anchor', 'middle').text('Prioritize');

  // Gridlines
  xScale.domain().forEach(d => {
    g.append('line').attr('class', 'matrix-gridline')
      .attr('x1', xScale(d)).attr('x2', xScale(d))
      .attr('y1', 0).attr('y2', h);
  });
  yScale.domain().forEach(d => {
    g.append('line').attr('class', 'matrix-gridline')
      .attr('x1', 0).attr('x2', w)
      .attr('y1', yScale(d)).attr('y2', yScale(d));
  });

  // Axes
  g.append('g').attr('class', 'matrix-axis').attr('transform', `translate(0,${h})`)
    .call(d3.axisBottom(xScale).tickFormat(d => effortLabels[effortPos[d] - 1]));
  g.append('g').attr('class', 'matrix-axis')
    .call(d3.axisLeft(yScale).tickFormat(d => impactLabels[impactPos[d] - 1]));

  // Axis labels
  g.append('text').attr('x', w / 2).attr('y', h + 50)
    .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', '#9ca3af')
    .attr('font-family', 'Inter, sans-serif').text('Implementation Effort →');
  g.append('text').attr('transform', 'rotate(-90)')
    .attr('x', -h / 2).attr('y', -46)
    .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', '#9ca3af')
    .attr('font-family', 'Inter, sans-serif').text('↑ Evidence Strength');

  // Bubbles + labels
  opps.forEach(opp => {
    const cx = xScale(opp.effort_estimate) || w / 2;
    const cy = yScale(opp.evidence_strength) || h / 2;
    const r  = sizeMap[opp.ai_uniqueness] || 12;
    const col = colorMap[opp.ai_uniqueness] || '#6b7280';

    const node = g.append('g').attr('transform', `translate(${cx},${cy})`).style('cursor', 'default');

    node.append('title').text(
      `#${opp.rank} ${opp.feature_name}\n${opp.one_liner}\nAI fit: ${opp.ai_uniqueness}`
    );
    node.append('circle').attr('r', r).attr('fill', col).attr('opacity', .85);
    node.append('text').attr('y', r + 11)
      .attr('text-anchor', 'middle').attr('font-size', 9.5).attr('fill', '#374151')
      .attr('font-family', 'Inter, sans-serif').attr('font-weight', 600)
      .text(opp.feature_name.length > 22 ? opp.feature_name.slice(0, 20) + '…' : opp.feature_name);
  });

  // Legend
  const legend = svg.append('g')
    .attr('transform', `translate(${margin.left + w - 120}, ${margin.top})`);
  [['high','#58cc02','High AI fit'],['medium','#3b82f6','Medium AI fit'],['low','#f59e0b','Low AI fit']].forEach(([k,c,label], i) => {
    const row = legend.append('g').attr('transform', `translate(0, ${i * 18})`);
    row.append('circle').attr('r', 5).attr('cy', 0).attr('fill', c).attr('opacity', .85);
    row.append('text').attr('x', 10).attr('y', 4)
      .attr('font-size', 10).attr('fill', '#6b7280').attr('font-family', 'Inter, sans-serif').text(label);
  });
}

function renderEvidence(data) {
  // Source chips
  const sourceRow = document.getElementById('source-row');
  const r  = data.reddit_count      || 0;
  const a  = data.app_store_count   || 0;
  const gp = data.google_play_count || 0;
  const hn = data.hacker_news_count || 0;
  const tw = data.twitter_count     || 0;
  const s  = data.seed_count        || 0;
  const chips = [
    r  ? `<span class="source-chip">Reddit — ${r}</span>` : '',
    a  ? `<span class="source-chip">App Store — ${a}</span>` : '',
    gp ? `<span class="source-chip">Google Play — ${gp}</span>` : '',
    hn ? `<span class="source-chip">Hacker News — ${hn}</span>` : '',
    tw ? `<span class="source-chip">Twitter/X — ${tw}</span>` : '',
    s  ? `<span class="source-chip">Seed — ${s}</span>` : '',
    `<span class="source-chip" style="background:#f0fdf4;border-color:#bbf7d0;color:#15803d">Total: ${r+a+gp+hn+tw+s}</span>`,
  ].filter(Boolean).join('');
  sourceRow.innerHTML = chips;

  // D3 opportunity matrix
  renderOpportunityMatrix(data);

  // Real theme frequency chart from ThemeAgent output
  renderThemeChart(data);

  // Real power quotes from QuoteAgent
  renderQuoteGrid(data);
}

function renderThemeChart(data) {
  const themes = data.themes || [];
  if (!themes.length) return;

  const freqPct = { very_high: 95, high: 72, medium: 48, low: 24 };
  const freqLabel = { very_high: 'Very high', high: 'High', medium: 'Medium', low: 'Low' };

  // Sort: very_high → high → medium → low
  const FREQ_ORDER = { very_high: 4, high: 3, medium: 2, low: 1 };
  const sorted = [...themes].sort(
    (a, b) => (FREQ_ORDER[b.frequency_estimate] || 0) - (FREQ_ORDER[a.frequency_estimate] || 0)
  );

  const chart = document.getElementById('theme-chart');
  chart.innerHTML = sorted.map(t => `
    <div class="bar-row">
      <div class="bar-label" title="${esc(t.theme_name)}">${esc(t.theme_name)}</div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${freqPct[t.frequency_estimate] || 40}%"></div>
      </div>
      <div class="bar-freq">${freqLabel[t.frequency_estimate] || t.frequency_estimate}</div>
    </div>
  `).join('');
}

function renderQuoteGrid(data) {
  // Use real QuoteAgent power_quotes — curated, compelling, source-attributed
  const quotes = data.power_quotes || [];

  const grid = document.getElementById('quote-grid');
  if (!quotes.length) {
    grid.innerHTML = '<p style="color:var(--muted);font-size:13px">No quotes available.</p>';
    return;
  }

  grid.innerHTML = quotes.map(q => `
    <div class="quote-card">
      <span class="quote-open-mark">"</span>
      <div class="quote-text">${esc(q.quote_text)}</div>
      <div class="quote-meta">
        <span class="quote-source-badge ${esc(q.source)}">${esc(q.source.replace('_', ' '))}</span>
        <span class="quote-theme">${esc(q.theme)}</span>
      </div>
    </div>
  `).join('');
}

// ===========================================================================
// Tabs
// ===========================================================================

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${tab}`));
}

// ===========================================================================
// Status helpers
// ===========================================================================

function setStatus(cls, text) {
  const badge = document.getElementById('status-badge');
  badge.className = `status-badge ${cls}`;
  badge.textContent = text;
}

function setStatusLine(msg) {
  const el = document.getElementById('status-line');
  if (el) el.textContent = msg;
}

// ===========================================================================
// Download
// ===========================================================================

async function downloadMarkdown() {
  if (!currentRunId) return;
  try {
    const res = await fetch(`/brief/${currentRunId}/markdown`);
    if (!res.ok) throw new Error('Not ready');
    const data = await res.json();
    const blob = new Blob([data.markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'duolingo_feature_brief.md';
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    alert('Brief not ready yet.');
  }
}

// ===========================================================================
// Misc helpers
// ===========================================================================

function esc(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

const AGENT_DEFAULT_STATUS = {
  ThemeAgent:          'Clusters feedback into themes',
  QuoteAgent:          'Mines compelling user evidence',
  AIFeasibilityAgent:  'Rates AI solvability per theme',
  GapAgent:            'Maps feature gaps & opportunities',
};

function resetRunningUI() {
  ['ThemeAgent', 'AIFeasibilityAgent', 'GapAgent', 'QuoteAgent'].forEach(a => {
    const card = document.getElementById(`agent-${a}`);
    const dot  = document.getElementById(`dot-${a}`);
    if (card) card.classList.remove('running', 'complete');
    if (dot)  dot.classList.remove('running', 'complete');
    _setAgentStatus(a, AGENT_DEFAULT_STATUS[a] || 'Waiting...');
    _agentCallCounts[a] = 0;
  });

  ['step-fetching', 'step-theme', 'step-agents', 'step-synthesis'].forEach(id => {
    setStep(id, null);
  });

  document.getElementById('orchestrator-panel').classList.add('hidden');
  document.getElementById('thinking-text').textContent = '';
  thinkingBuffer = '';

  const wall = document.getElementById('discovery-wall');
  if (wall) wall.classList.add('hidden');
  const liveCards = document.getElementById('theme-cards-live');
  if (liveCards) liveCards.innerHTML = '';

  const timer = document.getElementById('run-timer');
  if (timer) { timer.classList.add('hidden'); timer.textContent = '0:00'; }

  setStatusLine('Connecting...');
  document.getElementById('download-btn').classList.add('hidden');
}

// ===========================================================================
// Share link
// ===========================================================================

function copyShareLink() {
  if (!currentRunId) return;
  const url = `${window.location.origin}${window.location.pathname}?share=${currentRunId}`;
  navigator.clipboard.writeText(url).then(() => {
    const toast = document.getElementById('share-toast');
    if (toast) {
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 2500);
    }
  });
}

// Handle ?share= param — load a previously generated brief directly
(function() {
  const params = new URLSearchParams(window.location.search);
  const shareId = params.get('share');
  if (shareId) {
    currentRunId = shareId;
    fetch(`/brief/${shareId}`)
      .then(r => { if (!r.ok) throw new Error('Brief not found'); return r.json(); })
      .then(data => {
        briefData = data;
        showResults();
        renderResults(data);
      })
      .catch(() => {
        // Brief not found — show landing normally
      });
  }
})();
