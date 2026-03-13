/* HellaCash Dashboard — app.js */
'use strict';

const API = '';  // same origin
let currentPair = 'BTC-EUR';
let currentInterval = '5m';
let chart = null;
let candleSeries = null;
let equityChart = null;
let ws = null;

// ── Lightweight Charts setup ──────────────────────────────────────────────────

function initPriceChart() {
  const el = document.getElementById('price-chart');
  el.innerHTML = '';
  chart = LightweightCharts.createChart(el, {
    width: el.clientWidth,
    height: 320,
    layout: { background: { color: '#1a1d27' }, textColor: '#94a3b8' },
    grid: { vertLines: { color: '#1e2235' }, horzLines: { color: '#1e2235' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#2d3148' },
    timeScale: { borderColor: '#2d3148', timeVisible: true },
  });
  candleSeries = chart.addCandlestickSeries({
    upColor: '#10b981', downColor: '#ef4444',
    borderUpColor: '#10b981', borderDownColor: '#ef4444',
    wickUpColor: '#10b981', wickDownColor: '#ef4444',
  });
  window.addEventListener('resize', () => chart.applyOptions({ width: el.clientWidth }));
}

async function loadCandles() {
  try {
    const res = await fetch(`${API}/api/candles/${currentPair}?interval=${currentInterval}&limit=200`);
    const data = await res.json();
    const candles = data.map(c => ({
      time: Math.floor(new Date(c.timestamp).getTime() / 1000),
      open: c.open, high: c.high, low: c.low, close: c.close,
    }));
    candleSeries.setData(candles);
  } catch (e) { console.error('loadCandles', e); }
}

function updateLastCandle(ticker) {
  if (ticker.symbol !== currentPair || !candleSeries) return;
  // Append a tick as the current candle's close update
}

// ── Equity curve chart (Chart.js) ─────────────────────────────────────────────

function initEquityChart() {
  const ctx = document.getElementById('equity-chart').getContext('2d');
  equityChart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Equity (€)', data: [], borderColor: '#3b82f6', fill: true, backgroundColor: 'rgba(59,130,246,0.08)', tension: 0.3, pointRadius: 0 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { grid: { color: '#1e2235' }, ticks: { color: '#64748b', font: { size: 11 } } },
      },
    },
  });
}

async function loadEquityHistory() {
  try {
    const res = await fetch(`${API}/api/portfolio/history?days=7`);
    const data = await res.json();
    if (!data.length) return;
    equityChart.data.labels = data.map(d => d.snapshot_at ? d.snapshot_at.slice(0, 16).replace('T', ' ') : '');
    equityChart.data.datasets[0].data = data.map(d => d.total_equity_eur);
    equityChart.update('none');
  } catch (e) { console.error('loadEquityHistory', e); }
}

// ── Portfolio & Analytics ─────────────────────────────────────────────────────

async function refreshPortfolio() {
  try {
    const res = await fetch(`${API}/api/portfolio`);
    const d = await res.json();
    const eq = d.equity_eur || 0;
    setText('equity', `€${fmt(eq)}`);
    setText('hdr-equity', `€${fmt(eq)}`);
    const dd = d.drawdown_pct || 0;
    document.getElementById('drawdown').innerHTML =
      `<span class="${dd > 3 ? 'pnl-neg' : 'text-green-400'}">${dd.toFixed(2)}%</span>`;
    setText('pos-count', (d.open_positions || []).length);
    renderPositions(d.open_positions || []);
  } catch (e) {}
}

async function refreshAnalytics() {
  try {
    const res = await fetch(`${API}/api/analytics?days=30`);
    const d = await res.json();
    setValueColored('a-pnl', d.total_pnl, `€${fmt(d.total_pnl)}`);
    setText('a-wr', d.win_rate != null ? `${(d.win_rate * 100).toFixed(1)}%` : '—');
    setText('hdr-winrate', d.win_rate != null ? `${(d.win_rate * 100).toFixed(1)}%` : '—');
    setText('a-sharpe', d.sharpe_ratio != null ? d.sharpe_ratio.toFixed(2) : '—');
    setText('a-trades', d.total_trades || 0);
    setValueColored('hdr-daily-pnl', d.total_pnl, `€${fmt(d.total_pnl)}`);
  } catch (e) {}
}

async function refreshTrades() {
  try {
    const res = await fetch(`${API}/api/trades?limit=50`);
    const trades = await res.json();
    renderTrades(trades);
  } catch (e) {}
}

async function refreshSentiment() {
  const pairs = ['BTC-EUR', 'ETH-EUR', 'SOL-EUR'];
  const el = document.getElementById('sentiment-list');
  const rows = await Promise.all(pairs.map(async p => {
    try {
      const r = await fetch(`${API}/api/sentiment/${p}`);
      const d = await r.json();
      return { symbol: p.split('-')[0], score: d.current_score || 0 };
    } catch { return { symbol: p.split('-')[0], score: 0 }; }
  }));
  el.innerHTML = rows.map(r => {
    const pct = Math.round(((r.score + 1) / 2) * 100);
    const color = r.score > 0.1 ? '#10b981' : r.score < -0.1 ? '#ef4444' : '#f59e0b';
    return `<div>
      <div class="flex justify-between text-xs mb-1"><span class="text-gray-400">${r.symbol}</span><span style="color:${color}">${r.score.toFixed(3)}</span></div>
      <div class="w-full bg-gray-800 rounded" style="height:4px"><div class="rounded" style="width:${pct}%;height:4px;background:${color}"></div></div>
    </div>`;
  }).join('');
}

async function refreshStatus() {
  try {
    const res = await fetch(`${API}/api/bot/status`);
    const d = await res.json();
    const modeBadge = document.getElementById('mode-badge');
    if (d.paper_trading) {
      modeBadge.textContent = 'PAPER'; modeBadge.className = 'text-xs px-2 py-1 rounded font-medium badge-paper';
    } else {
      modeBadge.textContent = 'LIVE'; modeBadge.className = 'text-xs px-2 py-1 rounded font-medium badge-live';
    }
    // Populate pair selector
    if (d.active_symbols && d.active_symbols.length) {
      const sel = document.getElementById('pair-select');
      const current = sel.value;
      sel.innerHTML = d.active_symbols.map(s => `<option value="${s}"${s===current?' selected':''}>${s}</option>`).join('');
    }
  } catch (e) {}
}

// ── Render helpers ────────────────────────────────────────────────────────────

function renderPositions(positions) {
  const el = document.getElementById('positions-list');
  if (!positions.length) { el.innerHTML = '<div class="text-xs text-gray-600 italic">No positions</div>'; return; }
  el.innerHTML = positions.map(p => {
    const pnl = p.unrealized_pnl || 0;
    return `<div class="flex justify-between items-center py-1 border-b border-gray-800">
      <div><div class="text-white font-medium">${p.symbol}</div><div class="text-xs text-gray-500">${p.strategy_name}</div></div>
      <div class="text-right"><div class="${pnl>=0?'pnl-pos':'pnl-neg'} font-medium">€${fmt(pnl)}</div><div class="text-xs text-gray-500">${p.quantity?.toFixed(4)||'—'}</div></div>
    </div>`;
  }).join('');
}

function renderTrades(trades) {
  const tbody = document.getElementById('trades-body');
  if (!trades.length) { tbody.innerHTML = '<tr><td colspan="10" class="text-gray-600 text-center py-6">No trades yet</td></tr>'; return; }
  tbody.innerHTML = trades.map(t => {
    const pnlClass = t.net_pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
    const hold = fmtDuration(t.hold_seconds);
    const dt = t.created_at ? t.created_at.slice(0, 16).replace('T', ' ') : '';
    return `<tr>
      <td class="font-medium text-white">${t.symbol}</td>
      <td class="text-gray-400">${t.strategy_name}</td>
      <td>€${fmt(t.entry_price)}</td>
      <td>€${fmt(t.exit_price)}</td>
      <td>${t.quantity?.toFixed(4)||'—'}</td>
      <td class="${pnlClass} font-medium">€${fmt(t.net_pnl)}</td>
      <td class="${pnlClass}">${t.roi_pct?.toFixed(2)||'—'}%</td>
      <td class="text-gray-400">${hold}</td>
      <td><span class="text-xs px-2 py-0.5 rounded bg-gray-800 text-gray-300">${t.exit_reason}</span></td>
      <td><span class="text-xs ${t.paper_trade?'text-yellow-500':'text-green-500'}">${t.paper_trade?'paper':'live'}</span></td>
    </tr>`;
  }).join('');
}

function addSignalFeedItem(data) {
  const el = document.getElementById('signal-feed');
  const dir = data.direction || data.type || 'NEUTRAL';
  const cls = dir === 'LONG' ? 'signal-long' : dir === 'SHORT' ? 'signal-short' : 'signal-neutral';
  const now = new Date().toLocaleTimeString();
  const item = document.createElement('div');
  item.className = 'flex gap-2 items-center';
  item.innerHTML = `<span class="text-gray-600">${now}</span><span class="${cls} font-medium">${dir}</span><span class="text-gray-400">${data.symbol||data.type||''}</span><span class="text-gray-500">${data.strategy_name||''}</span><span class="text-gray-600">${data.strength ? (data.strength*100).toFixed(0)+'%' : ''}</span>`;
  el.prepend(item);
  while (el.children.length > 30) el.removeChild(el.lastChild);
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/feed`);

  ws.onmessage = e => {
    try {
      const msg = JSON.parse(e.data);
      const type = msg.type;
      const d = msg.data || {};

      if (type === 'market.ticker') updateLastCandle(d);
      if (type === 'strategy.signal') addSignalFeedItem(d);
      if (type === 'trade.opened') { addSignalFeedItem({...d, direction:'LONG', symbol:d.symbol}); refreshTrades(); }
      if (type === 'trade.closed') { refreshTrades(); refreshPortfolio(); }
      if (type === 'portfolio.update') refreshPortfolio();
      if (type === 'sentiment.updated') refreshSentiment();
      if (type === 'risk.halt') {
        document.getElementById('halt-badge').classList.remove('hidden');
        addSignalFeedItem({type:'HALT', symbol:'', strategy_name: d.reason||'Circuit breaker'});
      }
    } catch {}
  };

  ws.onclose = () => setTimeout(connectWS, 3000);
  ws.onerror = () => ws.close();

  // keep-alive ping
  setInterval(() => { if (ws.readyState === 1) ws.send('ping'); }, 30000);
}

// ── Bot controls ──────────────────────────────────────────────────────────────

async function botStart() {
  await fetch(`${API}/api/bot/start`, { method: 'POST' });
  refreshStatus();
}
async function botStop() {
  await fetch(`${API}/api/bot/stop`, { method: 'POST' });
  refreshStatus();
}

function switchPair(pair) {
  currentPair = pair;
  loadCandles();
}
function switchInterval(iv) {
  currentInterval = iv;
  document.querySelectorAll('.tf-btn').forEach(b => {
    b.className = 'tf-btn px-3 py-1 text-xs rounded ' + (b.textContent.trim() === iv ? 'bg-blue-700 text-white' : 'bg-gray-800 text-gray-400 hover:text-white');
  });
  loadCandles();
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function fmt(n) {
  if (n == null) return '—';
  return Number(n).toLocaleString('nl-NL', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
function fmtDuration(secs) {
  if (!secs) return '—';
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.round(secs/60)}m`;
  return `${Math.round(secs/3600)}h`;
}
function setText(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }
function setValueColored(id, num, text) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.className = num >= 0 ? 'pnl-pos font-bold' : 'pnl-neg font-bold';
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────

async function init() {
  initPriceChart();
  initEquityChart();
  connectWS();

  await Promise.all([
    refreshStatus(), refreshPortfolio(), refreshAnalytics(),
    loadCandles(), loadEquityHistory(), refreshTrades(), refreshSentiment(),
  ]);

  // Periodic refresh every 30s
  setInterval(async () => {
    await Promise.all([
      refreshPortfolio(), refreshAnalytics(), loadEquityHistory(),
    ]);
  }, 30000);

  setInterval(refreshTrades, 60000);
  setInterval(refreshSentiment, 120000);
}

document.addEventListener('DOMContentLoaded', init);
