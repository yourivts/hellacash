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

function pricePrecision(price) {
  if (!price || price <= 0) return { precision: 2, minMove: 0.01 };
  if (price < 0.01) return { precision: 6, minMove: 0.000001 };
  if (price < 1) return { precision: 4, minMove: 0.0001 };
  return { precision: 2, minMove: 0.01 };
}

async function loadCandles() {
  try {
    const res = await fetch(`${API}/api/candles/${currentPair}?interval=${currentInterval}&limit=200`);
    const data = await res.json();
    const candles = data.map(c => ({
      time: Math.floor(new Date(c.timestamp).getTime() / 1000),
      open: c.open, high: c.high, low: c.low, close: c.close,
    })).sort((a, b) => a.time - b.time);

    // Determine price precision from the last candle's close price
    const lastClose = candles.length ? candles[candles.length - 1].close : 0;
    const pf = pricePrecision(lastClose);

    // Remove old series and create fresh one to avoid stale data conflicts
    chart.removeSeries(candleSeries);
    candleSeries = chart.addCandlestickSeries({
      upColor: '#10b981', downColor: '#ef4444',
      borderUpColor: '#10b981', borderDownColor: '#ef4444',
      wickUpColor: '#10b981', wickDownColor: '#ef4444',
      priceFormat: { type: 'price', precision: pf.precision, minMove: pf.minMove },
    });
    candleSeries.setData(candles);
  } catch (e) { console.error('loadCandles', e); }
}

let lastCandle = null;
let prevPrice = null;
let priceLine = null;
let pollPaused = false;

function updateLastCandle(ticker) {
  if (ticker.symbol !== currentPair || !candleSeries) return;
}

async function pollPrice() {
  if (!candleSeries || pollPaused) return;
  try {
    const res = await fetch(`${API}/api/candles/ticker/${currentPair}?_=${Date.now()}`, { cache: 'no-store' });
    const data = await res.json();
    // Use orderbook mid-price for real-time accuracy (last trade price updates slowly)
    const bestBid = data.bids && data.bids.length ? parseFloat(data.bids[0][0]) : 0;
    const bestAsk = data.asks && data.asks.length ? parseFloat(data.asks[0][0]) : 0;
    const price = (bestBid && bestAsk) ? (bestBid + bestAsk) / 2 : parseFloat(data.price);
    if (!price) return;

    // Update live price display
    const priceEl = document.getElementById('live-price');
    const deltaEl = document.getElementById('live-price-delta');
    if (priceEl) priceEl.textContent = `€${fmt(price)}`;
    if (deltaEl && prevPrice) {
      const diff = price - prevPrice;
      if (diff !== 0) {
        const sign = diff > 0 ? '+' : '';
        const pf = pricePrecision(price);
        deltaEl.textContent = `${sign}${diff.toFixed(pf.precision)}`;
        deltaEl.className = `text-xs font-medium ${diff > 0 ? 'pnl-pos' : 'pnl-neg'}`;
      }
    }
    prevPrice = price;

    // Update candle
    const now = Math.floor(Date.now() / 1000);
    const intervalSecs = { '1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400, '1d': 86400 };
    const bucket = intervalSecs[currentInterval] || 300;
    const candleTime = Math.floor(now / bucket) * bucket;

    if (lastCandle && lastCandle.time === candleTime) {
      lastCandle.close = price;
      lastCandle.high = Math.max(lastCandle.high, price);
      lastCandle.low = Math.min(lastCandle.low, price);
    } else {
      lastCandle = { time: candleTime, open: price, high: price, low: price, close: price };
    }
    candleSeries.update(lastCandle);

    // Update price tracking line on chart
    if (priceLine) candleSeries.removePriceLine(priceLine);
    priceLine = candleSeries.createPriceLine({
      price: price,
      color: '#3b82f6',
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dotted,
      axisLabelVisible: true,
      title: '',
    });

    // Render orderbook
    renderOrderbook(data.bids || [], data.asks || []);
  } catch (e) { /* silent — runs every 500ms */ }
}

function renderOrderbook(bids, asks) {
  const el = document.getElementById('orderbook');
  if (!el) return;

  const maxBidVol = Math.max(...bids.map(b => parseFloat(b[1])), 0.0001);
  const maxAskVol = Math.max(...asks.map(a => parseFloat(a[1])), 0.0001);

  const bidHtml = bids.slice(0, 10).map(b => {
    const price = parseFloat(b[0]);
    const vol = parseFloat(b[1]);
    const pct = (vol / maxBidVol) * 100;
    return `<div class="ob-row ob-bid">
      <div class="ob-bg" style="width:${pct}%"></div>
      <span class="text-green-400 relative">${fmt(price)}</span>
      <span class="text-gray-400 relative">${vol.toFixed(4)}</span>
    </div>`;
  }).join('');

  const askHtml = asks.slice(0, 10).map(a => {
    const price = parseFloat(a[0]);
    const vol = parseFloat(a[1]);
    const pct = (vol / maxAskVol) * 100;
    return `<div class="ob-row ob-ask">
      <div class="ob-bg" style="width:${pct}%"></div>
      <span class="text-red-400 relative">${fmt(price)}</span>
      <span class="text-gray-400 relative">${vol.toFixed(4)}</span>
    </div>`;
  }).join('');

  el.innerHTML = `<div>${bidHtml}</div><div>${askHtml}</div>`;

  // Spread
  if (bids.length && asks.length) {
    const bestBid = parseFloat(bids[0][0]);
    const bestAsk = parseFloat(asks[0][0]);
    const spread = bestAsk - bestBid;
    const spreadPct = ((spread / bestAsk) * 100).toFixed(3);
    setText('ob-spread', `Spread: €${fmt(spread)} (${spreadPct}%)`);
  }
}

// ── Equity curve chart (Chart.js) ─────────────────────────────────────────────

function initEquityChart() {
  const ctx = document.getElementById('equity-chart').getContext('2d');
  equityChart = new Chart(ctx, {
    type: 'line',
    data: { datasets: [{ label: 'Equity', data: [], borderColor: '#3b82f6', fill: true, backgroundColor: 'rgba(59,130,246,0.08)', tension: 0.3, pointRadius: 0 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          type: 'time',
          time: { unit: 'hour', displayFormats: { hour: 'MMM d HH:mm', day: 'MMM d' }, tooltipFormat: 'MMM d HH:mm' },
          grid: { color: '#1e2235' },
          ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 6 },
        },
        y: { grid: { color: '#1e2235' }, ticks: { color: '#64748b', font: { size: 11 } } },
      },
    },
  });
}

async function loadEquityHistory() {
  try {
    const res = await fetch(`${API}/api/portfolio/equity-history?days=7`);
    const data = await res.json();
    if (!data.length) {
      equityChart.data.datasets[0].data = [{ x: new Date(), y: 0 }];
      equityChart.update('none');
      return;
    }
    equityChart.data.datasets[0].data = data.map(d => ({
      x: new Date(d.timestamp),
      y: d.equity,
    }));
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
    const dailyPnl = d.daily_pnl || 0;
    setValueColored('hdr-daily-pnl', dailyPnl, `€${fmt(dailyPnl)}`);
  } catch (e) {}
}

async function refreshTrades() {
  try {
    const res = await fetch(`${API}/api/trades?limit=50`);
    const trades = await res.json();
    renderTrades(trades);
  } catch (e) {}
}

let allSentiment = [];

async function refreshSentiment() {
  try {
    const res = await fetch(`${API}/api/sentiment/all/scores`);
    allSentiment = await res.json();
  } catch (err) { console.error('sentiment fetch', err); return; }
  setText('sentiment-count', allSentiment.length);
  const query = (document.getElementById('sentiment-search')?.value || '').toLowerCase();
  renderSentimentList(query);
}

function filterSentiment(query) {
  renderSentimentList(query.toLowerCase());
}

function renderSentimentList(query) {
  const el = document.getElementById('sentiment-list');
  let rows = allSentiment;
  if (query) {
    rows = rows.filter(r => r.base.toLowerCase().includes(query) || r.symbol.toLowerCase().includes(query));
  }
  if (!rows.length) {
    el.innerHTML = '<div class="text-xs text-gray-600 italic">No results</div>';
    return;
  }
  el.innerHTML = rows.map(r => {
    const pct = Math.round(((r.score + 1) / 2) * 100);
    const color = r.score > 0.1 ? '#10b981' : r.score < -0.1 ? '#ef4444' : '#f59e0b';
    const label = r.score > 0.1 ? 'Bullish' : r.score < -0.1 ? 'Bearish' : 'Neutral';
    const isActive = r.symbol === currentPair;
    const highlight = isActive ? ' text-white font-medium' : ' text-gray-400';
    return `<div class="cursor-pointer hover:bg-gray-800 rounded px-1 py-0.5" onclick="switchPair('${r.symbol}')">
      <div class="flex justify-between text-xs mb-0.5"><span class="${highlight}">${r.base}</span><span style="color:${color}">${r.score.toFixed(3)} ${label}</span></div>
      <div class="w-full bg-gray-800 rounded" style="height:3px"><div class="rounded" style="width:${pct}%;height:3px;background:${color}"></div></div>
    </div>`;
  }).join('');
}

let allMarkets = [];

async function loadMarkets() {
  try {
    const res = await fetch(`${API}/api/candles/markets`);
    allMarkets = await res.json();
    if (!allMarkets.length) return;
    populateMarketSelect(allMarkets);
  } catch (e) { console.error('loadMarkets', e); }
}

function populateMarketSelect(markets) {
  const sel = document.getElementById('pair-select');
  const current = sel.value || currentPair;
  sel.innerHTML = markets.map(m =>
    `<option value="${m.symbol}"${m.symbol === current ? ' selected' : ''}>${m.base} — €${fmt(m.price)}</option>`
  ).join('');
}

function filterMarkets(query) {
  const q = query.toLowerCase().trim();
  if (!q) { populateMarketSelect(allMarkets); return; }
  const filtered = allMarkets.filter(m => m.base.toLowerCase().includes(q) || m.symbol.toLowerCase().includes(q));
  populateMarketSelect(filtered);
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
    updateToggleButton(d.running);
    updatePaperResetVisibility(d.paper_trading);

    // Candle loading status
    const candleEl = document.getElementById('candle-status');
    const candleText = document.getElementById('candle-status-text');
    if (d.candles_ready) {
      candleEl.style.background = '#10b98122';
      candleEl.style.color = '#10b981';
      candleText.textContent = 'Candles ready';
      setTimeout(() => candleEl.style.opacity = '0.6', 3000);
    } else if (d.candles_progress) {
      const p = d.candles_progress;
      const pct = p.total > 0 ? Math.round(p.loaded / p.total * 100) : 0;
      candleText.textContent = `Loading candles… ${p.loaded}/${p.total} (${pct}%)`;
    }

    // Rate limit status
    if (d.rate_limit) {
      const rl = d.rate_limit;
      const rlText = document.getElementById('rate-limit-text');
      const rlBadge = document.getElementById('rate-limit-badge');
      if (rlText && rlBadge) {
        rlText.textContent = `${rl.remaining}/${rl.limit}`;
        if (rl.remaining < 100) {
          rlBadge.style.background = '#ef444422';
          rlBadge.style.color = '#ef4444';
        } else if (rl.remaining < 300) {
          rlBadge.style.background = '#f59e0b22';
          rlBadge.style.color = '#f59e0b';
        } else {
          rlBadge.style.background = '#10b98122';
          rlBadge.style.color = '#10b981';
        }
      }
    }
  } catch (e) {}
}

// ── Render helpers ────────────────────────────────────────────────────────────

let openPositions = [];

function renderPositions(positions) {
  openPositions = positions;
  _drawPositions();
}

function _drawPositions() {
  const el = document.getElementById('positions-list');
  if (!openPositions.length) { el.innerHTML = '<div class="text-xs text-gray-600 italic">No positions</div>'; return; }
  el.innerHTML = openPositions.map(p => {
    const pnl = p.unrealized_pnl || 0;
    const price = p.current_price || 0;
    const value = price * (p.quantity || 0);
    const entry = p.entry_price || 0;
    const roiPct = entry > 0 ? ((price - entry) / entry * 100) : 0;
    return `<div class="flex justify-between items-center py-1 border-b border-gray-800">
      <div>
        <div class="text-white font-medium">${p.symbol}</div>
        <div class="text-xs text-gray-500">${p.strategy_name}</div>
        <div class="text-xs text-gray-600">Entry €${fmt(entry)}</div>
      </div>
      <div class="text-right">
        <div class="${pnl>=0?'pnl-pos':'pnl-neg'} font-medium">€${fmt(pnl)} <span class="text-xs">(${roiPct>=0?'+':''}${roiPct.toFixed(2)}%)</span></div>
        <div class="text-xs text-gray-400">€${fmt(value)}</div>
        <div class="text-xs text-gray-500">${p.quantity?.toFixed(4)||'—'} @ €${fmt(price)}</div>
      </div>
    </div>`;
  }).join('');
}

function updatePositionPrices(symbol, price) {
  let changed = false;
  for (const p of openPositions) {
    if (p.symbol === symbol) {
      p.current_price = price;
      const entry = p.entry_price || 0;
      const qty = p.quantity || 0;
      const dir = p.direction || 'LONG';
      p.unrealized_pnl = dir === 'SHORT' ? (entry - price) * qty : (price - entry) * qty;
      changed = true;
    }
  }
  if (changed) _drawPositions();
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

      if (type === 'market.ticker') { updateLastCandle(d); if (d.symbol && d.price) updatePositionPrices(d.symbol, d.price); }
      if (type === 'strategy.signal') addSignalFeedItem(d);
      if (type === 'trade.opened') { addSignalFeedItem({...d, direction:'LONG', symbol:d.symbol}); refreshTrades(); }
      if (type === 'trade.closed') { refreshTrades(); refreshPortfolio(); }
      if (type === 'portfolio.update') { refreshPortfolio(); refreshAnalytics(); }
      if (type === 'sentiment.updated') { refreshSentiment(); refreshAnalytics(); }
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

let botRunning = false;

async function botToggle() {
  const btn = document.getElementById('btn-toggle');
  btn.disabled = true;
  if (botRunning) {
    await fetch(`${API}/api/bot/stop`, { method: 'POST' });
  } else {
    await fetch(`${API}/api/bot/start`, { method: 'POST' });
  }
  await refreshStatus();
  btn.disabled = false;
}

function updateToggleButton(running) {
  botRunning = running;
  const btn = document.getElementById('btn-toggle');
  if (running) {
    btn.textContent = 'Stop';
    btn.className = 'px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg text-sm font-medium transition';
  } else {
    btn.textContent = 'Start';
    btn.className = 'px-4 py-2 bg-green-600 hover:bg-green-500 text-white rounded-lg text-sm font-medium transition';
  }
}

async function switchPair(pair) {
  pollPaused = true;
  currentPair = pair;
  lastCandle = null;
  prevPrice = null;
  if (priceLine) { try { candleSeries.removePriceLine(priceLine); } catch(e) {} priceLine = null; }
  await loadCandles();
  pollPaused = false;
  refreshSentiment();
}
async function switchInterval(iv) {
  pollPaused = true;
  currentInterval = iv;
  lastCandle = null;
  prevPrice = null;
  if (priceLine) { try { candleSeries.removePriceLine(priceLine); } catch(e) {} priceLine = null; }
  document.querySelectorAll('.tf-btn').forEach(b => {
    b.className = 'tf-btn px-3 py-1 text-xs rounded ' + (b.textContent.trim() === iv ? 'bg-blue-700 text-white' : 'bg-gray-800 text-gray-400 hover:text-white');
  });
  await loadCandles();
  pollPaused = false;
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function fmt(n) {
  if (n == null) return '—';
  const v = Math.abs(Number(n));
  let decimals = 2;
  if (v > 0 && v < 0.01) decimals = 6;
  else if (v >= 0.01 && v < 1) decimals = 4;
  return Number(n).toLocaleString('nl-NL', { minimumFractionDigits: 2, maximumFractionDigits: decimals });
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

// ── Strategy Performance ──────────────────────────────────────────────────

async function refreshStrategies() {
  try {
    const res = await fetch(`${API}/api/analytics/strategies?days=30`);
    const strategies = await res.json();
    const tbody = document.getElementById('strategy-body');
    if (!strategies.length) {
      tbody.innerHTML = '<tr><td colspan="4" class="text-gray-600 text-center py-4">No data</td></tr>';
      return;
    }
    tbody.innerHTML = strategies.map(s => {
      const pnlClass = s.total_pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
      const wr = (s.win_rate * 100).toFixed(1);
      return `<tr>
        <td class="text-white">${s.name}</td>
        <td>${s.total_trades}</td>
        <td>${wr}%</td>
        <td class="${pnlClass} font-medium">${fmt(s.total_pnl)}</td>
      </tr>`;
    }).join('');
  } catch (e) { console.error('refreshStrategies', e); }
}

// ── Risk Gate Log ─────────────────────────────────────────────────────────

async function refreshRiskGate() {
  try {
    const res = await fetch(`${API}/api/risk/decisions?limit=10`);
    const decisions = await res.json();
    const el = document.getElementById('risk-gate-log');
    if (!el) return;
    if (!decisions.length) {
      el.innerHTML = '<div class="text-gray-600 italic">No decisions yet</div>';
      return;
    }
    el.innerHTML = decisions.map(d => {
      const badge = d.approved
        ? '<span class="px-1.5 py-0.5 rounded text-xs font-medium" style="background:#10b98122;color:#10b981">APPROVED</span>'
        : '<span class="px-1.5 py-0.5 rounded text-xs font-medium" style="background:#ef444422;color:#ef4444">REJECTED</span>';
      const ts = d.timestamp ? d.timestamp.slice(11, 19) : '';
      const failedGates = d.gate_details
        ? Object.entries(d.gate_details).filter(([, g]) => !g.passed).map(([name]) => name.replace('_', ' ')).join(', ')
        : '';
      const gateInfo = !d.approved && failedGates
        ? `<div class="text-gray-500 mt-0.5">Failed: <span class="text-red-400">${failedGates}</span></div>`
        : '';
      return `<div class="py-1.5 border-b border-gray-800">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <span class="text-gray-500">${ts}</span>
            <span class="text-white font-medium">${d.symbol || '?'}</span>
            <span class="text-gray-400">${d.side || ''}</span>
          </div>
          ${badge}
        </div>
        ${gateInfo}
      </div>`;
    }).join('');
  } catch (e) { console.error('refreshRiskGate', e); }
}

// ── Paper Reset ───────────────────────────────────────────────────────────

let isPaperMode = false;

async function paperReset() {
  if (!confirm('Reset paper trading balance and positions?')) return;
  const btn = document.getElementById('btn-paper-reset');
  btn.disabled = true;
  btn.textContent = 'Resetting...';
  try {
    await fetch(`${API}/api/bot/paper/reset`, { method: 'POST' });
    await refreshPortfolio();
    await refreshTrades();
    await loadEquityHistory();
  } catch (e) { console.error('paperReset', e); }
  btn.disabled = false;
  btn.textContent = 'Reset Paper';
}

function updatePaperResetVisibility(paperTrading) {
  isPaperMode = paperTrading;
  const btn = document.getElementById('btn-paper-reset');
  if (btn) {
    if (paperTrading) {
      btn.classList.remove('hidden');
    } else {
      btn.classList.add('hidden');
    }
  }
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────

async function init() {
  initPriceChart();
  initEquityChart();
  connectWS();

  await Promise.all([
    loadMarkets(), refreshStatus(), refreshPortfolio(), refreshAnalytics(),
    loadCandles(), loadEquityHistory(), refreshTrades(), refreshSentiment(),
    refreshStrategies(), refreshRiskGate(),
  ]);

  // Poll status every 2s until candles are ready, then slow to 30s
  const statusPoll = setInterval(async () => {
    await refreshStatus();
    const el = document.getElementById('candle-status-text');
    if (el && el.textContent.includes('ready')) {
      clearInterval(statusPoll);
      setInterval(refreshStatus, 30000);
    }
  }, 2000);

  // Periodic refresh every 30s
  setInterval(async () => {
    await Promise.all([
      refreshPortfolio(), refreshAnalytics(), loadEquityHistory(),
      refreshStrategies(), refreshRiskGate(),
    ]);
  }, 30000);

  setInterval(refreshTrades, 60000);
  setInterval(refreshSentiment, 120000);

  // Live price update every 500ms
  setInterval(pollPrice, 1000);
}

document.addEventListener('DOMContentLoaded', init);
