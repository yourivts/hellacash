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
    height: 340,
    layout: { background: { color: '#0c0c0f' }, textColor: '#71717a' },
    grid: { vertLines: { color: '#18181b' }, horzLines: { color: '#18181b' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#1e1e22' },
    timeScale: { borderColor: '#1e1e22', timeVisible: true },
  });
  candleSeries = chart.addCandlestickSeries({
    upColor: '#22c55e', downColor: '#ef4444',
    borderUpColor: '#22c55e', borderDownColor: '#ef4444',
    wickUpColor: '#22c55e', wickDownColor: '#ef4444',
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
      upColor: '#22c55e', downColor: '#ef4444',
      borderUpColor: '#22c55e', borderDownColor: '#ef4444',
      wickUpColor: '#22c55e', wickDownColor: '#ef4444',
      priceFormat: { type: 'price', precision: pf.precision, minMove: pf.minMove },
    });
    candleSeries.setData(candles);
  } catch (e) { console.error('loadCandles', e); }
}

let lastCandle = null;
let lastCandlePair = null;
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
        deltaEl.className = diff > 0 ? 'pnl-pos' : 'pnl-neg';
        deltaEl.style.fontSize = '12px';
        deltaEl.style.fontWeight = '500';
      }
    }
    prevPrice = price;

    // Update candle
    const now = Math.floor(Date.now() / 1000);
    const intervalSecs = { '1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400, '1d': 86400 };
    const bucket = intervalSecs[currentInterval] || 300;
    const candleTime = Math.floor(now / bucket) * bucket;

    if (lastCandlePair !== currentPair) {
      lastCandle = null;
      lastCandlePair = currentPair;
    }

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
      <span class="ob-price" style="color:var(--green)">${fmt(price)}</span>
      <span class="ob-vol" style="color:var(--text-faint)">${vol.toFixed(4)}</span>
    </div>`;
  }).join('');

  const askHtml = asks.slice(0, 10).map(a => {
    const price = parseFloat(a[0]);
    const vol = parseFloat(a[1]);
    const pct = (vol / maxAskVol) * 100;
    return `<div class="ob-row ob-ask">
      <div class="ob-bg" style="width:${pct}%"></div>
      <span class="ob-price" style="color:var(--red)">${fmt(price)}</span>
      <span class="ob-vol" style="color:var(--text-faint)">${vol.toFixed(4)}</span>
    </div>`;
  }).join('');

  el.innerHTML = `<div class="ob-col">${bidHtml}</div><div class="ob-col">${askHtml}</div>`;

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
    data: { datasets: [{ label: 'Equity', data: [], borderColor: '#3b82f6', borderWidth: 2, fill: true, backgroundColor: 'rgba(59,130,246,0.06)', tension: 0.3, pointRadius: 0 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          type: 'time',
          time: { unit: 'hour', displayFormats: { hour: 'MMM d HH:mm', day: 'MMM d' }, tooltipFormat: 'MMM d HH:mm' },
          grid: { color: '#18181b' },
          ticks: { color: '#3f3f46', font: { size: 10 }, maxTicksLimit: 5 },
        },
        y: { grid: { color: '#18181b' }, ticks: { color: '#3f3f46', font: { size: 10 } } },
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
      `<span class="${dd > 3 ? 'pnl-neg' : 'pnl-pos'}">${dd.toFixed(2)}%</span>`;
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
    el.innerHTML = '<div style="font-size:12px;color:var(--text-faint);font-style:italic;">No results</div>';
    return;
  }
  el.innerHTML = rows.map(r => {
    const pct = Math.round(((r.score + 1) / 2) * 100);
    const color = r.score > 0.1 ? 'var(--green)' : r.score < -0.1 ? 'var(--red)' : 'var(--amber)';
    const label = r.score > 0.1 ? 'Bullish' : r.score < -0.1 ? 'Bearish' : 'Neutral';
    const isActive = r.symbol === currentPair;
    const nameColor = isActive ? 'var(--text-primary)' : 'var(--text-secondary)';
    const nameWeight = isActive ? '500' : '400';
    return `<div style="cursor:pointer;padding:4px 2px;border-radius:4px;" onmouseover="this.style.background='var(--bg-hover)'" onmouseout="this.style.background=''" onclick="switchPair('${r.symbol}')">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
        <span style="font-size:12px;color:${nameColor};font-weight:${nameWeight};">${r.base}</span>
        <div style="display:flex;align-items:center;gap:6px;">
          <div style="width:48px;height:4px;background:var(--bg-surface);border-radius:2px;overflow:hidden;">
            <div class="sentiment-bar" style="width:${pct}%;background:${color};"></div>
          </div>
          <span style="font-size:11px;color:${color};width:36px;text-align:right;">${r.score.toFixed(2)}</span>
        </div>
      </div>
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
      modeBadge.textContent = 'PAPER'; modeBadge.className = 'badge badge-paper';
    } else {
      modeBadge.textContent = 'LIVE'; modeBadge.className = 'badge badge-live';
    }
    updateToggleButton(d.running);
    updatePaperResetVisibility(d.paper_trading);

    // Candle loading status
    const candleEl = document.getElementById('candle-status');
    const candleText = document.getElementById('candle-status-text');
    if (d.candles_ready) {
      candleEl.className = 'badge';
      candleEl.style.background = 'rgba(34,197,94,0.12)';
      candleEl.style.color = '#22c55e';
      candleEl.style.border = '1px solid rgba(34,197,94,0.25)';
      candleText.textContent = 'Candles ready';
      setTimeout(() => candleEl.style.opacity = '0.5', 3000);
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
          rlBadge.style.background = 'rgba(239,68,68,0.12)';
          rlBadge.style.color = '#ef4444';
          rlBadge.style.border = '1px solid rgba(239,68,68,0.25)';
        } else if (rl.remaining < 300) {
          rlBadge.style.background = 'rgba(245,158,11,0.12)';
          rlBadge.style.color = '#f59e0b';
          rlBadge.style.border = '1px solid rgba(245,158,11,0.25)';
        } else {
          rlBadge.style.background = 'rgba(34,197,94,0.12)';
          rlBadge.style.color = '#22c55e';
          rlBadge.style.border = '1px solid rgba(34,197,94,0.25)';
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
  if (!openPositions.length) { el.innerHTML = '<div style="font-size:12px;color:var(--text-faint);font-style:italic;">No positions</div>'; return; }
  el.innerHTML = openPositions.map(p => {
    const pnl = p.unrealized_pnl || 0;
    const price = p.current_price || 0;
    const value = price * (p.quantity || 0);
    const entry = p.entry_price || 0;
    const roiPct = entry > 0 ? ((price - entry) / entry * 100) : 0;
    return `<div class="pos-item">
      <div>
        <div style="color:var(--text-primary);font-size:13px;font-weight:500;">${p.symbol}</div>
        <div style="color:var(--text-faint);font-size:11px;">${p.strategy_name}</div>
        <div style="color:var(--text-faint);font-size:11px;">Entry €${fmt(entry)}</div>
      </div>
      <div style="text-align:right;">
        <div class="${pnl>=0?'pnl-pos':'pnl-neg'}" style="font-weight:500;font-size:13px;">€${fmt(pnl)} <span style="font-size:11px;">(${roiPct>=0?'+':''}${roiPct.toFixed(2)}%)</span></div>
        <div style="color:var(--text-faint);font-size:11px;">${p.quantity?.toFixed(4)||'—'} @ €${fmt(price)}</div>
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
  if (!trades.length) { tbody.innerHTML = '<tr><td colspan="10" style="text-align:center;color:var(--text-faint);padding:24px;">No trades yet</td></tr>'; return; }
  tbody.innerHTML = trades.map(t => {
    const pnlClass = t.net_pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
    const hold = fmtDuration(t.hold_seconds);
    return `<tr>
      <td style="color:var(--text-primary);font-weight:500;">${t.symbol}</td>
      <td style="color:var(--text-muted);">${t.strategy_name}</td>
      <td class="text-right">€${fmt(t.entry_price)}</td>
      <td class="text-right">€${fmt(t.exit_price)}</td>
      <td class="text-right">${t.quantity?.toFixed(4)||'—'}</td>
      <td class="${pnlClass} text-right" style="font-weight:500;">€${fmt(t.net_pnl)}</td>
      <td class="${pnlClass} text-right">${t.roi_pct?.toFixed(2)||'—'}%</td>
      <td style="color:var(--text-muted);">${hold}</td>
      <td><span class="reason-badge">${t.exit_reason}</span></td>
      <td><span style="font-size:11px;color:${t.paper_trade?'var(--amber)':'var(--green)'}">${t.paper_trade?'paper':'live'}</span></td>
    </tr>`;
  }).join('');
}

function addSignalFeedItem(data) {
  const el = document.getElementById('signal-feed');
  const dir = data.direction || data.type || 'NEUTRAL';
  const badgeCls = dir === 'LONG' ? 'signal-badge-long' : dir === 'SHORT' ? 'signal-badge-short' : '';
  const now = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
  const item = document.createElement('div');
  item.className = 'signal-item';
  item.innerHTML = `<span style="color:var(--text-faint)">${now}</span><span class="signal-badge ${badgeCls}">${dir}</span><span style="color:var(--text-secondary)">${data.symbol||data.type||''}</span><span style="color:var(--text-faint);margin-left:auto;">${data.strength ? 'strength '+(data.strength*100).toFixed(0)+'%' : ''}</span>`;
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
    btn.className = 'btn btn-stop';
  } else {
    btn.textContent = 'Start';
    btn.className = 'btn btn-start';
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
    b.className = 'tf-btn' + (b.textContent.trim() === iv ? ' active' : '');
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
  el.className = num >= 0 ? 'pnl-pos' : 'pnl-neg';
  el.style.fontWeight = '500';
}

// ── Strategy Performance ──────────────────────────────────────────────────

async function refreshStrategies() {
  try {
    const res = await fetch(`${API}/api/analytics/strategies?days=30`);
    const strategies = await res.json();
    const tbody = document.getElementById('strategy-body');
    if (!strategies.length) {
      tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text-faint);padding:16px;">No data</td></tr>';
      return;
    }
    tbody.innerHTML = strategies.map(s => {
      const pnlClass = s.total_pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
      const wr = (s.win_rate * 100).toFixed(1);
      return `<tr>
        <td style="color:var(--text-primary);">${s.name}</td>
        <td>${s.total_trades}</td>
        <td>${wr}%</td>
        <td class="${pnlClass} text-right" style="font-weight:500;">€${fmt(s.total_pnl)}</td>
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
      el.innerHTML = '<div style="font-size:12px;color:var(--text-faint);font-style:italic;">No decisions yet</div>';
      return;
    }
    el.innerHTML = decisions.map(d => {
      const dotColor = d.approved ? 'var(--green)' : 'var(--red)';
      const label = d.approved ? 'approved' : 'rejected';
      const ts = d.timestamp ? d.timestamp.slice(11, 16) : '';
      const failedGates = d.gate_details
        ? Object.entries(d.gate_details).filter(([, g]) => !g.passed).map(([name]) => name.replace('_', ' ')).join(', ')
        : '';
      const gateInfo = !d.approved && failedGates
        ? `<div style="font-size:10px;color:var(--text-faint);margin-top:2px;margin-left:14px;">Failed: <span style="color:var(--red)">${failedGates}</span></div>`
        : '';
      return `<div class="risk-item" style="flex-wrap:wrap;">
        <div style="width:6px;height:6px;border-radius:50%;background:${dotColor};flex-shrink:0;"></div>
        <span style="color:var(--text-secondary);font-size:12px;">${d.symbol || '?'} ${label}</span>
        <span style="color:var(--text-faint);font-size:11px;margin-left:auto;">${ts}</span>
        ${gateInfo ? `<div style="width:100%;">${gateInfo}</div>` : ''}
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
