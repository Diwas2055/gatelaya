/* GateLaya dashboard — Phase 3 frontend.
   Alpine.js store, vanilla fetch. No build step.
   All state lives in one root store: dashboard(). */

'use strict';

const PAGE_SIZE = 50;
const THRESHOLD_CHECKS = ['pii', 'injection', 'toxicity', 'secret_leak'];
const DEFAULT_THRESHOLDS = { pii: 0.85, injection: 0.9, toxicity: 0.9, secret_leak: 0.85 };
const DEFAULT_ACTIONS = { pii: 'mask', injection: 'block', toxicity: 'block', secret_leak: 'block' };
const FILTER_KEYS = ['check', 'action', 'mode', 'sha256', 'since', 'until'];

/* ---------- formatting helpers (pure) ---------- */

function relTime(iso) {
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return iso || '';
  let sec = Math.round((Date.now() - t) / 1000);
  const past = sec >= 0;
  sec = Math.abs(sec);
  if (sec < 45) return past ? 'just now' : 'in a moment';
  const units = [
    ['y', 31536000], ['mo', 2592000], ['w', 604800],
    ['d', 86400], ['h', 3600], ['m', 60],
  ];
  let out = 'just now';
  for (const [name, size] of units) {
    if (sec >= size) {
      out = Math.floor(sec / size) + name;
      break;
    }
  }
  return past ? out + ' ago' : 'in ' + out;
}

function fmtClock(iso) {
  const t = new Date(iso);
  if (Number.isNaN(t.getTime())) return String(iso || '—');
  return t.toLocaleString();
}

function fmtNum(n, digits) {
  return typeof n === 'number' && Number.isFinite(n) ? n.toFixed(digits) : '—';
}

/* ---------- root store ---------- */

function dashboard() {
  return {
    /* ===== core ===== */
    tab: 'decisions',
    token: localStorage.getItem('gatelaya_token') || '',
    tokenDraft: '',
    unauthorized: false,
    gearOpen: false,
    health: 'unknown', // ok | down | unknown
    stats: null,
    statsLoading: true,
    pendingCount: 0,
    toasts: [],

    /* ===== decisions ===== */
    filters: { check: '', action: '', mode: '', sha256: '', since: '', until: '' },
    applied: { check: '', action: '', mode: '', sha256: '', since: '', until: '' },
    decisions: [],
    decisionsTotal: 0,
    decisionsLoading: false,
    decisionsLoaded: false,
    page: 0,
    drawer: null,

    /* ===== review ===== */
    reviewTab: 'pending',
    reviewPending: [],
    reviewResolved: [],
    pendingTotal: 0,
    resolvedTotal: 0,
    reviewLoading: false,
    reviewLoaded: false,
    confirming: null, // "<id>|<action>"
    contextOpen: {},
    drafts: {},

    /* ===== settings ===== */
    cfg: null,
    configPath: '',
    configExists: true,
    configLoading: true,
    draft: null,
    saving: false,
    saveBanner: null,

    /* ===== calibration ===== */
    cal: { temperatures: null, path: '', exists: false },
    calLoaded: false,
    calLoading: true,
    uploading: false,
    uploadFile: null,
    uploadCheck: '',
    uploadResult: null,
    uploadError: null,

    /* ================= lifecycle ================= */

    init() {
      this.tokenDraft = this.token;
      this.resetDraft(); // settings usable even if /api/config fails
      this.checkHealth();
      this.loadStats();
      this.loadPendingCount();
      this.loadDecisions();
      setInterval(() => { if (!this.unauthorized) this.loadStats(); }, 15000);
      setInterval(() => { if (!this.unauthorized) this.checkHealth(); }, 30000);
    },

    setTab(id) {
      if (this.tab === id) return;
      this.tab = id;
      this.confirming = null;
      if (id === 'review') this.loadReview();
      if (id === 'settings' && !this.cfg) this.loadConfig();
      if (id === 'calibration' && !this.calLoaded) this.loadCalibration();
    },

    /* ================= transport ================= */

    async api(path, opts = {}) {
      const headers = Object.assign({}, opts.headers);
      if (this.token) headers['Authorization'] = 'Bearer ' + this.token;
      if (opts.body && !(opts.body instanceof FormData) && !headers['Content-Type']) {
        headers['Content-Type'] = 'application/json';
      }
      const res = await fetch(path, Object.assign({}, opts, { headers }));

      if (res.status === 401) {
        this.unauthorized = true;
        this.gearOpen = false;
        const err = new Error('Unauthorized');
        err.status = 401;
        throw err;
      }
      if (!res.ok) {
        let detail = '';
        try {
          const j = await res.json();
          detail = typeof j.detail === 'string'
            ? j.detail
            : (j.detail ? JSON.stringify(j.detail) : (j.error || j.message || JSON.stringify(j)));
        } catch (_) {
          detail = res.statusText || '';
        }
        const err = new Error(detail || 'HTTP ' + res.status);
        err.status = res.status;
        throw err;
      }
      if (path.endsWith('.jsonl')) return res;
      const ct = res.headers.get('content-type') || '';
      if (ct.includes('application/json')) return res.json();
      return res;
    },

    toast(message, type) {
      const id = Date.now() + Math.random();
      this.toasts.push({ id, message, type: type || 'error' });
      setTimeout(() => {
        this.toasts = this.toasts.filter((t) => t.id !== id);
      }, 5000);
    },

    isAuthError(e) {
      return e && e.status === 401;
    },

    /* ================= header: token + health ================= */

    saveToken() {
      this.token = (this.tokenDraft || '').trim();
      if (this.token) localStorage.setItem('gatelaya_token', this.token);
      else localStorage.removeItem('gatelaya_token');
      this.gearOpen = false;
      this.unauthorized = false;
      this.health = 'unknown';
      this.checkHealth();
      this.loadStats();
      this.loadPendingCount();
      this.loadDecisions();
      if (this.tab === 'review') this.loadReview();
      if (this.tab === 'settings') this.loadConfig();
      if (this.tab === 'calibration') this.loadCalibration();
      this.toast(this.token ? 'Token saved' : 'Token cleared', 'ok');
    },

    async checkHealth() {
      try {
        const r = await this.api('/api/health');
        this.health = r && r.status === 'ok' ? 'ok' : 'down';
      } catch (e) {
        this.health = this.isAuthError(e) ? 'unknown' : 'down';
      }
    },

    /* ================= stats ================= */

    async loadStats() {
      try {
        this.stats = await this.api('/api/stats?hours=24');
        this.statsLoading = false;
        this.loadPendingCount();
      } catch (e) {
        this.statsLoading = false;
        if (!this.isAuthError(e)) this.toast('Stats unavailable: ' + e.message);
      }
    },

    statCards() {
      const s = this.stats;
      const n = (k) => (s && typeof s[k] === 'number' ? String(s[k]) : '—');
      return [
        { label: 'Total', value: n('total'), sub: 'last 24h', bar: 'bg-slate-400' },
        { label: 'Blocked', value: n('blocked'), sub: 'requests denied', bar: 'bg-rose-500' },
        { label: 'Flagged', value: n('flagged'), sub: 'needs review', bar: 'bg-amber-500' },
        { label: 'Masked', value: n('masked'), sub: 'PII redacted', bar: 'bg-violet-500' },
        { label: 'Allowed', value: n('allowed'), sub: 'passed through', bar: 'bg-emerald-500' },
        { label: 'Avg latency', value: fmtNum(s && s.avg_latency_ms, 1), sub: 'ms / request', bar: 'bg-sky-400' },
      ];
    },

    /* ================= decisions ================= */

    async loadDecisions() {
      this.decisionsLoading = true;
      const p = new URLSearchParams();
      for (const k of FILTER_KEYS) {
        if (this.applied[k]) p.set(k, this.applied[k]);
      }
      p.set('limit', String(PAGE_SIZE));
      p.set('offset', String(this.page * PAGE_SIZE));
      try {
        const r = await this.api('/api/decisions?' + p.toString());
        this.decisions = (r && r.items) || [];
        this.decisionsTotal = (r && r.total) || 0;
        this.decisionsLoaded = true;
      } catch (e) {
        if (!this.isAuthError(e)) this.toast('Failed to load decisions: ' + e.message);
      } finally {
        this.decisionsLoading = false;
      }
    },

    applyFilters() {
      this.applied = Object.assign({}, this.filters);
      this.page = 0;
      this.loadDecisions();
    },

    resetFilters() {
      this.filters = { check: '', action: '', mode: '', sha256: '', since: '', until: '' };
      this.applied = Object.assign({}, this.filters);
      this.page = 0;
      this.loadDecisions();
    },

    rangeLabel() {
      if (this.decisionsTotal === 0) return '0 of 0';
      const from = this.page * PAGE_SIZE + 1;
      const to = Math.min((this.page + 1) * PAGE_SIZE, this.decisionsTotal);
      return from + '\u2013' + to + ' of ' + this.decisionsTotal;
    },

    /* ================= review ================= */

    async loadPendingCount() {
      try {
        const r = await this.api('/api/review?status=pending&limit=1&offset=0');
        this.pendingCount = (r && r.total) || 0;
      } catch (_) { /* silent — badge refresh */ }
    },

    async loadReview() {
      this.reviewLoading = true;
      const status = this.reviewTab === 'pending' ? 'pending' : 'resolved';
      try {
        const r = await this.api('/api/review?status=' + status + '&limit=' + PAGE_SIZE + '&offset=0');
        const items = (r && r.items) || [];
        for (const item of items) {
          const id = item.decision.id;
          if (!this.drafts[id]) this.drafts[id] = { text: '', note: '' };
          if (typeof this.contextOpen[id] !== 'boolean') this.contextOpen[id] = false;
        }
        if (status === 'pending') {
          this.reviewPending = items;
          this.pendingTotal = (r && r.total) || 0;
          this.pendingCount = this.pendingTotal;
        } else {
          this.reviewResolved = items;
          this.resolvedTotal = (r && r.total) || 0;
        }
        this.reviewLoaded = true;
      } catch (e) {
        if (!this.isAuthError(e)) this.toast('Failed to load review queue: ' + e.message);
      } finally {
        this.reviewLoading = false;
      }
    },

    reviewItems() {
      return this.reviewTab === 'pending' ? this.reviewPending : this.reviewResolved;
    },

    toggleContext(id) {
      this.contextOpen[id] = !this.contextOpen[id];
    },

    async labelDecision(id, action) {
      const key = id + '|' + action;
      if (this.confirming !== key) {
        this.confirming = key;
        return;
      }
      this.confirming = null;
      const draft = this.drafts[id] || {};
      const body = { label: action };
      if (draft.note) body.note = draft.note;
      if (draft.text) body.text = draft.text;
      try {
        await this.api('/api/review/' + encodeURIComponent(id) + '/label', {
          method: 'POST',
          body: JSON.stringify(body),
        });
        this.reviewPending = this.reviewPending.filter((x) => x.decision.id !== id);
        this.pendingTotal = Math.max(0, this.pendingTotal - 1);
        this.pendingCount = Math.max(0, this.pendingCount - 1);
        this.resolvedTotal = this.resolvedTotal + 1;
        this.toast('Labeled "' + action + '"', 'ok');
        if (this.reviewTab === 'resolved') this.loadReview();
      } catch (e) {
        if (e.status === 409) {
          this.toast('Already labeled — refreshing queue', 'error');
          this.loadReview();
        } else if (!this.isAuthError(e)) {
          this.toast('Label failed: ' + e.message);
        }
      }
    },

    async exportJsonl() {
      try {
        const res = await this.api('/api/review/export.jsonl');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'gatelaya-review.jsonl';
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        this.toast('Export downloaded', 'ok');
      } catch (e) {
        if (!this.isAuthError(e)) this.toast('Export failed: ' + e.message);
      }
    },

    /* ================= settings ================= */

    async loadConfig() {
      this.configLoading = true;
      try {
        const r = await this.api('/api/config');
        this.cfg = (r && r.config) || {};
        this.configPath = (r && r.path) || '';
        this.configExists = !r || r.exists !== false;
        this.resetDraft();
      } catch (e) {
        if (!this.isAuthError(e)) this.toast('Failed to load config: ' + e.message);
      } finally {
        this.configLoading = false;
      }
    },

    resetDraft() {
      const c = this.cfg || {};
      const thresholds = Object.assign({}, DEFAULT_THRESHOLDS, c.thresholds || {});
      const actions = Object.assign({}, DEFAULT_ACTIONS, c.actions || {});
      for (const k of THRESHOLD_CHECKS) {
        if (typeof thresholds[k] !== 'number') thresholds[k] = DEFAULT_THRESHOLDS[k];
        if (!actions[k]) actions[k] = DEFAULT_ACTIONS[k];
      }
      this.draft = {
        thresholds,
        actions,
        enabled_checks: Array.isArray(c.enabled_checks) && c.enabled_checks.length
          ? c.enabled_checks.slice()
          : THRESHOLD_CHECKS.slice(),
        fail_open: c.fail_open !== false,
      };
    },

    draftThreshold(k) {
      if (!this.draft) return DEFAULT_THRESHOLDS[k];
      const v = this.draft.thresholds[k];
      return typeof v === 'number' ? v : DEFAULT_THRESHOLDS[k];
    },

    async saveConfig() {
      if (!this.draft) return;
      this.saving = true;
      try {
        const r = await this.api('/api/config', {
          method: 'PUT',
          body: JSON.stringify({
            thresholds: this.draft.thresholds,
            actions: this.draft.actions,
            enabled_checks: this.draft.enabled_checks,
            fail_open: this.draft.fail_open,
          }),
        });
        if (r && r.path) this.configPath = r.path;
        this.saveBanner = 'Saved to ' + (this.configPath || 'config') + '. Restart LiteLLM proxy to apply.';
        this.toast('Config saved', 'ok');
      } catch (e) {
        if (!this.isAuthError(e)) this.toast('Save failed: ' + e.message);
      } finally {
        this.saving = false;
      }
    },

    /* ================= calibration ================= */

    async loadCalibration() {
      this.calLoading = true;
      try {
        this.cal = await this.api('/api/calibration');
        this.calLoaded = true;
      } catch (e) {
        if (!this.isAuthError(e)) this.toast('Failed to load calibration: ' + e.message);
      } finally {
        this.calLoading = false;
      }
    },

    tempRows() {
      return this.flattenTemps(this.cal && this.cal.temperatures);
    },

    resultRows() {
      return this.flattenTemps(this.uploadResult && this.uploadResult.temperatures);
    },

    flattenTemps(temps) {
      const rows = [];
      if (!temps || typeof temps !== 'object') return rows;
      for (const group of Object.keys(temps)) {
        const v = temps[group];
        if (v && typeof v === 'object') {
          for (const bucket of Object.keys(v)) rows.push({ group, bucket, value: v[bucket] });
        } else {
          rows.push({ group, bucket: 'default', value: v });
        }
      }
      return rows;
    },

    fmtT(v) {
      return typeof v === 'number' && Number.isFinite(v) ? 'T = ' + v.toFixed(3) : String(v);
    },

    async uploadCalibration() {
      if (!this.uploadFile) {
        this.toast('Choose a .jsonl file first');
        return;
      }
      this.uploading = true;
      this.uploadError = null;
      this.uploadResult = null;
      const fd = new FormData();
      fd.append('file', this.uploadFile);
      if (this.uploadCheck) fd.append('check', this.uploadCheck);
      try {
        const r = await this.api('/api/calibration/upload', { method: 'POST', body: fd });
        this.uploadResult = r;
        if (this.cal) this.cal = Object.assign({}, this.cal, { temperatures: r.temperatures, exists: true });
        this.toast('Calibration fitted (' + (r.n_rows || 0) + ' rows)', 'ok');
      } catch (e) {
        if (e.status === 503) {
          this.uploadError = 'Laya is not installed on the server — run: pip install laya';
        } else if (e.status === 400) {
          this.uploadError = 'Bad input: ' + e.message;
        } else if (!this.isAuthError(e)) {
          this.uploadError = e.message;
        }
      } finally {
        this.uploading = false;
      }
    },

    /* ================= display helpers ================= */

    fmtP(v) {
      return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(2) : '—';
    },

    fmtMs(v) {
      return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(1) : '—';
    },

    fmtProbs(probs) {
      if (!probs || typeof probs.raw !== 'number') return '—';
      return this.fmtP(probs.raw) + ' \u2192 ' + this.fmtP(probs.calibrated);
    },

    relTime: relTime,
    fmtClock: fmtClock,

    trunc(s, n) {
      if (!s) return '—';
      s = String(s);
      return s.length > n ? s.slice(0, n) + '\u2026' : s;
    },

    checkPill(check) {
      const map = {
        pii: 'border-sky-500/30 bg-sky-500/15 text-sky-300',
        injection: 'border-fuchsia-500/30 bg-fuchsia-500/15 text-fuchsia-300',
        toxicity: 'border-pink-500/30 bg-pink-500/15 text-pink-300',
        secret_leak: 'border-amber-500/30 bg-amber-500/15 text-amber-300',
        routing: 'border-teal-500/30 bg-teal-500/15 text-teal-300',
        routing_error: 'border-orange-500/30 bg-orange-500/15 text-orange-300',
        agent_error: 'border-rose-500/30 bg-rose-500/15 text-rose-300',
      };
      return map[check] || 'border-slate-700 bg-slate-800 text-slate-300';
    },

    actionPill(action) {
      const map = {
        block: 'border-rose-500/30 bg-rose-500/15 text-rose-300',
        flag: 'border-amber-500/30 bg-amber-500/15 text-amber-300',
        mask: 'border-violet-500/30 bg-violet-500/15 text-violet-300',
        allow: 'border-emerald-500/30 bg-emerald-500/15 text-emerald-300',
      };
      return map[action] || 'border-slate-700 bg-slate-800 text-slate-300';
    },

    detailEntries(d) {
      const detail = d && d.detail;
      if (!detail || typeof detail !== 'object') return [];
      return Object.entries(detail);
    },

    fmtDetailValue(v) {
      let s;
      if (typeof v === 'string') s = v;
      else {
        try { s = JSON.stringify(v); } catch (_) { s = String(v); }
      }
      if (s === undefined || s === null) s = String(v);
      return s.length > 48 ? s.slice(0, 48) + '\u2026' : s;
    },

    async copy(text) {
      if (!text) return;
      try {
        await navigator.clipboard.writeText(String(text));
        this.toast('Copied to clipboard', 'ok');
      } catch (_) {
        const ta = document.createElement('textarea');
        ta.value = String(text);
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        let ok = false;
        try { ok = document.execCommand('copy'); } catch (_) { ok = false; }
        ta.remove();
        this.toast(ok ? 'Copied to clipboard' : 'Copy failed', ok ? 'ok' : 'error');
      }
    },
  };
}

window.dashboard = dashboard;
