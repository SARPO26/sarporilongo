/*
 * Spatial pose optimizer for a segmented RuView layout.
 *
 * Pipeline per WebSocket message:
 *   1. Per link (b2b = board-to-board, b1r / b2r = board-to-router):
 *      Hampel spike filter -> PC1 of the subcarriers -> rolling variance.
 *      Phase is cleaned of CFO/SFO (linear fit over subcarriers removed).
 *   2. Zone from link disturbance (EMPTY / DESK / BED / DOOR), with hysteresis.
 *   3. Raw model pose probabilities x zone prior (Bayes), then EMA smoothing.
 *
 * Works in the browser (window.SpatialPose) and in Node (require).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.SpatialPose = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ---------- math helpers ----------

  function mean(a) {
    if (!a.length) return 0;
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i];
    return s / a.length;
  }

  function variance(a) {
    if (a.length < 2) return 0;
    const m = mean(a);
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m);
    return s / a.length;
  }

  function median(a) {
    if (!a.length) return 0;
    const s = Array.from(a).sort((x, y) => x - y);
    const h = s.length >> 1;
    return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
  }

  // Midpoint for variances, which span orders of magnitude.
  function geoMid(a, b) {
    return a > 0 && b > 0 ? Math.sqrt(a * b) : (a + b) / 2;
  }

  function isNum(x) {
    return typeof x === 'number' && Number.isFinite(x);
  }

  // ---------- DSP ----------

  function unwrapPhase(phase) {
    const out = new Array(phase.length);
    if (!phase.length) return out;
    let offset = 0;
    out[0] = phase[0];
    for (let i = 1; i < phase.length; i++) {
      const d = phase[i] - phase[i - 1];
      if (d > Math.PI) offset -= 2 * Math.PI;
      else if (d < -Math.PI) offset += 2 * Math.PI;
      out[i] = phase[i] + offset;
    }
    return out;
  }

  /**
   * Remove CFO/SFO: unwrap phase across subcarriers, fit a line
   * (phase = a*k + b) and subtract it. What is left is the phase change
   * caused by the environment.
   * @param {number[]} phase   radians, one per subcarrier
   * @param {number[]} [index] subcarrier indices (default 0..N-1)
   */
  function sanitizePhase(phase, index) {
    const n = phase.length;
    if (n < 2) return Array.from(phase);
    const u = unwrapPhase(phase);
    const k = index && index.length === n ? index : u.map((_, i) => i);
    const mk = mean(k);
    const mu = mean(u);
    let num = 0;
    let den = 0;
    for (let i = 0; i < n; i++) {
      num += (k[i] - mk) * (u[i] - mu);
      den += (k[i] - mk) * (k[i] - mk);
    }
    const slope = den ? num / den : 0;
    const intercept = mu - slope * mk;
    return u.map((v, i) => v - (slope * k[i] + intercept));
  }

  /**
   * Centered Hampel filter: the middle sample of the window is replaced by
   * the window median if it is more than nSigma robust standard deviations
   * away. Output lags input by floor(window / 2) samples; this keeps real
   * movement (which has neighbours on both sides) from being flagged.
   */
  class HampelFilter {
    constructor(window = 7, nSigma = 3, minScale = 1e-6) {
      this.window = Math.max(3, window | 0) | 1; // odd
      this.nSigma = nSigma;
      this.minScale = minScale;
      this.buf = [];
      this.outliers = 0;
      this.count = 0;
    }

    push(x) {
      this.count++;
      this.buf.push(x);
      if (this.buf.length > this.window) this.buf.shift();
      if (this.buf.length < this.window) return x; // start-up: pass through
      const c = this.buf[this.window >> 1];
      const med = median(this.buf);
      const mad = 1.4826 * median(this.buf.map((v) => Math.abs(v - med)));
      if (Math.abs(c - med) > this.nSigma * Math.max(mad, this.minScale)) {
        this.outliers++;
        return med;
      }
      return c;
    }
  }

  /** One Hampel filter per subcarrier. */
  class VectorHampel {
    constructor(window, nSigma) {
      this.window = window;
      this.nSigma = nSigma;
      this.filters = [];
    }

    push(vec) {
      if (this.filters.length !== vec.length) {
        this.filters = vec.map(() => new HampelFilter(this.window, this.nSigma));
      }
      return vec.map((v, i) => this.filters[i].push(v));
    }

    get outliers() {
      return this.filters.reduce((s, f) => s + f.outliers, 0);
    }

    get count() {
      return this.filters.reduce((s, f) => s + f.count, 0);
    }
  }

  /**
   * Tracks the first principal component of a sliding window of subcarrier
   * amplitude vectors and returns the newest sample projected onto it.
   * The subcarriers that move together the most dominate PC1.
   */
  class PC1Tracker {
    constructor(window = 20, iterations = 12) {
      this.window = Math.max(3, window | 0);
      this.iterations = iterations;
      this.buf = [];
      this.v = null;
      this.explained = 0;
    }

    push(vec) {
      const n = vec.length;
      if (this.buf.length && this.buf[0].length !== n) {
        this.buf = [];
        this.v = null;
      }
      this.buf.push(vec);
      if (this.buf.length > this.window) this.buf.shift();
      const w = this.buf.length;

      const mu = new Array(n).fill(0);
      for (const row of this.buf) for (let j = 0; j < n; j++) mu[j] += row[j] / w;
      if (w < 3) return 0;
      const X = this.buf.map((row) => row.map((x, j) => x - mu[j]));

      // Power iteration on X^T X, warm-started from the last eigenvector.
      let v = this.v && this.v.length === n ? this.v.slice() : new Array(n).fill(1 / Math.sqrt(n));
      let lambda = 0;
      for (let it = 0; it < this.iterations; it++) {
        const xv = X.map((row) => dot(row, v));
        const next = new Array(n).fill(0);
        for (let r = 0; r < w; r++) for (let j = 0; j < n; j++) next[j] += X[r][j] * xv[r];
        const norm = Math.sqrt(dot(next, next));
        if (!norm) break;
        lambda = norm;
        for (let j = 0; j < n; j++) next[j] /= norm;
        v = next;
      }
      // Keep the sign stable between samples.
      if (this.v && this.v.length === n && dot(v, this.v) < 0) v = v.map((x) => -x);
      this.v = v;

      let total = 0;
      for (const row of X) total += dot(row, row);
      this.explained = total ? lambda / total : 0;

      return dot(X[w - 1], v);
    }
  }

  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  /**
   * Cleans one link. Accepts either a scalar amplitude or
   * { amp: number[], phase?: number[], index?: number[] }.
   */
  class LinkProcessor {
    constructor(opts) {
      this.opts = opts;
      this.reset();
    }

    reset() {
      const o = this.opts;
      this.scalarHampel = new HampelFilter(o.hampelWindow, o.hampelSigma);
      this.vectorHampel = new VectorHampel(o.hampelWindow, o.hampelSigma);
      this.pca = new PC1Tracker(o.pcaWindow);
    }

    process(input) {
      if (isNum(input)) {
        const f = this.scalarHampel.push(input);
        return { signal: f, meanAmp: f, phase: null, mode: 'scalar' };
      }
      if (input && Array.isArray(input.amp) && input.amp.length) {
        const filtered = this.vectorHampel.push(input.amp);
        const phase = Array.isArray(input.phase) && input.phase.length > 1 ? sanitizePhase(input.phase, input.index) : null;
        return {
          signal: filtered.length > 1 ? this.pca.push(filtered) : filtered[0],
          meanAmp: mean(filtered),
          phase,
          mode: 'subcarriers',
        };
      }
      return null;
    }

    get outliers() {
      return this.scalarHampel.outliers + this.vectorHampel.outliers;
    }

    get count() {
      return this.scalarHampel.count + this.vectorHampel.count;
    }
  }

  // ---------- zones and priors ----------

  const ZONES = ['EMPTY', 'DESK', 'BED', 'DOOR'];
  const LINKS = ['b2b', 'b1r', 'b2r'];

  const DEFAULT_PRIORS = {
    DESK: { sitting: 0.9, standing: 0.08, lying: 0.02 },
    BED: { lying: 0.5, sitting: 0.4, standing: 0.1 },
    DOOR: { moving: 0.6, standing: 0.4, sitting: 0.0, lying: 0.0 },
  };

  const DEFAULT_SETTINGS = {
    windowSize: 20, // samples used for rolling variance / attenuation
    hampelWindow: 11, // odd; output lags by half of this
    hampelSigma: 4, // ~1% false hits on plain noise; 7 / 3 gives ~5%
    pcaWindow: 20,
    // Defaults are tuned to the bundled simulator. Calibrate on real hardware.
    thresholds: {
      presenceVar: 5, // b2b variance above this = someone is in the room
      doorRouterVar: 80, // both router links above this ...
      doorB2BVarMax: 30, // ... and b2b below this (null = ignore b2b) = DOOR
      deskAttenMin: 6, // b2b attenuation at or above this = DESK
      bedAttenMin: 1.5, // b2b attenuation at or above this = BED
    },
    baselineB2BAmp: null, // empty-room b2b amplitude; set by calibrating EMPTY
    zoneHold: 5, // samples a new zone must persist before the zone switches
    smoothing: 0.35, // weight of the newest posterior (1 = no smoothing)
    priorFloor: 0.005, // stops a 0 prior from wiping out every pose
    priors: DEFAULT_PRIORS,
  };

  function clone(o) {
    return JSON.parse(JSON.stringify(o));
  }

  function normalize(dist) {
    const out = {};
    let sum = 0;
    for (const k in dist) {
      const v = dist[k];
      if (isNum(v) && v > 0) {
        out[k] = v;
        sum += v;
      }
    }
    if (!sum) return null;
    for (const k in out) out[k] /= sum;
    return out;
  }

  /**
   * posterior(pose) ∝ likelihood(pose) * prior(pose | zone)
   * - Raw model output is normalised first.
   * - A pose the zone prior knows but the model never outputs (e.g. "moving")
   *   gets the average raw likelihood, i.e. "the model has no opinion".
   * - Priors are floored so a 0 prior can't make the whole sum 0.
   */
  function applyBayesianPrior(prior, rawProbs, floor = 0.005) {
    const raw = normalize(rawProbs || {});
    const priorKeys = Object.keys(prior || {});
    if (!raw) {
      const p = normalize(Object.fromEntries(priorKeys.map((k) => [k, Math.max(prior[k], floor)])));
      return finish(p || {});
    }
    if (!priorKeys.length) return finish(raw);

    const rawKeys = Object.keys(raw);
    const neutral = 1 / rawKeys.length;
    const poses = new Set([...rawKeys, ...priorKeys]);
    const post = {};
    for (const pose of poses) {
      const lik = pose in raw ? raw[pose] : neutral;
      const pr = Math.max(isNum(prior[pose]) ? prior[pose] : 0, floor);
      post[pose] = lik * pr;
    }
    return finish(normalize(post) || raw);
  }

  function finish(distribution) {
    let pose = 'unknown';
    let confidence = 0;
    for (const k in distribution) {
      if (distribution[k] > confidence) {
        confidence = distribution[k];
        pose = k;
      }
    }
    return { pose, confidence, distribution };
  }

  /**
   * Zone rules. `f` = { vB2B, vB1R, vB2R, att } where att may be null
   * (no empty-room baseline yet: DESK and BED can't be told apart).
   */
  function classifyZone(f, t) {
    const doorB2BOk = t.doorB2BVarMax == null || f.vB2B < t.doorB2BVarMax;
    if (f.vB1R > t.doorRouterVar && f.vB2R > t.doorRouterVar && doorB2BOk) return 'DOOR';
    if (f.att == null) return f.vB2B > t.presenceVar ? 'DESK' : 'EMPTY';
    if (f.att >= t.deskAttenMin) return 'DESK';
    if (f.att >= t.bedAttenMin || f.vB2B > t.presenceVar) return 'BED';
    return 'EMPTY';
  }

  /**
   * Turn per-zone calibration recordings into thresholds.
   * calib[zone] = { vB2B, vB1R, vB2R, b2bAmp } (medians over the recording)
   */
  function deriveThresholds(calib, current) {
    const t = Object.assign({}, current.thresholds);
    const warnings = [];
    let baseline = current.baselineB2BAmp;
    const E = calib.EMPTY;
    const D = calib.DESK;
    const B = calib.BED;
    const R = calib.DOOR;

    if (E) baseline = E.b2bAmp;
    else warnings.push('Record EMPTY first: it sets the baseline used to tell DESK from BED.');
    const att = (z) => (baseline == null || !z ? null : baseline - z.b2bAmp);

    if (E) {
      const occupied = [D, B].filter(Boolean).map((z) => z.vB2B);
      if (occupied.length) {
        const lo = Math.min(...occupied);
        if (lo > E.vB2B) t.presenceVar = geoMid(E.vB2B, lo);
        else {
          t.presenceVar = E.vB2B * 2;
          warnings.push('B2B variance with someone in the room is not above the empty room. Presence relies on attenuation only.');
        }
      } else t.presenceVar = Math.max(E.vB2B * 3, 1e-6);
    }

    if (E && B) {
      const a = att(B);
      if (a > 0) t.bedAttenMin = a / 2;
      else warnings.push('Lying on the bed does not attenuate the B2B link. BED is detected from movement only.');
    }

    if (D && B && baseline != null) {
      const aD = att(D);
      const aB = att(B);
      if (aD > aB) t.deskAttenMin = (aD + aB) / 2;
      else warnings.push('Sitting at the desk does not block the B2B link more than the bed. DESK and BED can\'t be separated by attenuation; try moving a board.');
    } else if (D && baseline != null && att(D) > 0) {
      t.deskAttenMin = Math.max(att(D) / 2, t.bedAttenMin + 1e-6);
    }

    if (R) {
      const others = [E, D, B].filter(Boolean);
      if (others.length) {
        const nonDoorMax = Math.max(...others.map((z) => Math.max(z.vB1R, z.vB2R)));
        const doorMin = Math.min(R.vB1R, R.vB2R);
        if (doorMin > nonDoorMax) t.doorRouterVar = geoMid(doorMin, nonDoorMax);
        else warnings.push('Walking through the door does not disturb both router links more than the other zones. Door detection will be unreliable.');

        const occ = [D, B].filter(Boolean);
        if (occ.length) {
          const occMax = Math.max(...occ.map((z) => z.vB2B));
          if (R.vB2B < occMax) t.doorB2BVarMax = geoMid(R.vB2B, occMax);
          else {
            t.doorB2BVarMax = null;
            warnings.push('The door also disturbs the B2B link. The B2B check for DOOR is turned off; router links decide alone.');
          }
        }
      }
    }

    if (t.deskAttenMin <= t.bedAttenMin) {
      warnings.push('Desk attenuation threshold is not above the bed one; DESK will win every tie.');
    }
    return { thresholds: t, baselineB2BAmp: baseline, warnings };
  }

  // ---------- optimizer ----------

  class SpatialPoseOptimizer {
    constructor(settings) {
      this.settings = clone(DEFAULT_SETTINGS);
      this.calibration = {};
      this.lastWarnings = [];
      this.setSettings(settings || {});
      this.reset();
    }

    setSettings(partial) {
      const s = this.settings;
      for (const k in partial) {
        if (k === 'thresholds' || k === 'priors') s[k] = Object.assign({}, s[k], clone(partial[k]));
        else if (k in DEFAULT_SETTINGS) s[k] = partial[k];
      }
      const rebuild = ['hampelWindow', 'hampelSigma', 'pcaWindow'].some((k) => k in partial);
      if (rebuild && this.links) this.reset();
    }

    getSettings() {
      return clone(this.settings);
    }

    reset() {
      const s = this.settings;
      this.links = {};
      this.windows = {};
      for (const l of LINKS) {
        this.links[l] = new LinkProcessor(s);
        this.windows[l] = [];
      }
      this.ampWindow = [];
      this.zone = null;
      this.pendingZone = null;
      this.pendingCount = 0;
      this.smoothed = null;
      this.recording = null;
      this.samples = 0;
    }

    /** Begin recording `samples` messages as calibration for `zone`. */
    startCalibration(zone, samples = 100) {
      if (!ZONES.includes(zone)) throw new Error('Unknown zone ' + zone);
      this.recording = { zone, settle: this.settings.windowSize, remaining: samples, total: samples, feats: [] };
    }

    cancelCalibration() {
      this.recording = null;
    }

    get calibrationProgress() {
      const r = this.recording;
      if (!r) return null;
      return { zone: r.zone, done: r.total - r.remaining, total: r.total, settling: r.settle > 0 };
    }

    clearCalibration() {
      this.calibration = {};
      this.lastWarnings = [];
    }

    /** Re-derive thresholds from all recorded zones and apply them. */
    applyCalibration() {
      const r = deriveThresholds(this.calibration, this.settings);
      this.settings.thresholds = r.thresholds;
      this.settings.baselineB2BAmp = r.baselineB2BAmp;
      this.lastWarnings = r.warnings;
      return r;
    }

    features(msg) {
      const s = this.settings;
      const out = { links: {} };
      for (const l of LINKS) {
        const input = msg[l] !== undefined ? msg[l] : msg[l + '_amp'];
        if (input === undefined || input === null) continue;
        const p = this.links[l].process(input);
        if (!p) continue;
        const w = this.windows[l];
        w.push(p.signal);
        if (w.length > s.windowSize) w.shift();
        if (l === 'b2b') {
          this.ampWindow.push(p.meanAmp);
          if (this.ampWindow.length > s.windowSize) this.ampWindow.shift();
        }
        out.links[l] = { mode: p.mode, phase: p.phase, outliers: this.links[l].outliers, values: this.links[l].count };
      }
      out.vB2B = variance(this.windows.b2b);
      out.vB1R = variance(this.windows.b1r);
      out.vB2R = variance(this.windows.b2r);
      out.b2bAmp = mean(this.ampWindow);
      if (isNum(msg.b2b_mean_attenuation)) out.att = msg.b2b_mean_attenuation;
      else if (s.baselineB2BAmp != null && this.ampWindow.length) out.att = s.baselineB2BAmp - out.b2bAmp;
      else out.att = null;
      return out;
    }

    /**
     * @param {object} msg
     *   b2b, b1r, b2r: number (amplitude) or { amp: [], phase?: [], index?: [] }
     *   (b2b_amp / b1r_amp / b2r_amp are accepted too)
     *   raw_pose_probs: { sitting: 0.2, standing: 0.7, ... }
     *   b2b_mean_attenuation: optional, overrides the computed attenuation
     */
    update(msg) {
      const s = this.settings;
      this.samples++;
      const f = this.features(msg || {});

      let calibrationDone = null;
      const r = this.recording;
      if (r) {
        if (r.settle > 0) r.settle--;
        else {
          r.feats.push({ vB2B: f.vB2B, vB1R: f.vB1R, vB2R: f.vB2R, b2bAmp: f.b2bAmp });
          if (--r.remaining <= 0) {
            const summary = { n: r.feats.length };
            for (const k of ['vB2B', 'vB1R', 'vB2R', 'b2bAmp']) summary[k] = median(r.feats.map((x) => x[k]));
            this.calibration[r.zone] = summary;
            this.recording = null;
            calibrationDone = r.zone;
            this.applyCalibration();
          }
        }
      }

      const candidate = classifyZone(f, s.thresholds);
      if (this.zone === null || candidate === this.zone) {
        this.zone = candidate;
        this.pendingZone = null;
        this.pendingCount = 0;
      } else {
        if (candidate === this.pendingZone) this.pendingCount++;
        else {
          this.pendingZone = candidate;
          this.pendingCount = 1;
        }
        if (this.pendingCount >= s.zoneHold) {
          this.zone = candidate;
          this.pendingZone = null;
          this.pendingCount = 0;
        }
      }

      const raw = normalize(msg && msg.raw_pose_probs) || {};
      let result;
      if (this.zone === 'EMPTY') {
        this.smoothed = null;
        result = { pose: 'none', confidence: null, distribution: {} };
      } else {
        const post = applyBayesianPrior(s.priors[this.zone], raw, s.priorFloor).distribution;
        this.smoothed = ema(this.smoothed, post, s.smoothing);
        result = finish(this.smoothed);
      }

      return {
        zone: this.zone,
        candidateZone: candidate,
        pose: result.pose,
        confidence: result.confidence,
        distribution: result.distribution,
        rawDistribution: raw,
        features: f,
        attenuationKnown: f.att != null,
        calibrationDone,
      };
    }
  }

  function ema(prev, next, alpha) {
    if (!prev) return Object.assign({}, next);
    const out = {};
    const keys = new Set([...Object.keys(prev), ...Object.keys(next)]);
    for (const k of keys) out[k] = alpha * (next[k] || 0) + (1 - alpha) * (prev[k] || 0);
    return normalize(out) || next;
  }

  return {
    SpatialPoseOptimizer,
    HampelFilter,
    VectorHampel,
    PC1Tracker,
    LinkProcessor,
    sanitizePhase,
    unwrapPhase,
    applyBayesianPrior,
    classifyZone,
    deriveThresholds,
    ZONES,
    DEFAULT_SETTINGS,
    DEFAULT_PRIORS,
    _util: { mean, median, variance },
  };
});
