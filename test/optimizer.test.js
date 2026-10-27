const test = require('node:test');
const assert = require('node:assert/strict');
const SP = require('../ui/dash/spatial-pose-optimizer.js');
const CsiSimulator = require('../ui/dash/csi-simulator.js');

const { SpatialPoseOptimizer, HampelFilter, PC1Tracker, sanitizePhase, applyBayesianPrior, classifyZone, deriveThresholds } = SP;

test('sanitizePhase removes a wrapped linear CFO/SFO ramp', () => {
  const n = 64;
  const motion = Array.from({ length: n }, (_, k) => 0.2 * Math.sin(k / 5));
  const wrapped = motion.map((m, k) => {
    const p = 1.3 + 0.4 * k + m;
    return Math.atan2(Math.sin(p), Math.cos(p));
  });
  const clean = sanitizePhase(wrapped);
  // What is left should be the motion term minus its own linear fit: small.
  const maxAbs = Math.max(...clean.map(Math.abs));
  assert.ok(maxAbs < 0.3, `residual too large: ${maxAbs}`);
  // And it must not be a ramp any more.
  assert.ok(Math.abs(clean[n - 1] - clean[0]) < 0.4);
});

test('Hampel filter flattens a spike but keeps small movement', () => {
  const h = new HampelFilter(7, 3);
  const out = [];
  for (let i = 0; i < 30; i++) {
    const x = 10 + 0.5 * Math.sin(i / 2) + (i === 15 ? 50 : 0);
    out.push(h.push(x));
  }
  // Output lags by 3 samples (window 7), so input i comes out at i + 3.
  assert.ok(out[18] < 12, `spike survived: ${out[18]}`);
  assert.ok(Math.abs(out[23] - (10 + 0.5 * Math.sin(10))) < 1e-9, 'normal sample changed');
  assert.equal(h.outliers, 1, 'smooth movement was flagged as a spike');
});

test('PC1 picks up the subcarriers that move together', () => {
  const pca = new PC1Tracker(30);
  const n = 16;
  const pattern = Array.from({ length: n }, (_, k) => (k < 4 ? 1 : 0));
  let last = 0;
  const signals = [];
  for (let t = 0; t < 60; t++) {
    const s = Math.sin(t / 3) * 5;
    const vec = pattern.map((p, k) => 20 + p * s + 0.01 * Math.cos(t * k));
    last = pca.push(vec);
    signals.push(last);
  }
  const top = pca.v.map(Math.abs);
  assert.ok(top.slice(0, 4).every((x) => x > 0.4), 'moving subcarriers not dominant');
  assert.ok(top.slice(4).every((x) => x < 0.05), 'static subcarriers leaked in');
  assert.ok(pca.explained > 0.95);
});

test('Bayes prior suppresses a hallucinated "standing" at the desk', () => {
  const r = applyBayesianPrior(SP.DEFAULT_PRIORS.DESK, { sitting: 0.3, standing: 0.6, lying: 0.1 });
  assert.equal(r.pose, 'sitting');
  const sum = Object.values(r.distribution).reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9);
});

test('DOOR prior never divides by zero, even when the model has no "moving" pose', () => {
  const r = applyBayesianPrior(SP.DEFAULT_PRIORS.DOOR, { sitting: 1, lying: 0 });
  for (const v of Object.values(r.distribution)) assert.ok(Number.isFinite(v));
  assert.notEqual(r.pose, 'unknown');
});

test('missing or empty raw probabilities fall back to the prior', () => {
  assert.equal(applyBayesianPrior(SP.DEFAULT_PRIORS.DESK, {}).pose, 'sitting');
  assert.equal(applyBayesianPrior(SP.DEFAULT_PRIORS.BED, undefined).pose, 'lying');
});

test('classifyZone follows the rules', () => {
  const t = SP.DEFAULT_SETTINGS.thresholds;
  assert.equal(classifyZone({ vB2B: 1, vB1R: 500, vB2R: 500, att: 0 }, t), 'DOOR');
  assert.equal(classifyZone({ vB2B: 1, vB1R: 500, vB2R: 500, att: 0 }, { ...t, doorB2BVarMax: 0.5 }), 'EMPTY');
  assert.equal(classifyZone({ vB2B: 1, vB1R: 500, vB2R: 500, att: 0 }, { ...t, doorB2BVarMax: null }), 'DOOR');
  assert.equal(classifyZone({ vB2B: 50, vB1R: 1, vB2R: 1, att: 9 }, t), 'DESK');
  assert.equal(classifyZone({ vB2B: 50, vB1R: 1, vB2R: 1, att: 3 }, t), 'BED');
  assert.equal(classifyZone({ vB2B: 0.1, vB1R: 0.1, vB2R: 0.1, att: 0 }, t), 'EMPTY');
  assert.equal(classifyZone({ vB2B: 50, vB1R: 1, vB2R: 1, att: null }, t), 'DESK');
});

test('zone hysteresis ignores a short blip', () => {
  const o = new SpatialPoseOptimizer({ zoneHold: 5, windowSize: 1 });
  o.setSettings({ thresholds: { presenceVar: 0 } });
  // Scalar input with b2b_mean_attenuation drives the zone directly.
  const msg = (att) => ({ b2b: 30, b1r: 1, b2r: 1, b2b_mean_attenuation: att, raw_pose_probs: { sitting: 1 } });
  for (let i = 0; i < 10; i++) o.update(msg(10));
  assert.equal(o.zone, 'DESK');
  for (let i = 0; i < 3; i++) o.update(msg(3));
  assert.equal(o.zone, 'DESK', 'switched too early');
  for (let i = 0; i < 5; i++) o.update(msg(3));
  assert.equal(o.zone, 'BED');
});

test('accepts the old scalar message format (b2b_amp, b1r_amp, b2r_amp)', () => {
  const o = new SpatialPoseOptimizer();
  let r;
  for (let i = 0; i < 30; i++) r = o.update({ b2b_amp: 30 + Math.sin(i), b1r_amp: 20, b2r_amp: 20, raw_pose_probs: { sitting: 0.5, standing: 0.5 } });
  assert.ok(r.zone);
  assert.ok(Number.isFinite(r.features.vB2B));
});

test('deriveThresholds warns when desk and bed cannot be separated', () => {
  const calib = {
    EMPTY: { vB2B: 0.4, vB1R: 0.4, vB2R: 0.4, b2bAmp: 30 },
    DESK: { vB2B: 80, vB1R: 5, vB2R: 5, b2bAmp: 28 },
    BED: { vB2B: 120, vB1R: 3, vB2R: 3, b2bAmp: 25 },
  };
  const r = deriveThresholds(calib, SP.DEFAULT_SETTINGS);
  assert.ok(r.warnings.some((w) => w.includes('DESK and BED')));
});

test('end to end: calibrate on the simulator, then zones and poses are right', () => {
  const sim = new CsiSimulator({ seed: 7 });
  const o = new SpatialPoseOptimizer();
  for (const z of ['EMPTY', 'DESK', 'BED', 'DOOR']) {
    sim.zone = z;
    o.startCalibration(z, 120);
    let r;
    do r = o.update(sim.next());
    while (!r.calibrationDone);
    assert.equal(r.calibrationDone, z);
  }
  assert.deepEqual(o.lastWarnings, []);

  for (const z of ['DESK', 'BED', 'EMPTY', 'DOOR']) {
    sim.zone = z;
    let zoneHits = 0;
    let poseHits = 0;
    let rawHits = 0;
    const n = 250;
    for (let i = 0; i < n + 40; i++) {
      const m = sim.next();
      const r = o.update(m);
      if (i < 40) continue; // let windows and hysteresis settle
      if (r.zone === z) zoneHits++;
      if (r.pose === (m.truth.pose || 'none')) poseHits++;
      const rawTop = Object.entries(m.raw_pose_probs).sort((a, b) => b[1] - a[1])[0][0];
      if (rawTop === m.truth.pose) rawHits++;
    }
    assert.ok(zoneHits / n > 0.9, `${z}: zone accuracy ${zoneHits / n}`);
    if (z === 'DESK' || z === 'BED') {
      assert.ok(poseHits > rawHits, `${z}: optimizer (${poseHits}) not better than raw model (${rawHits})`);
    }
  }
});
