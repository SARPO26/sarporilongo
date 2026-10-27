/*
 * Fake CSI feed for trying the dashboard without hardware.
 * Produces the same message shape the optimizer expects:
 *   { b2b: {amp, phase}, b1r: {...}, b2r: {...}, raw_pose_probs }
 * Effects per zone are rough guesses; real rooms will differ, which is
 * what calibration is for.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.CsiSimulator = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // How each zone affects each link.
  // atten: drop in b2b amplitude. motion: size of movement on the link.
  const ZONE_EFFECTS = {
    EMPTY: { atten: 0, b2b: 0, router: 0 },
    DESK: { atten: 9, b2b: 1.0, router: 0.3 },
    BED: { atten: 3.5, b2b: 1.4, router: 0.3 },
    DOOR: { atten: 0.5, b2b: 0.4, router: 3.0 },
  };

  // The pose a perfect model would report in each zone.
  const TRUE_POSE = { DESK: 'sitting', BED: 'lying', DOOR: 'standing', EMPTY: null };
  const POSES = ['sitting', 'standing', 'lying'];

  class CsiSimulator {
    constructor(opts = {}) {
      this.nSub = opts.subcarriers || 64;
      this.zone = opts.zone || 'EMPTY';
      this.hallucination = opts.hallucination ?? 0.3; // chance the model picks a wrong pose
      this.spikeRate = opts.spikeRate ?? 0.02; // chance of an interference spike per link
      this.noise = opts.noise ?? 0.3;
      this.rand = mulberry32(opts.seed ?? 1);
      this.t = 0;
      this.links = {};
      for (const l of ['b2b', 'b1r', 'b2r']) {
        this.links[l] = {
          base: this.smoothProfile(30, 8),
          pattern: this.smoothProfile(0, 1),
          walk: 0,
        };
      }
    }

    gauss() {
      const u = 1 - this.rand();
      const v = this.rand();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    smoothProfile(center, spread) {
      const a = this.rand() * 2 * Math.PI;
      const b = this.rand() * 2 * Math.PI;
      return Array.from({ length: this.nSub }, (_, k) => center + spread * (Math.sin(k / 7 + a) + 0.5 * Math.sin(k / 3 + b)));
    }

    linkSample(name, motion, atten) {
      const L = this.links[name];
      // Bounded random walk = person moving.
      L.walk = 0.9 * L.walk + motion * this.gauss();
      const breathing = motion > 0 ? 0.3 * motion * Math.sin(this.t / 8) : 0;
      const amp = L.base.map((b, k) => b - atten + (L.walk + breathing) * L.pattern[k] + this.noise * this.gauss());
      if (this.rand() < this.spikeRate) amp[(this.rand() * this.nSub) | 0] += 40;

      // Phase with a random CFO slope and offset every packet.
      const slope = (this.rand() - 0.5) * 0.6;
      const offset = this.rand() * 2 * Math.PI;
      const phase = amp.map((_, k) => {
        const p = offset + slope * k + 0.05 * L.walk * L.pattern[k];
        return Math.atan2(Math.sin(p), Math.cos(p));
      });
      return { amp, phase };
    }

    poseProbs() {
      const truth = TRUE_POSE[this.zone];
      if (!truth) {
        return Object.fromEntries(POSES.map((p) => [p, 1 / POSES.length]));
      }
      let picked = truth;
      if (this.rand() < this.hallucination) picked = POSES[(this.rand() * POSES.length) | 0];
      const probs = {};
      let sum = 0;
      for (const p of POSES) {
        probs[p] = (p === picked ? 3 : 0.5) + this.rand();
        sum += probs[p];
      }
      for (const p of POSES) probs[p] /= sum;
      return probs;
    }

    next() {
      this.t++;
      const e = ZONE_EFFECTS[this.zone];
      return {
        b2b: this.linkSample('b2b', e.b2b, e.atten),
        b1r: this.linkSample('b1r', e.router, 0),
        b2r: this.linkSample('b2r', e.router, 0),
        raw_pose_probs: this.poseProbs(),
        truth: { zone: this.zone, pose: TRUE_POSE[this.zone] },
      };
    }
  }

  CsiSimulator.ZONE_EFFECTS = ZONE_EFFECTS;
  CsiSimulator.TRUE_POSE = TRUE_POSE;
  return CsiSimulator;
});
