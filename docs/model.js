// Loading model in the browser, a port of analysis/loading.py for the website sliders.
// One simplification: curb at each window's peak hour is the mean over its 10 weekday
// half-hours (Python averages Erlang B over them instead), so numbers can differ a little.

const CurbModel = (() => {
  // Erlang B, with non-integer spaces interpolated between floor(c) and floor(c) + 1
  function erlangB(c, a) {
    if (a <= 0) return 0;
    const n = Math.floor(c);
    let b = 1;
    for (let k = 1; k <= n; k++) b = (a * b) / (k + a * b);
    const next = (a * b) / (n + 1 + a * b);
    return b + (c - n) * (next - b);
  }

  function spacesNeeded(a, target) {
    if (a <= 0) return 0;
    let c = 1;
    while (erlangB(c, a) >= target) c++;
    return c;
  }

  // params: window ("07-11"), demand ("low" | "high"), dwell (min), vacant (0-1),
  // convert (parking spaces per face turned into loading), convertWhere ("all" | "short"),
  // offPeak (share of the window's deliveries moved out of it), bikes (share moved to cargo bikes),
  // shiftWhere ("all" | "short"): which blocks the off-peak and cargo-bike shifts apply to
  function run(features, model, p) {
    const w = p.window.replace("-", "_");
    const [h0, h1] = model.windows[p.window];
    const peak = model.peak_hour[p.window];
    const keep = (1 - p.offPeak) * (1 - p.bikes);
    const space = model.space_m;

    // "short" targets come from the scenario with nothing changed: a face is short when its
    // mismatch index is over 0.25, a block when any of its faces is
    const targeted = (p.convertWhere === "short" && p.convert > 0) || (p.shiftWhere === "short" && keep < 1);
    const base = targeted ? run(features, model, { ...p, convert: 0, offPeak: 0, bikes: 0 }).byId : null;
    const shortBlocks = new Set();
    if (base) for (const r of base.values()) if (r.mi > 0.25) shortBlocks.add(r.block);

    const faces = features.map(f => {
      const q = f.properties;
      let loading = q[`loading_${w}`] || 0, parking = q[`parking_${w}`] || 0;
      const short = q[`short_term_${w}`] || 0;
      const eligible = p.convertWhere !== "short" || !base || base.get(q.id).mi > 0.25;
      const moved = eligible ? Math.min(p.convert * model.paid_space_m, parking) : 0;
      const paidMoved = Math.min(moved, q[`paid_${w}`] || 0);   // paid spaces are converted first
      loading += moved; parking -= moved;
      const shifted = p.shiftWhere !== "short" || !base || shortBlocks.has(q.block);
      return { id: q.id, block: q.block, deliv: q[`deliv_${p.demand}`] * (shifted ? keep : 1), moved, paidMoved,
               loading, short, parking, usable: loading + short + p.vacant * parking };
    });

    const blocks = new Map();
    for (const f of faces) {
      const b = blocks.get(f.block) || { deliv: 0, spaces: 0 };
      b.deliv += f.deliv; b.spaces += f.usable / space;
      blocks.set(f.block, b);
    }
    for (const b of blocks.values()) {
      const lam = b.deliv * model.hour_share[peak];
      b.p = erlangB(b.spaces, (lam * p.dwell) / 60);
      b.need = spacesNeeded((lam * p.dwell) / 60, model.target_p);
      // trucks in the whole window that find no space (peak-hour curb used for every hour)
      b.missedPerDeliv = 0;
      for (let h = h0; h < h1; h++) {
        const a = (b.deliv * model.hour_share[h] * p.dwell) / 60;
        b.missedPerDeliv += model.hour_share[h] * erlangB(b.spaces, a);
      }
    }

    const byId = new Map();
    for (const f of faces) {
      const b = blocks.get(f.block);
      const share = b.deliv > 0 ? f.deliv / b.deliv : 0;
      const required = b.need * space * share;
      const top = Math.max(required, f.usable);
      const mi = top > 0 ? (required - f.usable) / top : 0;
      // parking revenue lost a weekday: paid spaces given up x rate x paid hours the loading zone
      // takes x share of those hours a space would be occupied (1 - empty share)
      const revenue = (f.paidMoved / model.paid_space_m) * model.rate_per_hour * model.lost_hours * (1 - p.vacant);
      byId.set(f.id, { ...f, required, mi, p: b.p, need: b.need, missed: f.deliv * b.missedPerDeliv, revenue });
    }

    let short = 0, shortM = 0, missed = 0, moved = 0, revenue = 0;
    for (const r of byId.values()) {
      if (r.mi > 0.25) short++;
      shortM += Math.max(0, r.required - r.usable);
      missed += r.missed;
      moved += r.moved;
      revenue += r.revenue;
    }
    return { byId, summary: { short, shortM, missed, spacesMoved: moved / model.paid_space_m,
                              revenueDay: revenue, revenueYear: revenue * model.weekdays } };
  }

  return { erlangB, spacesNeeded, run };
})();

if (typeof module !== "undefined") module.exports = CurbModel;
