// Fit a rotated map to a set of [lng, lat] points. MapLibre's fitBounds fits the north-up
// box around them, which leaves a long thin area tiny once the map is turned to the street grid.
function fitRotated(map, points, bearing, pad) {
  const R = 6378137, rad = Math.PI / 180;
  const merc = ([lng, lat]) => [R * lng * rad, R * Math.log(Math.tan(Math.PI / 4 + (lat * rad) / 2))];
  const a = bearing * rad;                                  // screen = world turned by the bearing
  const xy = points.map(merc).map(([x, y]) => [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a)]);
  const xs = xy.map(p => p[0]), ys = xy.map(p => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const box = map.getContainer().getBoundingClientRect();
  const w = box.width - pad.left - pad.right, h = box.height - pad.top - pad.bottom;
  const res = Math.max((x1 - x0) / w, (y1 - y0) / h);     // metres per pixel needed
  const zoom = Math.log2(78271.517 / res);                // 512 px tiles
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;           // centre, turned back to the world
  const wx = cx * Math.cos(-a) - cy * Math.sin(-a), wy = cx * Math.sin(-a) + cy * Math.cos(-a);
  const center = [wx / R / rad, (2 * Math.atan(Math.exp(wy / R)) - Math.PI / 2) / rad];
  map.jumpTo({ center, zoom, bearing, padding: pad });
}
