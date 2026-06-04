export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  const W = 1280, H = 720;
  const bg = "#F3F6FB";
  const blue = "#1557C0";
  const dark = "#17324D";
  const orange = "#D97706";
  const magenta = "#C026D3";
  const green = "#14803C";
  const border = "#C9D7EE";
  const soft = "#EAF1FB";
  const text = "#22324A";
  slide.background.fill = { color: bg };

  ctx.addText(slide, { x: 32, y: 18, w: 620, h: 34, text: "GB10 ds4 Performance Update", fontSize: 24, bold: true, color: dark, face: ctx.fonts.title });
  ctx.addText(slide, { x: 940, y: 18, w: 300, h: 24, text: "Measured on GB10 · 273 GB/s · 123 TOPS", fontSize: 12, color: "#4A6483", align: "right" });

  const topY = 56, panelH = 390, gap = 16;
  const panelW = (W - 32*2 - gap*2)/3;
  const xs = [32, 32 + panelW + gap, 32 + 2*(panelW + gap)];

  function panel(x, title, pill, pillColor) {
    ctx.addShape(slide, { x, y: topY, w: panelW, h: panelH, fill: { color: "#FFFFFF" }, line: ctx.line(border, 1) });
    ctx.addShape(slide, { x, y: topY, w: panelW, h: 46, fill: { color: blue }, line: ctx.line(blue, 1) });
    ctx.addText(slide, { x: x + 14, y: topY + 9, w: panelW - 130, h: 28, text: title, fontSize: 16, bold: true, color: "#FFFFFF" });
    ctx.addShape(slide, { x: x + panelW - 108, y: topY + 7, w: 92, h: 30, fill: { color: pillColor }, line: ctx.line(pillColor, 1) });
    ctx.addText(slide, { x: x + panelW - 108, y: topY + 9, w: 92, h: 22, text: pill, fontSize: 12, bold: true, color: "#FFFFFF", align: "center", valign: "mid" });
  }

  panel(xs[0], "① Cold Boot: Model Load", "Load", orange);
  panel(xs[1], "② Prefill: Resident BW + Context Compute", "Prefill", magenta);
  panel(xs[2], "③ Decode: UMA Weights + Long Context", "Decode", green);

  // Load panel
  let x = xs[0] + 16;
  let y = topY + 58;
  ctx.addText(slide, { x, y, w: panelW - 32, h: 24, text: "Total model: 80.76 GB  (8.16 GB dense + 72.6 GB experts)", fontSize: 14, bold: true, color: dark });
  y += 40;
  ctx.addText(slide, { x, y, w: panelW - 32, h: 24, text: "PCIe 6.0 x4: 32 GiB/s  vs  PCIe 5.0 x4: 15.75 GiB/s  →  2.03× faster", fontSize: 13, color: text });
  y += 34;
  ctx.addText(slide, { x, y, w: panelW - 32, h: 24, text: "IO BW utilization: 56%  (measured on GB10)", fontSize: 13, color: text });
  y += 36;
  ctx.addShape(slide, { x, y, w: panelW - 32, h: 96, fill: { color: soft }, line: ctx.line(border, 1) });
  ctx.addText(slide, { x: x + 10, y: y + 8, w: panelW - 52, h: 18, text: "T = Model_Size / (BW × Utilization)", fontSize: 12, bold: true, color: dark });
  ctx.addText(slide, { x: x + 10, y: y + 34, w: panelW - 52, h: 18, text: "PCIe 6.0 x4: 80.76 / (32 × 0.56) = 4.51 s", fontSize: 12, color: text });
  ctx.addText(slide, { x: x + 10, y: y + 56, w: panelW - 52, h: 18, text: "PCIe 5.0 x4: 80.76 / (15.75 × 0.56) = 9.16 s", fontSize: 12, color: text });
  ctx.addText(slide, { x: x + 10, y: y + 78, w: panelW - 52, h: 18, text: "Speedup = 9.16 / 4.51 = 2.03×", fontSize: 12, bold: true, color: green });
  y += 114;
  ctx.addText(slide, { x, y, w: panelW - 32, h: 20, text: "Updated conclusion", fontSize: 13, bold: true, color: dark });
  y += 22;
  ctx.addText(slide, { x, y, w: panelW - 32, h: 66, text: "Cold Boot is SSD/NVMe → UMA/DDR loading. With measured 56% IO efficiency, PCIe 6.0 x4 reduces startup from 9.16 s to 4.51 s.", fontSize: 12, color: text, insets: { left: 0, right: 0, top: 0, bottom: 0 } });

  // Prefill panel
  x = xs[1] + 14; y = topY + 58;
  ctx.addShape(slide, { x, y, w: panelW - 28, h: 88, fill: { color: soft }, line: ctx.line(border, 1) });
  ctx.addText(slide, { x: x + 10, y: y + 8, w: panelW - 48, h: 16, text: "Calculation logic", fontSize: 12, bold: true, color: dark });
  ctx.addText(slide, { x: x + 10, y: y + 28, w: panelW - 48, h: 16, text: "η_mem,prefill = [ceil(P/2048) × 80.76 / prefill_s] / 273", fontSize: 11, color: text });
  ctx.addText(slide, { x: x + 10, y: y + 46, w: panelW - 48, h: 16, text: "m = mem / (mem + gpu),  c = gpu / (mem + gpu)", fontSize: 11, color: text });
  ctx.addText(slide, { x: x + 10, y: y + 64, w: panelW - 48, h: 16, text: "Pred(273/120)=Obs/(m+c×123/120); Pred(546/240)=Obs/(0.5m+c×123/240)", fontSize: 10, color: text });
  y += 98;
  // table header
  const colsP = [76, 96, 78, 72, 120, 120];
  const headersP = ["Context", "GB10 TPS", "Mem Util", "GPU Busy", "273GB/120T", "546GB/240T"];
  let tx = x;
  let rowY = y;
  headersP.forEach((h, i) => {
    ctx.addShape(slide, { x: tx, y: rowY, w: colsP[i], h: 32, fill: { color: blue }, line: ctx.line("#FFFFFF", 1) });
    ctx.addText(slide, { x: tx + 4, y: rowY + 6, w: colsP[i]-8, h: 20, text: h, fontSize: 11, bold: true, color: "#FFFFFF", align: "center", valign: "mid" });
    tx += colsP[i];
  });
  const preRows = [
    ["128", "148.98", "34.43%", "0.00%", "148.98", "297.96"],
    ["1,024", "379.51", "10.96%", "5.00%", "376.56", "753.12"],
    ["2,048", "396.40", "5.73%", "7.00%", "391.02", "782.05"],
    ["8,192", "390.78", "5.64%", "9.90%", "384.65", "769.31"],
    ["65,536", "336.37", "4.86%", "87.08%", "328.59", "657.18"],
    ["131,072", "290.63", "4.20%", "92.05%", "283.84", "567.69"],
  ];
  rowY += 32;
  preRows.forEach((r, ridx) => {
    let cx = x;
    const fill = ridx % 2 === 0 ? "#FFF9E6" : "#EAF4FF";
    r.forEach((v, i) => {
      ctx.addShape(slide, { x: cx, y: rowY, w: colsP[i], h: 24, fill: { color: fill }, line: ctx.line(border, 1) });
      ctx.addText(slide, { x: cx + 4, y: rowY + 4, w: colsP[i]-8, h: 16, text: v, fontSize: 10.5, color: i >= 4 ? green : text, bold: i >= 4, align: "center", valign: "mid" });
      cx += colsP[i];
    });
    rowY += 24;
  });
  rowY += 10;
  ctx.addText(slide, { x, y: rowY, w: panelW - 28, h: 18, text: "Updated conclusion", fontSize: 13, bold: true, color: dark });
  rowY += 20;
  ctx.addText(slide, { x, y: rowY, w: panelW - 28, h: 54, text: "Small-context Prefill is resident-bandwidth dominated, so 546 GB/s nearly doubles 128-token throughput. With 240T compute, long-context Prefill also scales strongly because bandwidth and context compute both improve.", fontSize: 11.5, color: text });

  // Decode panel
  x = xs[2] + 14; y = topY + 58;
  ctx.addShape(slide, { x, y, w: panelW - 28, h: 88, fill: { color: soft }, line: ctx.line(border, 1) });
  ctx.addText(slide, { x: x + 10, y: y + 8, w: panelW - 48, h: 16, text: "Calculation logic", fontSize: 12, bold: true, color: dark });
  ctx.addText(slide, { x: x + 10, y: y + 28, w: panelW - 48, h: 16, text: "Active weights/token = 10.97 GiB = 8.20 + 1.07 + 72.56×6/256", fontSize: 10.5, color: text });
  ctx.addText(slide, { x: x + 10, y: y + 46, w: panelW - 48, h: 16, text: "η_mem,decode = generation_tps × 10.97 / 273", fontSize: 11, color: text });
  ctx.addText(slide, { x: x + 10, y: y + 64, w: panelW - 48, h: 16, text: "Pred(273/120)=Obs/(m+c×123/120); Pred(546/240)=Obs/(0.5m+c×123/240)", fontSize: 10, color: text });
  y += 98;
  const colsD = [76, 96, 78, 72, 120, 120];
  const headersD = ["Context", "GB10 TPS", "Mem Util", "GPU Busy", "273GB/120T", "546GB/240T"];
  tx = x; rowY = y;
  headersD.forEach((h, i) => {
    ctx.addShape(slide, { x: tx, y: rowY, w: colsD[i], h: 32, fill: { color: blue }, line: ctx.line("#FFFFFF", 1) });
    ctx.addText(slide, { x: tx + 4, y: rowY + 6, w: colsD[i]-8, h: 20, text: h, fontSize: 11, bold: true, color: "#FFFFFF", align: "center", valign: "mid" });
    tx += colsD[i];
  });
  const decRows = [
    ["128", "16.12", "64.78%", "8.62%", "16.07", "32.15"],
    ["1,024", "15.26", "61.32%", "8.29%", "15.21", "30.43"],
    ["2,048", "14.58", "58.59%", "8.13%", "14.54", "29.07"],
    ["8,192", "14.36", "57.70%", "94.38%", "14.14", "28.28"],
    ["65,536", "12.25", "49.22%", "96.00%", "12.05", "24.10"],
    ["131,072", "10.73", "43.12%", "96.00%", "10.55", "21.10"],
  ];
  rowY += 32;
  decRows.forEach((r, ridx) => {
    let cx = x;
    const fill = ridx % 2 === 0 ? "#FFF9E6" : "#EAF4FF";
    r.forEach((v, i) => {
      ctx.addShape(slide, { x: cx, y: rowY, w: colsD[i], h: 24, fill: { color: fill }, line: ctx.line(border, 1) });
      ctx.addText(slide, { x: cx + 4, y: rowY + 4, w: colsD[i]-8, h: 16, text: v, fontSize: 10.5, color: i >= 4 ? green : text, bold: i >= 4, align: "center", valign: "mid" });
      cx += colsD[i];
    });
    rowY += 24;
  });
  rowY += 10;
  ctx.addText(slide, { x, y: rowY, w: panelW - 28, h: 18, text: "Updated conclusion", fontSize: 13, bold: true, color: dark });
  rowY += 20;
  ctx.addText(slide, { x, y: rowY, w: panelW - 28, h: 54, text: "Decode is dominated by UMA-weight and KV/state bandwidth. With 546 GB/s plus 240T, short-context Decode approaches 2x, while long-context Decode still benefits materially despite context-path overhead.", fontSize: 11.5, color: text });

  // Bottom summary
  const sumY = 462;
  ctx.addShape(slide, { x: 32, y: sumY, w: W-64, h: 226, fill: { color: "#FFFFFF" }, line: ctx.line(border, 1) });
  ctx.addShape(slide, { x: 32, y: sumY, w: W-64, h: 36, fill: { color: blue }, line: ctx.line(blue, 1) });
  ctx.addText(slide, { x: 44, y: sumY + 8, w: W-100, h: 20, text: "Performance Summary: GB10 Measured vs Next-Gen Memory/Compute Configs", fontSize: 15, bold: true, color: "#FFFFFF" });

  const sx = 44, sy = sumY + 48;
  const colsS = [170, 140, 140, 140, 120, 350];
  const headersS = ["Scenario", "GB10 (273/123T)", "Next A (273/120T)", "Next B (546/240T)", "Improvement", "Bottleneck"];
  tx = sx; rowY = sy;
  headersS.forEach((h, i) => {
    ctx.addShape(slide, { x: tx, y: rowY, w: colsS[i], h: 30, fill: { color: "#DDE9FB" }, line: ctx.line(border, 1) });
    ctx.addText(slide, { x: tx + 4, y: rowY + 6, w: colsS[i]-8, h: 18, text: h, fontSize: 11, bold: true, color: dark, align: "center", valign: "mid" });
    tx += colsS[i];
  });
  const sumRows = [
    ["Cold Boot", "9.16 s", "9.16 s", "9.16 s", "same", "Storage BW only (NVMe/PCIe path; UMA/compute unchanged)"],
    ["Prefill C=128", "148.98 tok/s", "148.98 tok/s", "297.96 tok/s", "2.00× @ 546/240T", "Resident UMA BW + fixed cost"],
    ["Prefill C=2,048 ★", "396.40 tok/s", "391.02 tok/s", "782.05 tok/s", "1.97× @ 546/240T", "Hybrid (resident BW + compute)"],
    ["Decode @128 ctx", "16.12 tok/s", "16.07 tok/s", "32.15 tok/s", "1.99× @ 546/240T", "UMA active-weight bandwidth"],
    ["Decode @8K ctx", "14.36 tok/s", "14.14 tok/s", "28.28 tok/s", "1.97× @ 546/240T", "Mixed (memory + context path)"],
    ["Decode @128K ctx", "10.73 tok/s", "10.55 tok/s", "21.10 tok/s", "1.97× @ 546/240T", "Long-context KV/state + compute path"],
  ];
  rowY += 30;
  sumRows.forEach((r, ridx) => {
    let cx = sx;
    const fill = ridx % 2 === 0 ? "#F8FBFF" : "#FFFDF7";
    r.forEach((v, i) => {
      ctx.addShape(slide, { x: cx, y: rowY, w: colsS[i], h: 26, fill: { color: fill }, line: ctx.line(border, 1) });
      ctx.addText(slide, { x: cx + 4, y: rowY + 5, w: colsS[i]-8, h: 16, text: v, fontSize: 10.2, color: i===4 ? green : text, bold: i===4, align: i===0 || i===5 ? "left" : "center", valign: "mid" });
      cx += colsS[i];
    });
    rowY += 26;
  });

  return slide;
}
