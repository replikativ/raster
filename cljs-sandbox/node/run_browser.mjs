// Drives Chromium over the built sandbox and reports WebGPU/WebGL results.
// Serves public/ on localhost (WebGPU needs a secure context — localhost qualifies),
// loads the page, waits for the experiments, prints console + #log text + a PNG.
//
//   node node/run_browser.mjs            # headful on DISPLAY=:0 (real GPU → WebGPU)
//   HEADLESS=1 node node/run_browser.mjs # headless (WebGPU likely unavailable)
//
// Needs puppeteer-core + a Chrome binary (uses CHROME_PATH or the puppeteer cache).

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join } from "node:path";
import puppeteer from "puppeteer-core";

const ROOT = new URL("../public/", import.meta.url).pathname;
const PORT = 8123;
const MIME = { ".html": "text/html", ".js": "text/javascript", ".wasm": "application/wasm",
               ".css": "text/css", ".map": "application/json" };

const server = createServer(async (req, res) => {
  try {
    let p = decodeURIComponent(new URL(req.url, `http://localhost:${PORT}`).pathname);
    if (p === "/") p = "/index.html";
    const body = await readFile(join(ROOT, p));
    res.writeHead(200, { "content-type": MIME[extname(p)] ?? "application/octet-stream" });
    res.end(body);
  } catch { res.writeHead(404); res.end("not found"); }
});

const CHROME = process.env.CHROME_PATH
  ?? "/home/christian-weilbach/.cache/puppeteer/chrome/linux-148.0.7778.97/chrome-linux64/chrome";

async function main() {
  await new Promise((r) => server.listen(PORT, r));
  console.log(`serving public/ at http://localhost:${PORT}`);

  const headless = !!process.env.HEADLESS;
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless,
    args: [
      "--no-sandbox",
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan,WebGPU",
      "--use-angle=vulkan",
      // headless software fallback if no real adapter:
      ...(headless ? ["--use-gl=angle", "--use-angle=swiftshader"] : []),
    ],
  });
  try {
    const page = await browser.newPage();
    const logs = [];
    page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
    page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));

    // Report adapter availability directly.
    await page.goto(`http://localhost:${PORT}/`, { waitUntil: "networkidle0", timeout: 30000 });
    const gpuPresent = await page.evaluate(() => !!navigator.gpu);
    const adapter = await page.evaluate(async () =>
      navigator.gpu ? !!(await navigator.gpu.requestAdapter()) : false);

    // Give the cljs experiments time to run + read back async GPU results.
    await new Promise((r) => setTimeout(r, 2500));
    const logText = await page.evaluate(() => {
      const el = document.getElementById("log");
      return el ? el.textContent : "(no #log element)";
    });
    await page.screenshot({ path: "/tmp/sandbox_screenshot.png" });

    console.log(`\nnavigator.gpu present: ${gpuPresent}`);
    console.log(`requestAdapter ok:     ${adapter}`);
    console.log(`\n--- page #log ---\n${logText}`);
    console.log(`\n--- console (${logs.length}) ---`);
    for (const l of logs) console.log(l);
    console.log(`\nscreenshot: /tmp/sandbox_screenshot.png`);
  } finally {
    await browser.close();
    server.close();
  }
}
main().catch((e) => { console.error(e); server.close(); process.exit(1); });
