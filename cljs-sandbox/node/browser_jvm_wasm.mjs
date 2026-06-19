import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { join, extname } from "node:path";
import puppeteer from "puppeteer-core";
const ROOT = new URL("../public/", import.meta.url).pathname, PORT = 8131;
const MIME = {".html":"text/html",".wasm":"application/wasm",".js":"text/javascript"};
const server = createServer(async (req,res)=>{ try{ let p=new URL(req.url,`http://x`).pathname; if(p==="/")p="/index.html";
  const b=await readFile(join(ROOT,p)); res.writeHead(200,{"content-type":MIME[extname(p)]??"application/octet-stream"}); res.end(b);}catch{res.writeHead(404);res.end();}});
await new Promise(r=>server.listen(PORT,r));
const CHROME="/home/christian-weilbach/.cache/puppeteer/chrome/linux-148.0.7778.97/chrome-linux64/chrome";
const browser=await puppeteer.launch({executablePath:CHROME,headless:true,args:["--no-sandbox"]});
const page=await browser.newPage();
await page.goto(`http://localhost:${PORT}/index.html`,{waitUntil:"load"});
const result=await page.evaluate(async ()=>{
  const bytes=await (await fetch("/saxpy_jvm.wasm")).arrayBuffer();
  const valid=WebAssembly.validate(bytes);
  const {instance}=await WebAssembly.instantiate(bytes,{});
  const {saxpy,memory}=instance.exports; const n=1000,a=3.0;
  const heap=new Float64Array(memory.buffer);
  let maxErr=0; for(let i=0;i<n;i++){heap[i]=i;heap[n+i]=0.5*i;}
  saxpy(a,0,n*8,n);
  for(let i=0;i<n;i++) maxErr=Math.max(maxErr,Math.abs(heap[n+i]-(a*i+0.5*i)));
  return {valid, y7:heap[n+7], maxErr};
});
console.log("real Chrome:", JSON.stringify(result), result.valid&&result.maxErr<1e-9?"=> PASS":"=> FAIL");
await browser.close(); server.close();
