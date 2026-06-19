import { createServer } from "node:http"; import { readFile } from "node:fs/promises";
import { join, extname } from "node:path"; import puppeteer from "puppeteer-core";
const ROOT=new URL("../public/",import.meta.url).pathname,PORT=8136;
const server=createServer(async(req,res)=>{try{let p=new URL(req.url,"http://x").pathname;if(p==="/")p="/index.html";const b=await readFile(join(ROOT,p));res.writeHead(200,{"content-type":{".wasm":"application/wasm",".html":"text/html"}[extname(p)]??"application/octet-stream"});res.end(b);}catch{res.writeHead(404);res.end();}});
await new Promise(r=>server.listen(PORT,r));
const b=await puppeteer.launch({executablePath:"/home/christian-weilbach/.cache/puppeteer/chrome/linux-148.0.7778.97/chrome-linux64/chrome",headless:true,args:["--no-sandbox"]});
const pg=await b.newPage(); await pg.goto(`http://localhost:${PORT}/index.html`,{waitUntil:"load"});
const r=await pg.evaluate(async()=>{const bytes=await (await fetch("/cadd_jvm.wasm")).arrayBuffer();const valid=WebAssembly.validate(bytes);const {instance}=await WebAssembly.instantiate(bytes,{});const {cadd,memory}=instance.exports;const n=64;const h=new Float64Array(memory.buffer);
  for(let i=0;i<n;i++){h[i]=i;h[n+i]=2*i;h[2*n+i]=10;h[3*n+i]=100;}
  cadd(0,n*8,2*n*8,3*n*8,4*n*8,5*n*8,n);
  let bad=0;for(let i=0;i<n;i++){if(h[4*n+i]!==i+10||h[5*n+i]!==2*i+100)bad++;}
  return {valid,bad,ok:valid&&bad===0};});
console.log("real Chrome cadd-SoA:",JSON.stringify(r),r.ok?"=> PASS":"=> FAIL"); await b.close(); server.close();
