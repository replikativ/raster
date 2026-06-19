import { createServer } from "node:http"; import { readFile } from "node:fs/promises";
import { join, extname } from "node:path"; import puppeteer from "puppeteer-core";
const ROOT=new URL("../public/",import.meta.url).pathname,PORT=8135;
const server=createServer(async(req,res)=>{try{let p=new URL(req.url,"http://x").pathname;if(p==="/")p="/index.html";const b=await readFile(join(ROOT,p));res.writeHead(200,{"content-type":{".wasm":"application/wasm",".html":"text/html"}[extname(p)]??"application/octet-stream"});res.end(b);}catch{res.writeHead(404);res.end();}});
await new Promise(r=>server.listen(PORT,r));
const b=await puppeteer.launch({executablePath:"/home/christian-weilbach/.cache/puppeteer/chrome/linux-148.0.7778.97/chrome-linux64/chrome",headless:true,args:["--no-sandbox"]});
const pg=await b.newPage(); await pg.goto(`http://localhost:${PORT}/index.html`,{waitUntil:"load"});
const r=await pg.evaluate(async()=>{const bytes=await (await fetch("/relu_jvm.wasm")).arrayBuffer();const valid=WebAssembly.validate(bytes);const {instance}=await WebAssembly.instantiate(bytes,{});const {relu,memory}=instance.exports;const n=1000;const h=new Float64Array(memory.buffer);for(let i=0;i<n;i++){h[i]=i-500.0;h[n+i]=0.0;}relu(0,n*8,n);let bad=0;for(let i=0;i<n;i++){if(Math.abs(h[n+i]-Math.max(i-500.0,0.0))>1e-9)bad++;}return {valid,bad,ok:valid&&bad===0};});
console.log("real Chrome relu:",JSON.stringify(r),r.ok?"=> PASS":"=> FAIL"); await b.close(); server.close();
