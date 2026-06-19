import { createServer } from "node:http"; import { readFile } from "node:fs/promises";
import { join, extname } from "node:path"; import puppeteer from "puppeteer-core";
const ROOT=new URL("../public/",import.meta.url).pathname,PORT=8133;
const server=createServer(async(req,res)=>{try{let p=new URL(req.url,"http://x").pathname;if(p==="/")p="/index.html";const b=await readFile(join(ROOT,p));res.writeHead(200,{"content-type":{".wasm":"application/wasm",".html":"text/html"}[extname(p)]??"application/octet-stream"});res.end(b);}catch{res.writeHead(404);res.end();}});
await new Promise(r=>server.listen(PORT,r));
const b=await puppeteer.launch({executablePath:"/home/christian-weilbach/.cache/puppeteer/chrome/linux-148.0.7778.97/chrome-linux64/chrome",headless:true,args:["--no-sandbox"]});
const pg=await b.newPage(); await pg.goto(`http://localhost:${PORT}/index.html`,{waitUntil:"load"});
const r=await pg.evaluate(async()=>{const bytes=await (await fetch("/ss_jvm.wasm")).arrayBuffer();const valid=WebAssembly.validate(bytes);const {instance}=await WebAssembly.instantiate(bytes,{});const {ss,memory}=instance.exports;const n=1000;const h=new Float64Array(memory.buffer);let exp=0;for(let i=0;i<n;i++){h[i]=i;exp+=2.0*i;}const got=ss(0,n,4.0);return {valid,got,exp,ok:valid&&Math.abs(got-exp)<1e-6};});
console.log("real Chrome scaledsum:",JSON.stringify(r),r.ok?"=> PASS":"=> FAIL"); await b.close(); server.close();
