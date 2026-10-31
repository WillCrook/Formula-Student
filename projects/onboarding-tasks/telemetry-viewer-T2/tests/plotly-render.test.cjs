// Exercises the actual Plotly SVG renderer in a DOM, with approximate text metrics.
// This checks runtime integration, not browser layout or visual appearance.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {spawn}=require('node:child_process');
const {JSDOM,VirtualConsole}=require('jsdom');
const root=path.join(__dirname,'..');
let server=null,dom=null;
async function waitFor(check,label,timeout=15000){const deadline=Date.now()+timeout;while(Date.now()<deadline){if(check())return;await new Promise(resolve=>setTimeout(resolve,25));}throw Error('Timed out: '+label);}
async function main(){
 let base='',output='',logs='';
 try{
  server=spawn(path.join(root,'.venv/bin/python'),[path.join(__dirname,'ui_server.py')],{cwd:root,stdio:['ignore','pipe','pipe']});
  server.stdout.on('data',chunk=>{output+=chunk;const line=output.split('\n').find(line=>line.startsWith('{"url"'));if(line)base=JSON.parse(line).url;});server.stderr.on('data',chunk=>logs+=chunk);
  await waitFor(()=>base||server.exitCode!==null,'test server');if(!base)throw Error(logs);
  const html=await fetch(base).then(r=>r.text()),errors=[];
  const console=new VirtualConsole();console.on('jsdomError',error=>errors.push(error.message));
  dom=new JSDOM(html,{url:base,runScripts:'outside-only',pretendToBeVisual:true,virtualConsole:console});
  const w=dom.window,d=w.document;
  w.structuredClone=structuredClone;w.AbortController=AbortController;w.FormData=FormData;w.fetch=(url,options)=>fetch(new URL(url,base),options);
  w.URL.createObjectURL=()=> 'blob:test';w.URL.revokeObjectURL=()=>{};
  w.HTMLCanvasElement.prototype.getContext=function(type){if(type!=='2d')return null;return new Proxy({measureText:text=>({width:String(text).length*6}),getImageData:()=>({data:new Uint8ClampedArray(16)}),createLinearGradient:()=>({addColorStop(){}})},{get:(target,key)=>key in target?target[key]:(()=>{})});};
  w.SVGElement.prototype.getBBox=function(){return {x:0,y:0,width:(this.textContent||'').length*6,height:12};};
  w.SVGElement.prototype.getComputedTextLength=function(){return (this.textContent||'').length*6;};
  Object.defineProperty(w.HTMLElement.prototype,'clientWidth',{get(){return this.id==='telemetry-plot'?700:400;}});
  Object.defineProperty(w.HTMLElement.prototype,'clientHeight',{get(){return this.id==='telemetry-plot'?900:360;}});
  w.eval(fs.readFileSync(path.join(root,'static/vendor/plotly-4.1.1.min.js'),'utf8'));
  assert.equal(w.Plotly.version,'4.1.1');
  w.eval(fs.readFileSync(path.join(root,'static/workbench.js'),'utf8'));
  await waitFor(()=>d.querySelectorAll('#telemetry-plot .scatterlayer .trace').length>=12,'real telemetry SVG');
  await waitFor(()=>d.querySelectorAll('#map-plot .scatterlayer .trace').length>5,'real map SVG');
  assert(!d.getElementById('toast').classList.contains('error'),d.getElementById('toast').textContent);
  assert(d.getElementById('telemetry-plot')._fullLayout.yaxis6);
  d.querySelector('[data-tab="gg"]').click();await waitFor(()=>d.querySelectorAll('#gg-plot .scatterlayer .trace').length>=2,'G-G SVG');
  d.querySelector('[data-domain="time"]').click();
  await waitFor(()=>d.getElementById('telemetry-plot').data.every(trace=>trace.meta.channel!=='delta'),'time-domain render');
  d.querySelector('[data-tab="consistency"]').click();await waitFor(()=>d.querySelector('#consistency-plot .boxlayer'),'consistency SVG');
  assert.equal(errors.length,0,errors.join('\n'));
  process.stdout.write('PASS: actual Plotly 4.1.1 telemetry/map, G-G, time-domain and consistency SVG rendering in a DOM. Browser visual layout remains unverified.\n');
 }finally{
  if(dom)dom.window.close();
  if(server){server.kill('SIGINT');await new Promise(resolve=>server.exitCode!==null?resolve():server.on('exit',resolve));}
 }
}
main().catch(error=>{process.stderr.write(error.stack+'\n');process.exitCode=1;});
