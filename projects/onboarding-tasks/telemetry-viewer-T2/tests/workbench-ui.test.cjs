// DOM/workflow integration test. Plotly rendering is stubbed; no browser is simulated.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {JSDOM,VirtualConsole}=require('jsdom');
let base=process.env.TELEMETRY_TEST_URL||'';
const errors=[],downloads=[],calls=[];
const virtualConsole=new VirtualConsole();
virtualConsole.on('jsdomError',error=>errors.push(error.message));
async function waitFor(check,label,timeout=12000){const deadline=Date.now()+timeout;while(Date.now()<deadline){if(check())return;await new Promise(resolve=>setTimeout(resolve,20));}throw Error('Timed out: '+label);}
async function run(){
 const html=await fetch(base).then(response=>response.text());
 const dom=new JSDOM(html,{url:base,runScripts:'outside-only',pretendToBeVisual:true,virtualConsole});
 const w=dom.window,d=w.document;
 w.structuredClone=structuredClone;w.AbortController=AbortController;w.FormData=FormData;
 w.fetch=(url,options)=>{calls.push(String(url));return fetch(new URL(url,base),options);};
 w.HTMLDialogElement.prototype.showModal=function(){this.open=true;};
 w.HTMLDialogElement.prototype.close=function(){this.open=false;};
 w.HTMLAnchorElement.prototype.click=function(){downloads.push(this.download);};
 w.URL.createObjectURL=()=> 'blob:test';w.URL.revokeObjectURL=()=>{};
 const handlers=new WeakMap();
 function element(id){return typeof id==='string'?d.getElementById(id):id;}
 w.Plotly={
  async react(id,traces,layout){const node=element(id);node.data=traces;node.layout=layout;if(!handlers.has(node))handlers.set(node,{});node.on=(name,callback)=>{handlers.get(node)[name]=callback;};},
  async restyle(id,updates,indices){const node=element(id);indices.forEach((index,i)=>Object.keys(updates).forEach(key=>node.data[index][key]=updates[key][i]));},
  async relayout(id,updates){const node=element(id);Object.assign(node.layout,updates);handlers.get(node)?.plotly_relayout?.(updates);},
  async downloadImage(id,options){downloads.push(options.filename+'.'+options.format);},
  Fx:{hover(){}}
 };
 const emit=(id,name,event)=>handlers.get(d.getElementById(id))?.[name]?.(event);
 const click=id=>d.getElementById(id).click();
 const change=id=>d.getElementById(id).dispatchEvent(new w.Event('change',{bubbles:true}));
 const submit=id=>d.getElementById(id).dispatchEvent(new w.Event('submit',{bubbles:true,cancelable:true}));
 const settle=async()=>{await new Promise(resolve=>setTimeout(resolve,180));await waitFor(()=>!d.getElementById('summary-cards').classList.contains('loading'),'analysis settled');};
 w.eval(fs.readFileSync(path.join(__dirname,'../static/workbench.js'),'utf8'));
 try{
  await waitFor(()=>d.getElementById('telemetry-plot').data&&d.querySelectorAll('.metric-lap').length===2,'initial workspace');
  assert.equal(d.querySelectorAll('.lap-row').length,13);
  assert(d.getElementById('telemetry-plot').data.some(trace=>trace.meta.channel==='glat'));
  d.getElementById('reference-select').value='onboarding:11';change('reference-select');await settle();
  assert(d.getElementById('summary-cards').textContent.includes('62.682'));
  click('select-all');await settle();assert.equal(d.querySelectorAll('.lap-row.selected').length,13);
  d.querySelector('[data-domain="time"]').click();await settle();
  assert(!d.getElementById('telemetry-plot').data.some(trace=>trace.meta.channel==='delta'));
  assert(d.getElementById('telemetry-plot').layout.xaxis.range[1]>67);
  d.querySelector('[data-domain="distance"]').click();click('select-fastest');await settle();
  d.getElementById('range-start').value='82';d.getElementById('range-end').value='205';click('apply-range');await settle();
  d.getElementById('local-delta').checked=true;change('local-delta');await settle();
  const gap=d.getElementById('telemetry-plot').data.find(trace=>trace.meta.channel==='delta'&&trace.meta.lap==='onboarding:12');
  assert(Math.abs(gap.y[0])<1e-10);
  const speed=d.getElementById('telemetry-plot').data.find(trace=>trace.meta.channel==='speed');
  emit('telemetry-plot','plotly_hover',{points:[{data:speed,pointNumber:100,x:speed.x[100]}]});
  await waitFor(()=>Number.isFinite(w.telemetryCursor),'linked cursor');
  assert(d.getElementById('map-plot').data.at(-1).x.length===1);
  const turn=d.getElementById('map-plot').data.find(trace=>trace.meta?.role==='corners');
  emit('map-plot','plotly_click',{points:[{data:turn,customdata:turn.customdata[2]}]});await settle();
  assert(d.getElementById('selection-title').textContent.includes(turn.text[2]));
  for(const tab of ['consistency','energy','gg']){d.querySelector('[data-tab="'+tab+'"]').click();await waitFor(()=>d.getElementById(tab==='gg'?'gg-plot':tab+'-plot').data?.length,'tab '+tab);}
  d.getElementById('gg-color').value='corner';change('gg-color');await settle();
  assert(d.getElementById('gg-plot').layout.coloraxis.colorbar.ticktext.includes('Straight'));
  d.getElementById('map-color').value='gain';change('map-color');await settle();
  assert(d.getElementById('map-plot').data.some(trace=>trace.marker?.colorbar?.title?.text==='s / 100 m'));
  click('edit-track');assert(d.getElementById('track-dialog').open);
  const count=d.querySelectorAll('#interval-table tbody tr').length;
  d.querySelectorAll('#interval-table .row-select')[0].checked=true;d.querySelectorAll('#interval-table .row-select')[1].checked=true;
  click('merge-intervals');assert.equal(d.querySelectorAll('#interval-table tbody tr').length,count-1);
  d.querySelector('#interval-table .row-select').checked=true;click('split-interval');assert.equal(d.querySelectorAll('#interval-table tbody tr').length,count);
  d.querySelector('[data-edit="name"]').value='Reviewed opening';d.querySelector('[data-edit="confirmed"]').checked=true;
  d.querySelector('[data-editor="sectors"]').click();click('add-interval');assert.equal(d.querySelectorAll('#interval-table tbody tr').length,4);
  click('save-track');await waitFor(()=>!d.getElementById('track-dialog').open,'track saved');await settle();
  assert([...d.getElementById('section-select').options].some(option=>option.text.includes('Reviewed opening')));
  click('save-view');d.getElementById('view-name').value='DOM integration view';submit('save-form');
  await waitFor(()=>[...d.getElementById('saved-views').options].some(option=>option.text==='DOM integration view'),'saved view');
  d.getElementById('export-format').value='csv';click('export');await waitFor(()=>downloads.includes('telemetry-comparison.csv'),'CSV export');
  d.getElementById('export-format').value='png';click('export');await waitFor(()=>downloads.includes('telemetry-comparison.png'),'PNG action');
  const csv=['time,x,y,power,throttle'];
  for(let i=0;i<1120;i++){const t=i/10,angle=.2+.2*t;csv.push([t,40*Math.sin(angle),40*Math.cos(angle),10,.5].join(','));}
  async function addRun(existing=false){
   click('open-import');
   Object.defineProperty(d.getElementById('import-files'),'files',{configurable:true,value:[new File([csv.join('\n')],'circle.csv',{type:'text/csv'})]});
   submit('upload-form');await waitFor(()=>d.getElementById('import-mapping').classList.contains('hidden')===false&&d.getElementById('import-map').data,'import mapping');
   await waitFor(()=>d.getElementById('preview-import').disabled===false,'source preview ready');
   const throttle=[...d.querySelectorAll('.channel-map-row')].find(row=>row.querySelector('[data-field=name]').value==='throttle');
   throttle.querySelector('[data-field=unit]').value='fraction';throttle.dispatchEvent(new w.Event('change',{bubbles:true}));
   if(existing){
    const track=[...d.getElementById('import-track').options].find(option=>option.text==='DOM circle');assert(track);
    d.getElementById('import-track').value=track.value;change('import-track');d.getElementById('import-name').value='DOM repeat';
   }else{
    d.getElementById('import-track-name').value='DOM circle';d.getElementById('import-name').value='DOM run';
    const preview=d.getElementById('import-map').data[0];emit('import-map','plotly_click',{points:[{data:preview,customdata:10}]});
   }
   await waitFor(()=>!d.getElementById('commit-import').disabled,'accepted finish');
   click('commit-import');await waitFor(()=>!d.getElementById('import-dialog').open,'import committed');await settle();
  }
  await addRun(false);
  assert(d.getElementById('track-select').selectedOptions[0].text==='DOM circle');
  await addRun(true);
  assert(d.querySelectorAll('.run-name').length===2);
  const saved=[...d.getElementById('saved-views').options].find(option=>option.text==='DOM integration view');
  d.getElementById('saved-views').value=saved.value;change('saved-views');await settle();
  assert.equal(d.getElementById('track-select').value,'onboarding');
  assert.equal(errors.length,0,errors.join('\n'));
  console.log('PASS: initial render, reference/all laps, distance/time, zoom/local gap, linked cursor, corner selection, analysis tabs, turn merge/split, sector edit, saved views, exports, new-track import, cross-run import, restore view.');
  console.log('Plotly drawing and responsive visual layout require a real browser; this test checks DOM/data/event flows.');
 }finally{dom.window.close();}
}
async function main(){
 let server=null;
 try{
  if(!base){
   const {spawn}=require('node:child_process');
   server=spawn(path.join(__dirname,'../.venv/bin/python'),[path.join(__dirname,'ui_server.py')],{cwd:path.join(__dirname,'..'),stdio:['ignore','pipe','pipe']});
   let output='',logs='';
   server.stdout.on('data',chunk=>{output+=chunk;const line=output.split('\n').find(line=>line.startsWith('{"url"'));if(line)base=JSON.parse(line).url;});
   server.stderr.on('data',chunk=>logs+=chunk);
   await waitFor(()=>base||server.exitCode!==null,'test server startup');
   if(!base)throw Error(logs||'Test server exited');
  }
  await run();
 }finally{
  if(server){server.kill('SIGINT');await new Promise(resolve=>{if(server.exitCode!==null)resolve();else server.on('exit',resolve);});}
 }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
