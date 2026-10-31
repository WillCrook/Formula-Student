'use strict';
const $=id=>document.getElementById(id);
const palette=['#117d68','#db794a','#677ac2','#a4659d','#73964d','#bd9b42','#599ca7','#b85e69'];
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(value,digits=3)=>value===null||value===undefined||!Number.isFinite(Number(value))?'—':Number(value).toFixed(digits);
const finite=value=>value!==null&&value!==undefined&&Number.isFinite(Number(value));
let meta=null,data=null,requestNumber=0,controller=null,plotLinks=[],cursorTraces=[],hoverFrame=0,rendering=false,synchronizing=false,toastTimer=null,activeTab='section',rangeTitle='Whole lap';
let state={track:'onboarding',reference:null,laps:[],start:0,end:1,domain:'distance',channels:['speed','delta','throttle','brake','glat','glong'],local_delta:false,gg_color:'speed',map_color:'laps'};
const config={responsive:true,displaylogo:false,scrollZoom:false,modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d']};
const baseLayout={font:{family:'-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif',color:'#63786e',size:10},paper_bgcolor:'#fff',plot_bgcolor:'#fff',colorway:palette,margin:{l:48,r:14,t:10,b:40}};
function toast(message,error=false){clearTimeout(toastTimer);$('toast').textContent=message;$('toast').classList.toggle('error',error);$('toast').style.display='block';toastTimer=setTimeout(()=>$('toast').style.display='none',error?8000:3500);}
async function api(path,options={}){const response=await fetch(path,options);let value;try{value=await response.json();}catch{throw Error('The server did not return data. Check the terminal.');}if(!response.ok)throw Error(value.error||'Request failed');return value;}
function jsonRequest(value,method='POST'){return {method,headers:{'Content-Type':'application/json'},body:JSON.stringify(value)};}
function activeRuns(){return (meta?.runs||[]).filter(run=>run.track_id===state.track);}
function availableLaps(){return activeRuns().flatMap(run=>run.laps);}
function colour(id){const order=[state.reference,...state.laps.filter(key=>key!==state.reference)];return palette[Math.max(0,order.indexOf(id))%palette.length];}
function selection(){return {...state,laps:[...state.laps],channels:[...state.channels]};}
function interpolate(value,x,y){if(value<=x[0])return y[0];if(value>=x[x.length-1])return y[y.length-1];let lo=0,hi=x.length-1;while(hi-lo>1){const mid=(lo+hi)>>1;if(x[mid]<=value)lo=mid;else hi=mid;}return y[lo]+(value-x[lo])/(x[hi]-x[lo])*(y[hi]-y[lo]);}
function chooseTrack(id){state.track=id;const ordered=availableLaps().sort((a,b)=>a.duration_s-b.duration_s);state.reference=ordered[0]?.id;state.laps=ordered.slice(0,2).map(l=>l.id);state.start=0;state.end=1;rangeTitle='Whole lap';}
async function initialise(trackId=null){meta=await api('/api/workspace');if(trackId||!meta.tracks.some(track=>track.id===state.track))chooseTrack(trackId||meta.tracks[0].id);else if(!state.reference)chooseTrack(state.track);renderNavigation();await refresh();if(meta.errors.length)toast(meta.errors[0].error,true);}
function renderNavigation(){
 $('track-select').innerHTML=meta.tracks.map(track=>'<option value="'+esc(track.id)+'">'+esc(track.name)+'</option>').join('');$('track-select').value=state.track;
 $('saved-views').innerHTML='<option value="">Saved views</option>'+meta.views.map(view=>'<option value="'+esc(view.id)+'">'+esc(view.name)+'</option>').join('');
 const laps=availableLaps();$('reference-select').innerHTML=laps.map(lap=>'<option value="'+esc(lap.id)+'">'+esc(lap.run)+' · L'+lap.number+' · '+fmt(lap.duration_s)+' s</option>').join('');$('reference-select').value=state.reference;
 const fastest=Math.min(...laps.map(lap=>lap.duration_s));
 $('lap-list').innerHTML=activeRuns().map(run=>'<div class="run-name">'+esc(run.name)+'</div>'+run.laps.map(lap=>'<div class="lap-row '+(state.laps.includes(lap.id)?'selected ':'')+(state.reference===lap.id?'reference':'')+'" data-lap="'+esc(lap.id)+'"><input type="checkbox" aria-label="Compare '+esc(lap.label)+'" '+(state.laps.includes(lap.id)?'checked':'')+'><span class="lap-label"><span class="swatch" style="background:'+colour(lap.id)+'"></span>Lap '+lap.number+'</span><span class="lap-time">'+fmt(lap.duration_s)+'<span class="lap-gap">+'+fmt(lap.duration_s-fastest)+' s</span></span></div>').join('')).join('');
 $('run-count').textContent='· '+activeRuns().length+' run'+(activeRuns().length===1?'':'s');
}
async function refresh(){
 const sequence=++requestNumber;if(controller)controller.abort();controller=new AbortController();
 $('summary-cards').classList.add('loading');
 try{const result=await api('/api/analysis',{...jsonRequest(selection()),signal:controller.signal});if(sequence!==requestNumber)return;data=result;state.reference=result.reference;state.laps=result.traces.map(t=>t.id);state.start=result.start;state.end=result.end;renderNavigation();renderControls();renderSummaries();renderInspector();await renderPlots();}catch(error){if(error.name!=='AbortError')toast(error.message,true);}finally{if(sequence===requestNumber)$('summary-cards').classList.remove('loading');}
}
function renderControls(){
 const track=data.track,length=track.length_m;
 $('range-start').max=length;$('range-end').max=length;$('range-start').value=fmt(state.start*length,1);$('range-end').value=fmt(state.end*length,1);
 $('selection-title').textContent=rangeTitle;$('selection-span').textContent=fmt(state.start*length,0)+'–'+fmt(state.end*length,0)+' m';
 const previous=$('section-select').value;
 $('section-select').innerHTML='<option value="full">Whole lap</option><option value="custom">Custom range</option><optgroup label="Timing sectors">'+track.sectors.map(s=>'<option value="sector:'+esc(s.id)+'">'+esc(s.name)+'</option>').join('')+'</optgroup><optgroup label="Turn suggestions">'+track.corners.map(c=>'<option value="corner:'+esc(c.id)+'">'+esc(c.name)+' '+esc(c.direction||'')+(c.confirmed?'':' · suggested')+'</option>').join('')+'</optgroup>';
 $('section-select').value=state.start===0&&state.end===1?'full':previous&&[...$('section-select').options].some(o=>o.value===previous)?previous:'custom';
 $('corner-count').textContent=track.corners.filter(c=>c.confirmed).length+' / '+track.corners.length+' turns confirmed';
 $('channel-controls').innerHTML=Object.entries(data.channels).filter(([key])=>key!=='local_delta').map(([key,c])=>'<label class="channel-chip" title="'+esc(c.unit)+(c.kind==='derived'?' · calculated':'')+'"><input type="checkbox" value="'+esc(key)+'" '+(state.channels.includes(key)?'checked':'')+'>'+esc(key==='delta'?'Gap':c.label)+'</label>').join('');
 $('local-delta').checked=state.local_delta;$('local-delta').disabled=state.domain==='time';$('map-color').value=state.map_color;$('gg-color').value=state.gg_color;
 document.querySelectorAll('#axis-mode button').forEach(button=>button.classList.toggle('active',button.dataset.domain===state.domain));
}
function renderSummaries(){
 const ref=data.traces[0],comparison=data.traces.find(t=>t.id!==data.reference)||ref,m=comparison.metrics;
 const referenceLap=data.all_laps.find(lap=>lap.id===state.reference);
 const values=[['Reference lap',fmt(referenceLap?.duration_s)+' s',ref.label],['Section time',fmt(m.time_s)+' s',comparison.label],['Section delta',(m.delta_s>0?'+':'')+fmt(m.delta_s)+' s','vs '+ref.label],['Logged energy',fmt(m.energy_kj,1)+' kJ',comparison.label]];
 $('summary-cards').innerHTML=values.map(([label,value,note],index)=>'<div class="stat"><span class="stat-label">'+esc(label)+'</span><strong class="'+(index===2?(m.delta_s>0?'loss':'gain'):'')+'">'+esc(value)+'</strong><small>'+esc(note)+'</small></div>').join('');
}
function plotData(){
 const traces=[],axes={},channels=state.channels.filter(key=>data.channels[key]&&(state.domain==='distance'||!['delta','local_delta'].includes(key)));
 const gap=.035,space=(1-gap*(channels.length-1))/Math.max(1,channels.length);
 plotLinks=[];
 channels.forEach((key,row)=>{
  const yName=row?'y'+(row+1):'y',axisName=row?'yaxis'+(row+1):'yaxis',low=1-(row+1)*space-row*gap;
  const actual=key==='delta'&&state.local_delta?'local_delta':key;
  const descriptor=data.channels[actual];
  axes[axisName]={domain:[Math.max(0,low),Math.min(1,low+space)],anchor:'x',title:{text:esc(descriptor.label.length>20?descriptor.label.slice(0,19)+'…':descriptor.label)+'<br>'+esc(key==='brake'&&descriptor.unit.includes('unknown')?'unit unknown':descriptor.unit),font:{size:9}},gridcolor:'#edf1ed',zeroline:['delta','glat','glong'].includes(key),zerolinecolor:'#aebeb5',fixedrange:true};
  data.traces.forEach(lap=>{
   if(!lap[actual])return;
   traces.push({x:state.domain==='time'?lap.elapsed:lap.u.map(u=>u*data.track.length_m),y:lap[actual],yaxis:yName,xaxis:'x',type:'scatter',mode:'lines',name:lap.label,legendgroup:lap.id,showlegend:row===0,line:{color:colour(lap.id),width:lap.id===state.reference?1.8:1.4},connectgaps:false,meta:{channel:key,lap:lap.id},hovertemplate:esc(lap.label)+': %{y:.3f} '+esc(descriptor.unit)+'<extra></extra>'});
   plotLinks.push({lap:lap.id,channel:key});
  });
 });
 return {traces,axes,channels};
}
async function renderPlots(){
 rendering=true;
 try{const plotted=plotData(),ref=data.traces[0];
  $('plot-empty').classList.toggle('hidden',plotted.channels.length!==0);$('telemetry-plot').classList.toggle('hidden',plotted.channels.length===0);
  if(plotted.channels.length){const xRange=state.domain==='time'?[Math.min(...data.traces.map(t=>t.elapsed[0])),Math.max(...data.traces.map(t=>t.elapsed.at(-1)))]:[state.start*data.track.length_m,state.end*data.track.length_m];
   await Plotly.react('telemetry-plot',plotted.traces,{...baseLayout,...plotted.axes,height:Math.max(440,plotted.channels.length*128+95),margin:{l:65,r:15,t:35,b:45},legend:{orientation:'h',x:0,y:1.045,font:{size:10}},hovermode:'x',hoversubplots:'axis',dragmode:'zoom',xaxis:{anchor:'free',position:0,range:xRange,title:{text:state.domain==='time'?'Elapsed lap time (s)':'Reference-path distance (m)'},gridcolor:'#f0f3ef',showspikes:true,spikemode:'across',spikesnap:'cursor'},uirevision:state.track+':'+state.reference+':'+state.start+':'+state.end+':'+state.domain},config);
   bindTelemetry();
  }
  await renderMap();await renderActiveChart();
 }finally{rendering=false;}
}
function bindTelemetry(){
 const plot=$('telemetry-plot');if(plot.dataset.bound)return;plot.dataset.bound='true';
 plot.on('plotly_hover',event=>{if(synchronizing||!event.points?.length)return;const point=event.points[0],lap=data.traces.find(t=>t.id===point.data.meta?.lap);if(lap&&Number.isInteger(point.pointNumber))scheduleCursor(lap.u[point.pointNumber]);});
 plot.on('plotly_relayout',event=>{if(rendering)return;let range=event['xaxis.range'];if(!range&&event['xaxis.range[0]']!==undefined)range=[event['xaxis.range[0]'],event['xaxis.range[1]']];if(!range)return;const ref=data.traces[0];const u=range.map(x=>state.domain==='time'?interpolate(x,ref.elapsed,ref.u):x/data.track.length_m);const start=Math.max(0,u[0]),end=Math.min(1,u[1]);if(end-start>=.001&&(Math.abs(start-state.start)>1e-5||Math.abs(end-state.end)>1e-5)){state.start=start;state.end=end;rangeTitle='Custom range';$('section-select').value='custom';refresh();}});
 plot.on('plotly_doubleclick',()=>{resetRange();return false;});
}
function scheduleCursor(u){if(!finite(u))return;cancelAnimationFrame(hoverFrame);hoverFrame=requestAnimationFrame(()=>updateCursor(u));}
function updateCursor(u){
 const updates={x:[],y:[]};
 data.traces.forEach(lap=>{updates.x.push([interpolate(u,lap.u,lap.x)]);updates.y.push([interpolate(u,lap.u,lap.y)]);});
 if(cursorTraces.length)Plotly.restyle('map-plot',updates,cursorTraces);
 if($('telemetry-plot').data&&!$('telemetry-plot').classList.contains('hidden')){const ref=data.traces[0],x=state.domain==='time'?interpolate(u,ref.u,ref.elapsed):u*data.track.length_m;Plotly.relayout('telemetry-plot',{shapes:[{type:'line',xref:'x',yref:'paper',x0:x,x1:x,y0:0,y1:1,line:{color:'#9cafa1',width:1,dash:'dot'}}]});}
 const ref=data.traces[0],speed=interpolate(u,ref.u,ref.speed);$('cursor-readout').textContent=fmt(u*data.track.length_m,1)+' m · '+fmt(speed,1)+' km/h';
 window.telemetryCursor=u;
}
async function renderMap(){
 const track=data.track,path=track.path,mapTraces=[{x:path.map(p=>p[0]),y:path.map(p=>p[1]),mode:'lines',line:{color:'#dbe3da',width:8},hoverinfo:'skip',showlegend:false}];
 data.traces.forEach(lap=>mapTraces.push({x:lap.full_x,y:lap.full_y,customdata:lap.full_u,mode:'lines',line:{color:colour(lap.id),width:1.2},opacity:state.map_color==='laps'?.65:.2,showlegend:false,meta:{role:'path'},hovertemplate:esc(lap.label)+'<extra></extra>'}));
 if(state.map_color!=='laps'){
  const lap=data.traces.find(t=>t.id!==state.reference)||data.traces[0],gain=state.map_color==='gain',values=gain?lap.local_rate:lap.full_speed;
  mapTraces.push({x:lap.full_x,y:lap.full_y,customdata:lap.full_u,mode:'markers',showlegend:false,meta:{role:'path'},marker:{size:4,color:values,colorscale:gain?[[0,'#128261'],[.5,'#e9eee4'],[1,'#db7447']]:'Viridis',cmin:gain?-Math.max(.05,...values.map(Math.abs)):undefined,cmax:gain?Math.max(.05,...values.map(Math.abs)):undefined,colorbar:{thickness:8,len:.55,title:{text:gain?'s / 100 m':'km/h',font:{size:9}},tickfont:{size:8}}},hovertemplate:gain?'Local loss %{marker.color:.3f} s/100 m<extra></extra>':'%{marker.color:.1f} km/h<extra></extra>'});
 }
 if(state.map_color==='laps'||state.start>0||state.end<1)data.traces.forEach(lap=>mapTraces.push({x:lap.x,y:lap.y,customdata:lap.u,mode:'lines',line:{color:colour(lap.id),width:3},showlegend:false,meta:{role:'path'},hovertemplate:esc(lap.label)+'<extra></extra>'}));
 const cornerX=[],cornerY=[];
 track.corners.forEach(c=>{cornerX.push(interpolate(c.position,data.traces[0].full_u,path.map(p=>p[0])));cornerY.push(interpolate(c.position,data.traces[0].full_u,path.map(p=>p[1])));});
 mapTraces.push({x:cornerX,y:cornerY,customdata:track.corners.map(c=>c.id),text:track.corners.map(c=>c.name),textposition:cornerX.map(x=>x>path.reduce((a,p)=>a+p[0],0)/path.length?'middle right':'middle left'),textfont:{size:9,color:'#557465'},mode:'markers+text',marker:{size:5,color:'#7f9b7b'},showlegend:false,meta:{role:'corners'},hovertemplate:'%{text}<extra>Click to inspect</extra>'});
 const lines=[track.line,...data.gates];
 lines.forEach((line,i)=>mapTraces.push({x:line.map(p=>p[0]),y:line.map(p=>p[1]),mode:'lines',line:{color:i?'#8eaa81':'#b54e3c',width:i?1.5:3,dash:i?'dash':'solid'},hoverinfo:'skip',showlegend:false}));
 cursorTraces=[];data.traces.forEach(lap=>{cursorTraces.push(mapTraces.length);mapTraces.push({x:[lap.x[0]],y:[lap.y[0]],mode:'markers',marker:{color:colour(lap.id),size:10,line:{color:'#fff',width:2}},hoverinfo:'skip',showlegend:false});});
 await Plotly.react('map-plot',mapTraces,{...baseLayout,margin:{l:10,r:state.map_color==='laps'?10:45,t:5,b:5},showlegend:false,hovermode:'closest',dragmode:'pan',xaxis:{visible:false,zeroline:false,scaleanchor:'y',scaleratio:1},yaxis:{visible:false,zeroline:false},uirevision:state.track}, {...config,scrollZoom:true,modeBarButtonsToRemove:['select2d','lasso2d']});
 const map=$('map-plot');if(!map.dataset.bound){map.dataset.bound='true';map.on('plotly_click',event=>{const point=event.points?.[0];if(!point)return;if(point.data.meta?.role==='corners'){chooseSection('corner:'+point.customdata);}else if(point.data.meta?.role==='path'){const u=Number(point.customdata),corner=data.track.corners.find(c=>u>=c.start&&u<=c.end);if(corner)chooseSection('corner:'+corner.id);else setRange(Math.max(0,u-.04),Math.min(1,u+.04),'Track selection');}});
 map.on('plotly_hover',event=>{const point=event.points?.[0];if(!point||point.data.meta?.role!=='path'||!finite(point.customdata))return;const u=Number(point.customdata);if(u<state.start||u>state.end)return;const ref=data.traces[0],index=Math.round((u-state.start)/(state.end-state.start)*(ref.u.length-1));synchronizing=true;try{if($('telemetry-plot').data)Plotly.Fx.hover('telemetry-plot',plotLinks.map((_,curveNumber)=>({curveNumber,pointNumber:index})));}finally{synchronizing=false;}scheduleCursor(u);});}
}
function setRange(start,end,title){state.start=Math.max(0,start);state.end=Math.min(1,end);rangeTitle=title;refresh();}
function resetRange(){rangeTitle='Whole lap';state.start=0;state.end=1;$('section-select').value='full';refresh();}
function chooseSection(value){$('section-select').value=value;if(value==='full'){resetRange();return;}if(value==='custom')return;const [type,id]=value.split(':');const item=(type==='corner'?data.track.corners:data.track.sectors).find(s=>s.id===id);if(!item)return;const context=type==='corner'?Number($('context-select').value)/data.track.length_m:0;setRange(item.start-context,item.end+context,item.name);}

function renderInspector(){
 $('section-metrics').innerHTML=data.traces.map(trace=>{const m=trace.metrics;return '<div class="metric-lap" style="border-color:'+colour(trace.id)+'"><div class="metric-lap-title"><span>'+esc(trace.label)+'</span><span class="'+(m.delta_s>0?'loss':'gain')+'">'+(m.delta_s>0?'+':'')+fmt(m.delta_s)+' s</span></div><div class="metric-grid"><div><strong>'+fmt(m.entry_kmh,1)+'</strong><small>Entry · km/h</small></div><div><strong>'+fmt(m.minimum_kmh,1)+'</strong><small>Minimum · km/h</small></div><div><strong>'+fmt(m.exit_kmh,1)+'</strong><small>Exit · km/h</small></div></div>'+(m.status!=='ok'?'<div class="metric-note">'+esc(m.status)+'</div>':'')+'</div>';}).join('');
 const stats=data.statistics;
 $('consistency-summary').innerHTML='<div class="analysis-stat"><strong>'+fmt(stats.composite_s)+' s</strong><span>Best-sector sum<br>'+stats.scope_laps+' complete laps</span></div>';
 $('consistency-table').innerHTML='<table><thead><tr><th>Sector</th><th>Best</th><th>Median</th><th>Std</th><th>n</th></tr></thead><tbody>'+stats.sectors.map(s=>'<tr title="'+esc(s.best_lap||'No valid gate timings')+'"><td>'+esc(s.name)+'<span class="hint" style="display:block">'+(s.best_lap?esc((s.samples.find(m=>m.id===s.best_lap)?.run||'')+' · L'+s.best_lap.split(':').at(-1)):'—')+'</span></td><td>'+fmt(s.best_s)+'</td><td>'+fmt(s.median_s)+'</td><td>'+fmt(s.std_s)+'</td><td>'+s.count+'</td></tr>').join('')+'</tbody></table>';
 const energy=data.traces.find(t=>t.id!==state.reference)||data.traces[0];
 $('energy-summary').innerHTML='<div class="analysis-stat"><strong>'+fmt(energy.metrics.energy_kj,1)+' kJ</strong><span>Selected section<br>'+esc(energy.label)+'</span></div>';
 $('energy-table').innerHTML='<table><thead><tr><th>Lap</th><th>Section kJ</th><th>kWh</th></tr></thead><tbody>'+data.traces.map(t=>'<tr><td>'+esc(t.label)+'</td><td>'+fmt(t.metrics.energy_kj,1)+'</td><td>'+fmt(t.metrics.energy_kj===null?null:t.metrics.energy_kj/3600,4)+'</td></tr>').join('')+'</tbody></table>';
}
async function renderActiveChart(){
 if(activeTab==='section')return;
 if(activeTab==='consistency'){
  const traces=data.statistics.sectors.map((s,i)=>({type:'box',name:s.name,y:s.samples.map(m=>m.time_s),text:s.samples.map(m=>m.run+' · L'+m.lap),boxpoints:'all',jitter:.3,pointpos:0,marker:{color:palette[i%palette.length],size:4},line:{width:1.2},hovertemplate:'%{text}<br>%{y:.3f} s<extra></extra>'}));
  await Plotly.react('consistency-plot',traces,{...baseLayout,showlegend:false,margin:{l:40,r:8,t:15,b:35},yaxis:{title:{text:'Sector time (s)'},gridcolor:'#edf1ed'}},config);
 }
 if(activeTab==='energy'){
  const laps=data.all_laps.filter(l=>finite(l.energy_kj));
  await Plotly.react('energy-plot',[{x:laps.map(l=>l.duration_s),y:laps.map(l=>l.energy_kj),text:laps.map(l=>l.label),type:'scatter',mode:'markers',marker:{color:laps.map(l=>colour(l.id)),size:laps.map(l=>state.laps.includes(l.id)?10:5),opacity:.85,symbol:laps.map(l=>l.id===state.reference?'diamond':'circle')},hovertemplate:'%{text}<br>%{x:.3f} s · %{y:.1f} kJ<extra></extra>'}],{...baseLayout,margin:{l:48,r:10,t:20,b:45},xaxis:{title:{text:'Complete lap time (s)'},gridcolor:'#edf1ed'},yaxis:{title:{text:'Logged energy (kJ)'},gridcolor:'#edf1ed'},showlegend:false},config);
 }
 if(activeTab==='gg'){
  const traces=[],speedColours=state.gg_color==='speed',corners=data.track.corners,cornerNames=['Straight',...corners.map(c=>c.name)];
  data.traces.forEach(lap=>{
   if(!lap.glat||!lap.glong)return;
   const indices=lap.u.map((u,i)=>i).filter(i=>finite(lap.glat[i])&&finite(lap.glong[i]));
   if(!indices.length)return;
   const ids=indices.map(i=>corners.findIndex(c=>lap.u[i]>=c.start&&lap.u[i]<=c.end)+1);
   traces.push({x:indices.map(i=>lap.glat[i]),y:indices.map(i=>lap.glong[i]),customdata:indices.map((i,k)=>[lap.u[i]*data.track.length_m,lap.speed[i],cornerNames[ids[k]]]),mode:'markers',type:'scatter',name:lap.label,marker:{size:4,opacity:.65,symbol:lap.id===state.reference?'circle':'diamond',color:speedColours?indices.map(i=>lap.speed[i]):ids,coloraxis:'coloraxis'},hovertemplate:esc(lap.label)+'<br>Lat %{x:.2f} g · Long %{y:.2f} g<br>%{customdata[0]:.1f} m · %{customdata[1]:.1f} km/h<br>%{customdata[2]}<extra></extra>'});
  });
  $('gg-empty').classList.toggle('hidden',traces.length!==0);$('gg-plot').classList.toggle('hidden',traces.length===0);
  if(traces.length){
   const count=cornerNames.length,scales=[];
   for(let i=0;i<count;i++){const col=i?palette[(i-1)%palette.length]:'#b4bdb6';scales.push([i/count,col],[(i+1)/count,col]);}
   const ticks=cornerNames.map((_,i)=>i).filter(i=>i===0||i%Math.max(1,Math.ceil(count/10))===0);
   const coloraxis=speedColours?{colorscale:'Viridis',colorbar:{title:{text:'km/h'},thickness:8,len:.7}}:{colorscale:scales,cmin:-.5,cmax:count-.5,colorbar:{thickness:8,tickvals:ticks,ticktext:ticks.map(i=>cornerNames[i]),tickfont:{size:8}}};
   await Plotly.react('gg-plot',traces,{...baseLayout,margin:{l:45,r:45,t:20,b:45},coloraxis,legend:{orientation:'h',y:1.08,font:{size:8}},xaxis:{title:{text:'Glat (g) · +left'},gridcolor:'#edf1ed',zerolinecolor:'#aab9ac'},yaxis:{title:{text:'Glong (g)'},scaleanchor:'x',scaleratio:1,gridcolor:'#edf1ed',zerolinecolor:'#aab9ac'}},config);
  }
 }
}
function setTab(tab){activeTab=tab;document.querySelectorAll('.tabs button').forEach(button=>button.classList.toggle('active',button.dataset.tab===tab));document.querySelectorAll('.tab-content').forEach(element=>element.classList.toggle('hidden',element.id!=='tab-'+tab));renderActiveChart().catch(error=>toast(error.message,true));}

let importDraft=null,importColumns=[],importGateIndex=null,importGood=false,importPreviewNumber=0,importPreviewTimer=null;
function columnOptions(value='',optional=false){return '<option value="">'+(optional?'Not mapped':'Choose column')+'</option>'+importColumns.map(c=>'<option value="'+esc(c.key)+'" '+(c.key===value?'selected':'')+'>'+esc(c.file+' · '+c.name)+' ('+c.samples+')</option>').join('');}
function guessColumn(pattern){return importColumns.find(column=>pattern.test(column.name))?.key||'';}
function markImportDirty(){importGood=false;$('commit-import').disabled=true;$('import-count').textContent='Preview changes';}
function showImportError(message){$('import-error').textContent=message;$('import-error').classList.toggle('hidden',!message);}
function addChannelRow(spec={}){
 const row=document.createElement('div');row.className='channel-map-row';
 row.innerHTML='<div class="channel-map-heading"><input data-field="name" aria-label="Channel name" placeholder="Channel name" value="'+esc(spec.name||'')+'"><button type="button" class="icon-button remove-channel" aria-label="Remove channel">×</button></div><div class="two-fields"><label>Value<select data-field="value">'+columnOptions(spec.value)+'</select></label><label>Timestamp<select data-field="time">'+columnOptions(spec.time||$('map-time').value)+'</select></label></div><div class="three-fields"><label>Units<input data-field="unit" list="units" value="'+esc(spec.unit||'unknown')+'"></label><label>Time<select data-field="time_unit"><option value="s">Seconds</option><option value="ms">Milliseconds</option><option value="iso">ISO 8601</option></select></label><label>Offset · s<input data-field="offset_s" type="number" step=".001" value="0"></label></div>';
 row.querySelector('[data-field=time_unit]').value=spec.time_unit||$('map-time-unit').value;
 row.querySelector('.remove-channel').onclick=()=>{row.remove();markImportDirty();};
 row.addEventListener('change',markImportDirty);
 $('import-channel-rows').appendChild(row);
}
function importMapping(){
 const channels=[...$('import-channel-rows').children].map(row=>Object.fromEntries([...row.querySelectorAll('[data-field]')].map(input=>[input.dataset.field,input.dataset.field==='offset_s'?Number(input.value):input.value])));
 return {coordinates:$('coordinate-mode').value,time:$('map-time').value,time_unit:$('map-time-unit').value,latitude:$('map-latitude').value,longitude:$('map-longitude').value,x:$('map-x').value,y:$('map-y').value,xy_unit:$('map-xy-unit').value,channels};
}
function importConfig(){return {draft:importDraft,name:$('import-name').value,track_name:$('import-track-name').value,track:$('import-track').value||null,mapping:importMapping(),gate_index:importGateIndex,gate_width:Number($('gate-width').value),gate_angle:Number($('gate-angle').value),gate_flip:$('gate-flip').checked};}
async function uploadFiles(event){
 event.preventDefault();showImportError('');const form=new FormData();for(const file of $('import-files').files)form.append('files',file);
 const submit=$('upload-form').querySelector('button');submit.disabled=true;submit.textContent='Reading…';
 try{
  const staged=await api('/api/import/stage',{method:'POST',body:form});importDraft=staged.id;importColumns=staged.columns;importGateIndex=null;importGood=false;
  $('source-preview-title').textContent=importColumns.length+' source columns';$('source-preview-table').innerHTML='<table><thead><tr><th>File / column</th><th>Samples</th><th>First values</th></tr></thead><tbody>'+importColumns.map(column=>'<tr><td>'+esc(column.file+' · '+column.name)+'</td><td>'+column.samples+'</td><td>'+esc(column.preview.join(' / '))+'</td></tr>').join('')+'</tbody></table>';
  $('import-name').value=staged.files[0].replace(/\.[^.]+$/,'');$('import-track-name').value=$('import-name').value;
  $('import-track').innerHTML='<option value="">New track</option>'+meta.tracks.map(track=>'<option value="'+esc(track.id)+'">'+esc(track.name)+'</option>').join('');
  document.querySelectorAll('.column-select').forEach(select=>select.innerHTML=columnOptions());
  $('map-time').value=guessColumn(/^(GPStime|time|timestamp|t)$/i)||guessColumn(/time/i);
  $('map-latitude').value=guessColumn(/^(GPSlat|lat|latitude)$/i);$('map-longitude').value=guessColumn(/^(GPSlon|lon|long|longitude)$/i);
  $('map-x').value=guessColumn(/^(x|easting|east)$/i);$('map-y').value=guessColumn(/^(y|northing|north)$/i);
  $('coordinate-mode').value=$('map-latitude').value&&$('map-longitude').value?'gps':'xy';
  const timeColumn=importColumns.find(c=>c.key===$('map-time').value);$('map-time-unit').value=timeColumn?.preview.some(value=>/T.*(Z|[+-]\d\d:\d\d)$/.test(value))?'iso':'s';
  $('import-channel-rows').innerHTML='';
  for(const spec of [{name:'throttle',value:/^(Throttle|rThrottle|throttle_pct)$/i,time:/^(ThrottleTime)$/i,unit:'%'},{name:'brake',value:/^(BrakeFront|pBrakeF|brake)$/i,time:/^(BrakeTime)$/i,unit:'unknown'},{name:'power',value:/^(Power|power_kw)$/i,time:/^(PowerTime)$/i,unit:'kW'},{name:'altitude',value:/^(GPSalt|altitude|elevation|ele)$/i,time:/^(GPStime|time)$/i,unit:'unknown'}]){
   const value=guessColumn(spec.value);if(value)addChannelRow({name:spec.name,value,time:guessColumn(spec.time)||$('map-time').value,unit:spec.unit});
  }
  $('import-mapping').classList.remove('hidden');toggleCoordinateFields();markImportDirty();await previewImport();
 }catch(error){showImportError(error.message);}finally{submit.disabled=false;submit.textContent='Read files';}
}
function toggleCoordinateFields(){const xy=$('coordinate-mode').value==='xy';$('gps-fields').classList.toggle('hidden',xy);$('xy-fields').classList.toggle('hidden',!xy);$('xy-unit-label').classList.toggle('hidden',!xy);}
async function previewImport(){
 if(!importDraft)return;const sequence=++importPreviewNumber;showImportError('');$('preview-import').disabled=true;$('import-count').textContent='Checking…';
 try{
  const preview=await api('/api/import/preview',jsonRequest(importConfig()));if(sequence!==importPreviewNumber)return;
  const traces=[{x:preview.x,y:preview.y,customdata:preview.indices,mode:'lines+markers',line:{color:'#94afa0',width:1},marker:{size:3,color:'#367e61'},hovertemplate:'Click to place finish<extra></extra>',meta:{role:'position'}}];
  if(preview.line)traces.push({x:preview.line.map(p=>p[0]),y:preview.line.map(p=>p[1]),mode:'lines',line:{color:'#c06645',width:4},hoverinfo:'skip'});
  await Plotly.react('import-map',traces,{...baseLayout,margin:{l:15,r:15,t:10,b:15},xaxis:{visible:false,scaleanchor:'y',scaleratio:1},yaxis:{visible:false},showlegend:false,dragmode:'pan',uirevision:importDraft}, {...config,scrollZoom:true});
  const plot=$('import-map');if(!plot.dataset.bound){plot.dataset.bound='true';plot.on('plotly_click',event=>{if($('import-track').value)return;const point=event.points?.[0];if(point?.data.meta?.role==='position'){importGateIndex=point.customdata;previewImport();}});}
  importGood=preview.complete_laps>0;$('commit-import').disabled=!importGood;$('import-count').textContent=importGood?preview.complete_laps+' complete laps':'Choose finish on path';
  $('import-status').textContent=importGood?preview.samples.toLocaleString()+' positions · '+preview.lap_times.map(time=>fmt(time,2)+' s').join(' / '):preview.samples.toLocaleString()+' positions';
  $('gate-controls').classList.toggle('hidden',!!$('import-track').value);$('new-track-name-label').classList.toggle('hidden',!!$('import-track').value);
 }catch(error){showImportError(error.message);importGood=false;$('commit-import').disabled=true;$('import-count').textContent='Check mapping / gate';}finally{if(sequence===importPreviewNumber)$('preview-import').disabled=false;}
}
async function commitImport(){
 if(!importGood)return;$('commit-import').disabled=true;$('commit-import').textContent='Adding…';showImportError('');
 try{const result=await api('/api/import/commit',jsonRequest(importConfig()));$('import-dialog').close();await initialise(result.track);const imported=meta.runs.find(run=>run.id===result.run);if(imported){const sorted=[...imported.laps].sort((a,b)=>a.duration_s-b.duration_s);state.laps=[state.reference,sorted[0].id];await refresh();}toast('Run added · '+result.laps+' complete laps');}
 catch(error){showImportError(error.message);}finally{$('commit-import').disabled=!importGood;$('commit-import').textContent='Add run';}
}

let editor={corners:[],sectors:[],tab:'corners'};
function openTrackEditor(){editor={corners:structuredClone(data.track.corners),sectors:structuredClone(data.track.sectors),tab:'corners'};$('track-error').classList.add('hidden');drawEditor();$('track-dialog').showModal();}
function readEditor(){const items=[...$('interval-table').querySelectorAll('tbody tr')].map(row=>({id:row.dataset.id,name:row.querySelector('[data-edit=name]').value,start:Number(row.querySelector('[data-edit=start]').value)/100,end:Number(row.querySelector('[data-edit=end]').value)/100,confirmed:row.querySelector('[data-edit=confirmed]').checked,position:Number(row.dataset.position),direction:row.dataset.direction}));editor[editor.tab]=items;return items;}
function drawEditor(){
 document.querySelectorAll('#editor-tabs button').forEach(button=>button.classList.toggle('active',button.dataset.editor===editor.tab));
 $('interval-table').querySelector('tbody').innerHTML=editor[editor.tab].map((item,index)=>'<tr data-id="'+esc(item.id||'edit-'+Date.now()+'-'+index)+'" data-position="'+(item.position??(item.start+item.end)/2)+'" data-direction="'+esc(item.direction||'')+'"><td><input type="checkbox" class="row-select" aria-label="Select '+esc(item.name)+'"></td><td><input type="text" data-edit="name" value="'+esc(item.name)+'" maxlength="40" aria-label="Name"></td><td><input type="number" data-edit="start" value="'+fmt(item.start*100,3)+'" step=".1" min="0" max="100" aria-label="Start percent"></td><td><input type="number" data-edit="end" value="'+fmt(item.end*100,3)+'" step=".1" min="0" max="100" aria-label="End percent"></td><td><input type="checkbox" data-edit="confirmed" aria-label="Confirmed" '+(item.confirmed!==false?'checked':'')+'></td><td><button class="text-button delete-interval" data-index="'+index+'" aria-label="Delete '+esc(item.name)+'">×</button></td></tr>').join('');
 $('confirm-corners').classList.toggle('hidden',editor.tab!=='corners');
}
function selectedEditorIndices(){return [...$('interval-table').querySelectorAll('tbody tr')].flatMap((row,i)=>row.querySelector('.row-select').checked?[i]:[]);}
function editFailure(message){$('track-error').textContent=message;$('track-error').classList.remove('hidden');}
function newInterval(name,start,end){return {id:'section-'+Date.now()+'-'+Math.round(Math.random()*1e6),name,start,end,position:(start+end)/2,confirmed:true};}
function mergeIntervals(){
 const selected=selectedEditorIndices(),items=readEditor();if(selected.length<2){editFailure('Select two or more adjacent rows.');return;}
 const first=Math.min(...selected),last=Math.max(...selected);if(last-first+1!==selected.length){editFailure('Select adjacent rows to merge.');return;}
 items.splice(first,last-first+1,newInterval(items[first].name+'–'+items[last].name,Math.min(...items.slice(first,last+1).map(i=>i.start)),Math.max(...items.slice(first,last+1).map(i=>i.end))));drawEditor();
}
function splitInterval(){
 const indices=selectedEditorIndices(),items=readEditor();if(indices.length!==1){editFailure('Select one row to split.');return;}
 const i=indices[0],item=items[i],cursor=window.telemetryCursor;
 const split=finite(cursor)&&cursor>item.start+.001&&cursor<item.end-.001?cursor:(item.start+item.end)/2;
 items.splice(i,1,newInterval(item.name+'a',item.start,split),newInterval(item.name+'b',split,item.end));drawEditor();
}
async function saveTrack(){
 readEditor();$('save-track').disabled=true;$('track-error').classList.add('hidden');
 try{await api('/api/tracks/'+encodeURIComponent(state.track),jsonRequest({corners:editor.corners,sectors:editor.sectors},'PUT'));$('track-dialog').close();meta=await api('/api/workspace');await refresh();toast('Track saved');}
 catch(error){editFailure(error.message);}finally{$('save-track').disabled=false;}
}
async function exportComparison(){
 const format=$('export-format').value;
 if(format==='png'){if(!$('telemetry-plot').data){toast('Choose a channel first.',true);return;}await Plotly.downloadImage('telemetry-plot',{format:'png',filename:'telemetry-comparison',scale:2});return;}
 const button=$('export');button.disabled=true;button.textContent='Exporting…';
 try{const response=await fetch('/api/export/'+format,jsonRequest(selection()));if(!response.ok){const value=await response.json();throw Error(value.error||'Export failed');}const blob=await response.blob(),url=URL.createObjectURL(blob),anchor=document.createElement('a');anchor.href=url;anchor.download='telemetry-comparison.'+format;anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
 catch(error){toast(error.message,true);}finally{button.disabled=false;button.textContent='Export ↓';}
}
function bindControls(){
 $('track-select').addEventListener('change',()=>{chooseTrack($('track-select').value);renderNavigation();refresh();});
 $('reference-select').addEventListener('change',()=>{state.reference=$('reference-select').value;if(!state.laps.includes(state.reference))state.laps.unshift(state.reference);refresh();});
 $('lap-list').addEventListener('click',event=>{const row=event.target.closest('[data-lap]');if(!row)return;const id=row.dataset.lap,checkbox=row.querySelector('input');if(event.target!==checkbox)checkbox.checked=!checkbox.checked;if(!checkbox.checked&&id===state.reference){checkbox.checked=true;toast('Choose a different reference before removing this lap.');return;}state.laps=checkbox.checked?[...new Set([...state.laps,id])]:state.laps.filter(key=>key!==id);refresh();});
 $('select-all').onclick=()=>{state.laps=availableLaps().map(lap=>lap.id);refresh();};
 $('select-fastest').onclick=()=>{state.laps=[state.reference,...availableLaps().filter(lap=>lap.id!==state.reference).sort((a,b)=>a.duration_s-b.duration_s).slice(0,1).map(lap=>lap.id)];refresh();};
 $('section-select').onchange=()=>chooseSection($('section-select').value);
 $('context-select').onchange=()=>{if($('section-select').value.startsWith('corner:'))chooseSection($('section-select').value);};
 $('full-lap').onclick=resetRange;
 $('apply-range').onclick=()=>{const start=Number($('range-start').value)/data.track.length_m,end=Number($('range-end').value)/data.track.length_m;if(!Number.isFinite(start+end)||start<0||end>1||end-start<.001){toast('Choose a valid start and end distance.',true);return;}$('section-select').value='custom';setRange(start,end,'Custom range');};
 for(const id of ['range-start','range-end'])$(id).addEventListener('keydown',event=>{if(event.key==='Enter')$('apply-range').click();});
 $('channel-controls').addEventListener('change',event=>{state.channels=[...$('channel-controls').querySelectorAll('input:checked')].map(input=>input.value);renderPlots().catch(error=>toast(error.message,true));});
 $('axis-mode').addEventListener('click',event=>{const button=event.target.closest('[data-domain]');if(!button)return;state.domain=button.dataset.domain;renderControls();renderPlots().catch(error=>toast(error.message,true));});
 $('local-delta').onchange=()=>{state.local_delta=$('local-delta').checked;renderPlots().catch(error=>toast(error.message,true));};
 $('map-color').onchange=()=>{state.map_color=$('map-color').value;renderMap().catch(error=>toast(error.message,true));};
 $('gg-color').onchange=()=>{state.gg_color=$('gg-color').value;renderActiveChart().catch(error=>toast(error.message,true));};
 document.querySelectorAll('.tabs button').forEach(button=>button.onclick=()=>setTab(button.dataset.tab));
 $('toggle-sidebar').onclick=()=>$('sidebar').classList.toggle('open');
 $('export').onclick=exportComparison;
 $('save-view').onclick=()=>{$('view-name').value=rangeTitle+' · '+data.traces.map(t=>'L'+t.number).join(' vs ');$('save-dialog').showModal();$('view-name').focus();};
 $('save-form').onsubmit=async event=>{event.preventDefault();try{await api('/api/views',jsonRequest({name:$('view-name').value,selection:selection()}));$('save-dialog').close();meta=await api('/api/workspace');renderNavigation();toast('Comparison saved');}catch(error){toast(error.message,true);}};
 $('saved-views').onchange=async()=>{const view=meta.views.find(v=>v.id===$('saved-views').value);if(!view)return;state={...state,...view.selection};rangeTitle=view.name;renderNavigation();await refresh();};
 $('edit-track').onclick=openTrackEditor;
 $('editor-tabs').onclick=event=>{const tab=event.target.closest('[data-editor]');if(!tab)return;readEditor();editor.tab=tab.dataset.editor;drawEditor();};
 $('interval-table').onclick=event=>{const button=event.target.closest('.delete-interval');if(!button)return;readEditor();editor[editor.tab].splice(Number(button.dataset.index),1);drawEditor();};
 $('add-interval').onclick=()=>{readEditor();const items=editor[editor.tab];if(editor.tab==='sectors'&&items.length){const last=items.pop(),middle=(last.start+last.end)/2;items.push(newInterval(last.name,last.start,middle),newInterval('S'+(items.length+2),middle,last.end));}else items.push(newInterval('New turn',state.end-state.start<.99?state.start:.2,state.end-state.start<.99?state.end:.25));drawEditor();};
 $('merge-intervals').onclick=mergeIntervals;$('split-interval').onclick=splitInterval;
 $('confirm-corners').onclick=()=>{readEditor();editor.corners.forEach(c=>c.confirmed=true);drawEditor();};
 $('corners-to-sectors').onclick=()=>{readEditor();const corners=[...editor.corners].sort((a,b)=>a.position-b.position);if(!corners.length){editFailure('Add a turn first.');return;}const boundaries=[0,...corners.slice(1).map((c,i)=>(c.position+corners[i].position)/2),1];editor.sectors=corners.map((c,i)=>newInterval(c.name,boundaries[i],boundaries[i+1]));editor.tab='sectors';drawEditor();};
 $('save-track').onclick=saveTrack;
 document.querySelectorAll('[data-close]').forEach(button=>button.onclick=()=>$(button.dataset.close).close());
 $('open-import').onclick=()=>{importDraft=null;importColumns=[];importGateIndex=null;importGood=false;importPreviewNumber++;$('import-mapping').classList.add('hidden');$('import-files').value='';$('commit-import').disabled=true;showImportError('');$('import-dialog').showModal();};
 $('upload-form').onsubmit=uploadFiles;$('add-channel').onclick=()=>addChannelRow();$('preview-import').onclick=previewImport;$('commit-import').onclick=commitImport;
 $('coordinate-mode').onchange=()=>{toggleCoordinateFields();markImportDirty();};
 $('import-track').onchange=()=>{importGateIndex=null;markImportDirty();$('gate-controls').classList.toggle('hidden',!!$('import-track').value);$('new-track-name-label').classList.toggle('hidden',!!$('import-track').value);previewImport();};
 for(const id of ['map-time','map-time-unit','map-latitude','map-longitude','map-x','map-y','map-xy-unit'])$(id).addEventListener('change',markImportDirty);
 for(const id of ['gate-width','gate-angle','gate-flip'])$(id).addEventListener('change',()=>{markImportDirty();clearTimeout(importPreviewTimer);importPreviewTimer=setTimeout(previewImport,150);});
 for(const id of ['import-name','import-track-name'])$(id).addEventListener('input',()=>{});
}
bindControls();
initialise().catch(error=>toast(error.message,true));
