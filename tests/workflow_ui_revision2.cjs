// Executes actual UI functions with fake DOM/transport. No browser/network.
const fs=require('fs'),vm=require('vm'),path=require('path');
let root=__dirname;while(!fs.existsSync(path.join(root,'app')))root=path.dirname(root);
const out=path.join(root,'research/project_assessment_20261023/revision2/workflow_ui');
const baseline=process.argv.includes('--baseline');
const html=fs.readFileSync(path.join(baseline?path.join(out,'baseline_sources'):root,'local_runtime/operation.html'),'utf8');
const results=[];
function extract(start,end){const a=html.indexOf(start),b=html.indexOf(end,a+start.length);if(a<0||b<0)throw Error('Source markers missing '+start);return html.slice(a,b);}
function record(id,description,actual,pass){results.push({id,description,actual,pass:!!pass});}
function node(id,document){
 const classes=new Set();
 return {id,dataset:{},children:[],disabled:false,tabIndex:0,isConnected:true,textContent:'',
  classList:{contains:c=>classes.has(c),add:c=>classes.add(c),remove:c=>classes.delete(c),toggle(c,v){v=v===undefined?!classes.has(c):v;v?classes.add(c):classes.delete(c);}},
  focus(){document.activeElement=this;},toggleAttribute(){},setAttribute(){},getClientRects(){return [1];},
  append(x){this.children.push(x);},replaceChildren(){this.children=[];this.textContent='';},querySelector(){return null;},
  querySelectorAll(){return this.children;},contains(x){return this===x||this.children.includes(x);}};
}
function jobsContext(api){
 const document={activeElement:null,querySelector:()=>nodes.header,createElement:()=>node('new',document)};
 const nodes={};['jobDrawer','workspace','header','jobQuestion','jobCenter'].forEach(id=>nodes[id]=node(id,document));nodes.jobDrawer.classList.add('open');
 const c=vm.createContext({$:id=>nodes[id],document,api,setInterval:()=>101,clearInterval:()=>{},renderJobs:items=>{c.rendered=items;},toast:()=>{},JOB_TERMINAL:new Set(['succeeded','failed','cancelled'])});
 vm.runInContext("let token='old',jobLoading=false,jobPoll=null,jobsSignature='',openJobId='',jobViewGeneration=0,jobReturnFocus=null;",c);
 for(const [a,b] of [['function ensureJobPoll(items){','async function loadJobs(){'],['async function loadJobs(){','function renderJobs(items){'],['function setJobsOpen(open){','$("jobCenter").onclick']])vm.runInContext(extract(a,b),c);
 return {c,nodes,document};
}
async function sendContext(disconnect){
 let lastAnswer,toastText='';const document={};const nodes={message:{value:''},messages:{scrollTop:0,scrollHeight:0}};
 const c=vm.createContext({$:id=>nodes[id]||(nodes[id]={}),TextDecoder,Uint8Array,
  lock:()=>{},toast:t=>{toastText=t;},setThread:()=>{},sessions:async()=>{},
  message:(role,text)=>{const e=node(role,document);e.textContent=text;if(role==='assistant')lastAnswer=e;return e;},
  inkFill:(el,t)=>{el.textContent=t;},inkAppend:(el,t)=>{el.textContent+=t;},
  fetch:async()=>{let count=0;return {ok:true,body:{getReader:()=>({read:async()=>{
   if(count++===0)return {done:false,value:new TextEncoder().encode('event: token\ndata: {"token":"PROVISIONAL"}\n\n')};
   if(disconnect)throw Error('MOCK_TRANSPORT_DISCONNECT');
   if(count===2)return {done:false,value:new TextEncoder().encode('event: complete\ndata: {"response":"PROVISIONAL"}\n\n')};
   return {done:true};}})}};}});
 vm.runInContext("let busy=false,modelReady=true,token='mock',threadId='mock';",c);vm.runInContext(extract('async function send(text){','$("send").onclick'),c);
 await vm.runInContext("send('mock')",c);return {answer:lastAnswer,toast:toastText};
}
(async()=>{
 let resolve;const pending=new Promise(r=>resolve=r);const x=jobsContext(()=>pending);
 const p=vm.runInContext('loadJobs()',x.c);vm.runInContext("setJobsOpen(false);token='new';",x.c);
 resolve({data:{items:[{job_id:'private_old',status:'running'}]}});await p;
 record('U01','旧账号列表晚回包不能渲染或重启关闭后的轮询', {rendered:x.c.rendered,poll:vm.runInContext('jobPoll',x.c)},!x.c.rendered&&vm.runInContext('jobPoll===null',x.c));

 let first,second;let n=0;const y=jobsContext(()=>new Promise(r=>{if(n++===0)first=r;else second=r;}));
 const a=vm.runInContext('loadJobs()',y.c);vm.runInContext('setJobsOpen(false);setJobsOpen(true);',y.c);
 first({data:{items:[]}});await a;const stillLoading=vm.runInContext('jobLoading',y.c);second({data:{items:[]}});await new Promise(r=>setImmediate(r));
 record('U02','旧请求finally不能清除新一轮列表的加载标记',{still_loading:stillLoading},stillLoading);

 let detailResolve;const z=jobsContext(()=>new Promise(r=>detailResolve=r));
 vm.runInContext(extract('async function showJob(jobId,card,button){','async function actJob(jobId,action){'),z.c);
 const card=node('card',z.document),button=node('button',z.document);z.c.card=card;z.c.button=button;
 const detail=vm.runInContext("showJob('old',card,button)",z.c);vm.runInContext("setJobsOpen(false);token='new';",z.c);
 detailResolve({data:{job_id:'old',status:'succeeded',result:{answer:'OLD_PRIVATE'}}});await detail;
 record('U03','旧详情晚回包不能写详情或污染openJobId',{children:card.children.length,open_job:vm.runInContext('openJobId',z.c)},!card.children.length&&vm.runInContext("openJobId==='';",z.c));

 const interrupted=await sendContext(true);record('U04','断流临时正文作废并持久标记，不能作为完整回答复制',{text:interrupted.answer.textContent,dataset:interrupted.answer.dataset},interrupted.answer.dataset.incomplete==='true'&&!interrupted.answer.dataset.raw);
 const complete=await sendContext(false);record('U05','正常SSE完成保持最终正文',{text:complete.answer.textContent,dataset:complete.answer.dataset},complete.answer.textContent==='PROVISIONAL'&&complete.answer.dataset.raw==='PROVISIONAL'&&!complete.answer.dataset.incomplete);

 let fetchResolve;const c=vm.createContext({fetch:()=>new Promise(r=>fetchResolve=r)});vm.runInContext("let token='old';",c);vm.runInContext(extract('async function api(path,options){','function inkAppend(el,text){'),c);
 const old=vm.runInContext("api('/mock')",c);vm.runInContext("token='new';",c);fetchResolve({ok:true,json:async()=>({data:'OLD_PRIVATE'})});let rejected=false;try{await old;}catch{rejected=true;}
 record('U06','公共API适配层拒绝跨账号旧响应',{rejected},rejected);

 const f=jobsContext(async()=>({data:{items:[]}}));const firstButton=node('first',f.document),lastButton=node('last',f.document);f.nodes.jobDrawer.children=[firstButton,lastButton];f.document.activeElement=lastButton;
 f.document.addEventListener=(_name,handler)=>{f.handler=handler;};vm.runInContext(extract('document.addEventListener("keydown",','$("themeToggle").onclick'),f.c);
 let prevented=false;f.handler({key:'Tab',shiftKey:false,preventDefault(){prevented=true;}});
 record('U07','任务弹窗Tab在最后一个控件时回到首控件',{focused:f.document.activeElement.id,prevented},f.document.activeElement===firstButton&&prevented);

 let copied=false;const cp=vm.createContext({navigator:{clipboard:{writeText:async()=>{copied=true;}}},toast:()=>{},clearTimeout:()=>{},setTimeout:()=>0});vm.runInContext(extract('async function copyReply(bubble,button){','async function status(){'),cp);
 cp.bubble={dataset:{raw:'draft',incomplete:'true'},textContent:'draft'};cp.button={classList:{add(){},remove(){}}};await vm.runInContext('copyReply(bubble,button)',cp);
 record('U08','未完成回答不能被复制为有效正文',{copied},!copied);
 const data={baseline,tests:results,summary:{total:results.length,pass:results.filter(r=>r.pass).length,fail:results.filter(r=>!r.pass).length},browser_used:false,network_used:false};
 fs.mkdirSync(out,{recursive:true});fs.writeFileSync(path.join(out,baseline?'remaining_baseline_ui.json':'after_ui.json'),JSON.stringify(data,null,2));console.log(JSON.stringify(data,null,2));process.exitCode=data.summary.fail?1:0;
})().catch(e=>{console.error(e);process.exitCode=1;});
