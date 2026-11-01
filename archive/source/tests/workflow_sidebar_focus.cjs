// Actual sidebar handlers with fake DOM/transport. No browser or real service.
const fs=require('fs'),vm=require('vm'),path=require('path');
let root=__dirname;while(!fs.existsSync(path.join(root,'app')))root=path.dirname(root);
const out=path.join(root,'research/project_assessment_20261023/revision2/workflow_ui/sidebar_focus_20261024');
const baseline=process.argv.includes('--baseline');
const html=fs.readFileSync(baseline?path.join(out,'before.html'):path.join(root,'local_runtime/operation.html'),'utf8');
function source(start,end){const a=html.indexOf(start),b=html.indexOf(end,a+start.length);if(a<0||b<0)throw Error('Source marker missing '+start);return html.slice(a,b);}
function setup(){
 const document={activeElement:null};const nodes={};
 function node(id){const classes=new Set();return {id,children:[],dataset:{},attrs:{},inert:false,
  classList:{contains:c=>classes.has(c),add:c=>classes.add(c),remove:c=>classes.delete(c),toggle(c,v){v=v===undefined?!classes.has(c):v;v?classes.add(c):classes.delete(c);return v;}},
  setAttribute(k,v){this.attrs[k]=v;},focus(){document.activeElement=this;},append(...items){this.children.push(...items);},replaceChildren(){this.children=[];},
  contains(target){return this===target||this.children.some(child=>child.contains(target));},querySelector(selector){return this.children.find(child=>selector==='.group-head'?/Toggle$/.test(child.id):/Contents$/.test(child.id));}};}
 for(const id of ['sessionGroup','sessionToggle','sessionContents','sessions','archiveGroup','archiveToggle','archiveContents','archivedSessions','archiveCount','archiveOld','archiveEmpty','body'])nodes[id]=node(id);
 nodes.sessionGroup.children=[nodes.sessionToggle,nodes.sessionContents];nodes.sessionContents.children=[nodes.sessions];nodes.sessions.children=[node('sessionButton')];nodes.sessionGroup.classList.add('open');
 nodes.archiveGroup.children=[nodes.archiveToggle,nodes.archiveContents];nodes.archiveContents.children=[nodes.archivedSessions];nodes.archivedSessions.children=[node('archiveButton')];
 document.activeElement=nodes.body;nodes.sessionToggle.attrs['aria-expanded']='true';nodes.archiveToggle.attrs['aria-expanded']='false';
 if(html.includes('id="sessionContents"')){nodes.sessionContents.attrs['aria-hidden']='false';nodes.archiveContents.attrs['aria-hidden']='true';nodes.archiveContents.inert=true;}
 const media={matches:false,addEventListener(_name,handler){this.handler=handler;}};
 const c=vm.createContext({$:id=>nodes[id],document,matchMedia:()=>media,api:async()=>({data:c.items||[]}),sessionRow:()=>node('row'),emptySessions:()=>node('empty'),isStale:()=>false,lock:()=>{},toast:()=>{}});
 vm.runInContext("let busy=false,threadId='active';",c);
 if(html.includes('function setSessionGroupOpen('))vm.runInContext(source('function setSessionGroupOpen(','async function sessions(){'),c);
 vm.runInContext(source('async function sessions(){','async function setArchived('),c);
 vm.runInContext(source('$("sessionToggle").onclick=','status();\n'),c);
 return {c,nodes,document,media};
}
const results=[];
function add(id,description,actual,pass){results.push({id,description,actual,pass:!!pass});}
function state(x,kind){const group=x.nodes[kind+'Group'],toggle=x.nodes[kind+'Toggle'],content=x.nodes[kind+'Contents'];return {open:group.classList.contains('open'),inert:content.inert,expanded:toggle.attrs['aria-expanded'],hidden:content.attrs['aria-hidden'],focused:x.document.activeElement.id};}
(async()=>{
 let x=setup();x.nodes.sessionToggle.onclick();let actual=state(x,'session');add('SIDE01','手动折叠同步inert与ARIA',actual,!actual.open&&actual.inert&&actual.expanded==='false'&&actual.hidden==='true');
 x.nodes.sessionToggle.onclick();actual=state(x,'session');add('SIDE02','重新展开恢复可访问性',actual,actual.open&&!actual.inert&&actual.expanded==='true'&&actual.hidden==='false');
 x=setup();x.document.activeElement=x.nodes.sessions.children[0];x.media.matches=true;x.media.handler();actual=state(x,'session');add('SIDE03','切换窄屏折叠时焦点移出隐藏内容',actual,actual.inert&&actual.focused==='sessionToggle'&&!actual.open);
 x=setup();vm.runInContext(source('async function setArchived(','async function deleteSession('),x.c);vm.runInContext('sessions=async()=>{};',x.c);await vm.runInContext("setArchived('active',true)",x.c);actual=state(x,'archive');add('SIDE04','归档当前会话自动展开时同时解除inert',actual,actual.open&&!actual.inert&&actual.expanded==='true'&&actual.hidden==='false');
 x=setup();x.nodes.sessionToggle.onclick();x.c.items=[{thread_id:'one'},{thread_id:'two',archived:true}];await vm.runInContext('sessions()',x.c);actual=state(x,'session');add('SIDE05','列表刷新保持折叠区不可聚焦',actual,!actual.open&&actual.inert&&actual.hidden==='true');
 x=setup();x.nodes.archiveToggle.onclick();x.document.activeElement=x.nodes.archivedSessions.children[0];x.c.items=[];await vm.runInContext('sessions()',x.c);actual=state(x,'archive');add('SIDE06','最后一条归档消失时关闭区域并移出焦点',{...actual,group_hidden:x.nodes.archiveGroup.classList.contains('hidden')},!actual.open&&actual.inert&&actual.focused==='sessionToggle'&&x.nodes.archiveGroup.classList.contains('hidden'));
 add('SIDE07','初始HTML声明控制区域与归档内容隐藏状态',{session_control:html.includes('aria-controls="sessionContents"'),archive_control:html.includes('aria-controls="archiveContents"'),archive_inert:/id="archiveContents"[^>]*aria-hidden="true"[^>]*inert/.test(html)},html.includes('aria-controls="sessionContents"')&&html.includes('aria-controls="archiveContents"')&&/id="archiveContents"[^>]*aria-hidden="true"[^>]*inert/.test(html));
 const data={baseline,tests:results,summary:{total:results.length,pass:results.filter(x=>x.pass).length,fail:results.filter(x=>!x.pass).length},browser_used:false,network_used:false};fs.mkdirSync(out,{recursive:true});fs.writeFileSync(path.join(out,baseline?'baseline_tests.json':'after_tests.json'),JSON.stringify(data,null,2));console.log(JSON.stringify(data,null,2));process.exitCode=data.summary.fail?1:0;
})().catch(e=>{console.error(e);process.exitCode=1;});
