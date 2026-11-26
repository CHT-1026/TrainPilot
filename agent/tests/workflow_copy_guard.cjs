// Actual message button + copy handler, fake DOM/clipboard; no browser/network.
const fs=require('fs'),vm=require('vm'),path=require('path');
let root=__dirname;while(!fs.existsSync(path.join(root,'app')))root=path.dirname(root);
const html=fs.readFileSync(path.join(root,'local_runtime/operation.html'),'utf8');
const results=[];
function extract(start,end){const a=html.indexOf(start),b=html.indexOf(end,a+1);if(a<0||b<0)throw Error('Missing source markers');return html.slice(a,b);}
function element(tag){const classes=new Set();return {tag,children:[],dataset:{},style:{},attributes:{},textContent:'',
 append(...items){this.children.push(...items);},setAttribute(k,v){this.attributes[k]=v;},
 classList:{add:c=>classes.add(c),remove:c=>classes.delete(c),contains:c=>classes.has(c)}};}
let copies=[],fallbacks=0,toastText='';const messages=element('messages');
const document={createElement:element,execCommand:()=>{fallbacks++;},body:element('body')};
const c=vm.createContext({document,$:()=>messages,toast:text=>{toastText=text;},navigator:{clipboard:{writeText:async text=>{copies.push(text);}}},clearTimeout:()=>{},setTimeout:()=>0});
vm.runInContext(extract('async function copyReply(bubble,button){','async function status(){'),c);
vm.runInContext(extract('function message(role,text){','function setThread(id){'),c);
(async()=>{
 const partial=vm.runInContext("message('assistant','partial')",c);partial.dataset.raw='STALE_DRAFT';partial.dataset.incomplete='true';
 const partialButton=messages.children.at(-1).children.find(n=>n.className==='copy');await partialButton.onclick();
 results.push({id:'COPY01',description:'实际消息复制按钮点击后，断流正文不写系统剪贴板或旧兼容剪贴板',actual:{copies:[...copies],fallbacks,toast:toastText,done:partialButton.classList.contains('done')},pass:copies.length===0&&fallbacks===0&&!!toastText&&!partialButton.classList.contains('done')});
 const full=vm.runInContext("message('assistant','完整正文')",c);full.dataset.raw='完整正文';
 const fullButton=messages.children.at(-1).children.find(n=>n.className==='copy');await fullButton.onclick();
 results.push({id:'COPY02',description:'正常完整消息的实际按钮仍能复制正文',actual:{copies:[...copies],done:fullButton.classList.contains('done')},pass:copies.length===1&&copies[0]==='完整正文'&&fullButton.classList.contains('done')});
 const css=(html.match(/\.copy\{([^}]+)\}/)||[])[1]||'';
 results.push({id:'COPY03',description:'复制按钮不依赖hover才接收点击，处于独立预留行和显式层级（静态检查）',actual:{copy_css:css,reserved_header:/\.assistant \.who\{[^}]*min-height:32px/.test(html)},pass:/pointer-events:auto/.test(css)&&/opacity:1/.test(css)&&/z-index:1/.test(css)&&/top:0(?:;|$)/.test(css)&&/\.assistant \.who\{[^}]*min-height:32px/.test(html),mode:'static_css_not_browser_hit_test'});
 const data={tests:results,summary:{total:results.length,pass:results.filter(x=>x.pass).length,fail:results.filter(x=>!x.pass).length},browser_used:false,network_used:false};
 const out=path.join(root,'research/project_assessment_20261023/revision2/workflow_ui/copy_guard_results.json');fs.writeFileSync(out,JSON.stringify(data,null,2));console.log(JSON.stringify(data,null,2));process.exitCode=data.summary.fail?1:0;
})().catch(e=>{console.error(e);process.exitCode=1;});
