import fs from 'node:fs';
import { createHash } from 'node:crypto';
const config=JSON.parse(fs.readFileSync(0,'utf8'));
const input=fs.readFileSync(config.input);
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
if(hash(input)!==config.inputSha256)throw Error('Long-path input bytes differ');
const headers={Cookie:config.cookie};
const body=new FormData();body.append('file',new File([input],'科研数据.csv',{type:'text/csv'}));
const created=await fetch(config.api+'/api/v1/projects',{method:'POST',headers,body});
if(created.status!==202)throw Error('Unicode-path file import failed: '+created.status);
const project=await created.json();let workspace;
const deadline=Date.now()+120000;
while(Date.now()<deadline){
 const response=await fetch(config.api+'/api/v1/projects/'+encodeURIComponent(project.projectId)+'/workspace',{headers});
 if(response.ok){workspace=await response.json();if(workspace.preview?.totalRows===12)break;}
 await new Promise(resolve=>setTimeout(resolve,100));
}
if(workspace?.preview?.totalRows!==12)throw Error('Imported long-path data did not reopen with 12 rows');
const exported=await fetch(config.api+config.pdfHref,{headers});
if(!exported.ok)throw Error('Frozen PDF unavailable');
const pdf=Buffer.from(await exported.arrayBuffer());
if(pdf.subarray(0,4).toString()!=='%PDF'||hash(pdf)!==config.pdfSha256)throw Error('Frozen PDF bytes differ');
fs.writeFileSync(config.output,pdf);
if(hash(fs.readFileSync(config.output))!==config.pdfSha256)throw Error('Long-path output bytes differ');
console.log(JSON.stringify({passed:true,projectId:project.projectId,rows:12,inputPathLength:config.input.length,outputPathLength:config.output.length,pdfSha256:config.pdfSha256,scope:'Bundled Node long Chinese file IO and authenticated import/reopen/frozen-export API; no native file dialog'}));
