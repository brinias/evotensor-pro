export const wait=(ms:number)=>new Promise(res=>setTimeout(res,ms))
export function downloadText(filename:string,text:string){const blob=new Blob([text],{type:'text/csv;charset=utf-8;'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=filename;a.click()}
