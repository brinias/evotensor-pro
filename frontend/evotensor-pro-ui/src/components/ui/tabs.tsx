import React from 'react'
export function Tabs({value,onValueChange,children}:{value:string;onValueChange:(v:string)=>void;children:React.ReactNode}){return <div data-tabs>{children}</div>}
export function TabsList({children,className=''}:{children:React.ReactNode;className?:string}){return <div className={`rounded-xl border bg-gray-50 p-1 ${className}`}>{children}</div>}
export function TabsTrigger({value,active,onClick,children}:{value:string;active?:boolean;onClick?:()=>void;children:React.ReactNode}){return <button onClick={onClick} className={`px-3 py-2 rounded-lg text-sm ${active?'bg-white border shadow':'hover:bg-white/70'}`}>{children}</button>}
export function TabsContent({hidden,children,className=''}:{hidden?:boolean;children:React.ReactNode;className?:string}){return <div className={`${hidden?'hidden':''} ${className}`}>{children}</div>}
