import React from 'react'
export function Card({children,className=''}:{children:React.ReactNode;className?:string}){return <div className={`rounded-2xl border bg-white shadow-sm ${className}`}>{children}</div>}
export function CardHeader({children,className=''}:{children:React.ReactNode;className?:string}){return <div className={`px-4 pt-4 ${className}`}>{children}</div>}
export function CardTitle({children,className=''}:{children:React.ReactNode;className?:string}){return <div className={`text-xl font-semibold ${className}`}>{children}</div>}
export function CardContent({children,className=''}:{children:React.ReactNode;className?:string}){return <div className={`px-4 pb-4 ${className}`}>{children}</div>}
