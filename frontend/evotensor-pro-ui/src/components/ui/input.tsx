import React from 'react'
export function Input(props:React.InputHTMLAttributes<HTMLInputElement>){return <input {...props} className={`h-10 px-3 rounded-xl border w-full ${props.className||''}`}/>} 
