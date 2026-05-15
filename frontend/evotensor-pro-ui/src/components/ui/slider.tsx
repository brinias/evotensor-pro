import React from 'react'
export function Slider({value,onValueChange}:{value:number[];onValueChange:(v:number[])=>void}){const v=value?.[0]??50;return <input type='range' min={0} max={100} value={v} onChange={(e)=>onValueChange([parseInt(e.target.value)])} className='w-full'/>}
