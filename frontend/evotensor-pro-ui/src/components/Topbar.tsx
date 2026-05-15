import React from 'react'
import { Button } from './ui/button'
import { Shield } from 'lucide-react'
export function Topbar({ email, onLogout }:{ email:string; onLogout:()=>void }){
  return(<div className='w-full border-b bg-white/80 backdrop-blur'>
    <div className='max-w-[1400px] mx-auto px-4 py-3 flex items-center justify-between'>
      <div className='flex items-center gap-2'><Shield className='h-5 w-5'/><span className='font-semibold'>Evotensor PRO</span><span className='text-xs text-gray-500'>production</span></div>
      <div className='flex items-center gap-3'><span className='text-sm text-gray-600 hidden sm:inline'>{email}</span><Button size='sm' variant='outline' onClick={onLogout}>Logout</Button></div>
    </div></div>)
}
