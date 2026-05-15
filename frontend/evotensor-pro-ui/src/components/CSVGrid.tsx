import React, { useEffect, useState } from 'react'
import Papa from 'papaparse'
import { Button } from './ui/button'
import { Trash2, Plus } from 'lucide-react'

export function CSVGrid({ csv, onChange }: { csv: string; onChange: (csv: string)=>void }) {
  const [headers, setHeaders] = useState<string[]>([])
  const [rows, setRows] = useState<string[][]>([])

  useEffect(() => {
    if (!csv.trim()) { setHeaders([]); setRows([]); return }
    const parsed = Papa.parse(csv.trim(), { header: true })
    const hs = (parsed.meta.fields || []) as string[]
    const rs = (parsed.data as any[]).map(r => hs.map(h => (r[h] ?? '')))
    setHeaders(hs); setRows(rs)
  }, [csv])

  const commit = (hs: string[], rs: string[][]) => {
    setHeaders(hs); setRows(rs)
    const records = rs.map(r => Object.fromEntries(hs.map((h, i) => [h, r[i] ?? ''])))
    onChange(Papa.unparse(records))
  }

  const onPaste = (e: React.ClipboardEvent) => {
    const text = e.clipboardData.getData('text/plain')
    if (!text) return
    e.preventDefault()
    const lines = text.replace(/\r/g, '').split('\n').filter(Boolean)
    const data = lines.map(l => l.split(/\t|,/g))
    if (!headers.length) { commit(data[0], data.slice(1)) } else { commit(headers, [...rows, ...data]) }
  }

  const addRow = () => commit(headers, [...rows, Array(headers.length).fill('')])
  const clear = () => onChange('')

  if (!headers.length) return <div className='text-sm text-gray-500'>No headers detected. Paste a table or upload a CSV with headers.</div>

  return (
    <div className='border rounded-xl overflow-hidden'>
      <div className='flex items-center justify-between px-3 py-2 bg-gray-50'>
        <div className='text-sm'>Rows: {rows.length}</div>
        <div className='flex gap-2'>
          <Button size='sm' variant='outline' onClick={addRow}><Plus className='h-4 w-4 mr-1'/>Row</Button>
          <Button size='sm' variant='outline' onClick={clear}><Trash2 className='h-4 w-4 mr-1'/>Clear</Button>
        </div>
      </div>
      <div className='overflow-auto max-h-[420px]' onPaste={onPaste}>
        <table className='w-full text-sm'>
          <thead className='sticky top-0 bg-white z-10'>
            <tr>{headers.map((h,i)=>(<th key={i} className='text-left px-3 py-2 border-b font-medium'>{h}</th>))}</tr>
          </thead>
          <tbody>
            {rows.map((row, rIdx) => (
              <tr key={rIdx} className='hover:bg-gray-50'>
                {headers.map((h, cIdx) => (
                  <td key={cIdx} className='px-3 py-2 border-b align-top'>
                    <div
                      contentEditable
                      suppressContentEditableWarning
                      className='min-w-[120px] outline-none focus:ring-2 ring-indigo-300 rounded px-1'
                      onBlur={(e)=>{
                        const val = (e.currentTarget.textContent ?? '').trim()
                        const copy = rows.map(r => [...r]); copy[rIdx][cIdx]=val; commit(headers, copy)
                      }}
                    >{row[cIdx]}</div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
