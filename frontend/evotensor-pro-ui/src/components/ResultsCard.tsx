import React, { useMemo } from 'react'
import Papa from 'papaparse'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'
import { Download, Database } from 'lucide-react'
import { downloadText } from '../utils'

export function ResultsCard({ csv }: { csv: string }) {
  const rows = useMemo(() => (Papa.parse(csv, { header: true }).data as any[]), [csv])
  const headers = useMemo(() => Object.keys(rows[0] || {}), [rows])
  return (
    <Card>
      <CardHeader className='pb-2'><CardTitle className='text-base flex items-center gap-2'><Database className='h-5 w-5'/> Predictions</CardTitle></CardHeader>
      <CardContent>
        <div className='overflow-auto max-h-[420px] border rounded-xl'>
          <table className='w-full text-sm'>
            <thead className='sticky top-0 bg-white'>
              <tr>{headers.map(h => <th key={h} className='text-left px-3 py-2 border-b font-medium'>{h}</th>)}</tr>
            </thead>
            <tbody>
              {rows.map((r,i)=>(<tr key={i} className='hover:bg-gray-50'>{headers.map(h => <td key={h} className='px-3 py-2 border-b'>{String(r[h] ?? '')}</td>)}</tr>))}
            </tbody>
          </table>
        </div>
        <div className='mt-3 flex gap-2'>
          <Button variant='outline' size='sm' onClick={()=>downloadText('predictions.csv', csv)}><Download className='h-4 w-4 mr-1'/>Download CSV</Button>
        </div>
      </CardContent>
    </Card>
  )
}
