// src/components/HeadsLibrary.tsx
import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'
import { Trash2 } from 'lucide-react'
import type { Project } from '../types' // <- FIX: σωστό σχετικό path

type HeadRow = {
  id: string
  task_name: string
  type: string
  createdAt: string | number
}

type Props = {
  project: Project | null
  onDelete?: (headId: string) => void
}

export function HeadsLibrary({ project, onDelete }: Props) {
  const heads = (project?.heads || []) as unknown as HeadRow[]

  return (
    <Card>
      <CardHeader className='pb-2'>
        <CardTitle className='text-base'>Heads ({heads.length})</CardTitle>
      </CardHeader>
      <CardContent>
        {heads.length === 0 ? (
          <div className='text-sm text-gray-500'>
            Δεν υπάρχουν custom heads ακόμα. Τρέξε Train και μετά κάνε refresh.
          </div>
        ) : (
          <div className='overflow-auto border rounded-xl'>
            <table className='w-full text-sm'>
              <thead className='bg-white sticky top-0'>
                <tr>
                  <th className='text-left px-3 py-2 border-b font-medium'>Task name</th>
                  <th className='text-left px-3 py-2 border-b font-medium'>Type</th>
                  <th className='text-left px-3 py-2 border-b font-medium'>Created</th>
                  <th className='text-left px-3 py-2 border-b font-medium w-[1%]'>Actions</th>
                </tr>
              </thead>
              <tbody>
                {heads.map((h: HeadRow) => (
                  <tr key={h.id} className='hover:bg-gray-50'>
                    <td className='px-3 py-2 border-b font-medium truncate'>{h.task_name}</td>
                    <td className='px-3 py-2 border-b'>{h.type}</td>
                    <td className='px-3 py-2 border-b'>
                      {new Date(h.createdAt).toLocaleString()}
                    </td>
                    <td className='px-3 py-2 border-b'>
                      <div className='flex gap-1'>
                        {onDelete && (
                          <Button
                            size='icon'
                            variant='destructive'
                            onClick={() => onDelete(h.id)}
                            aria-label='Delete head'
                          >
                            <Trash2 className='h-4 w-4' />
                          </Button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
