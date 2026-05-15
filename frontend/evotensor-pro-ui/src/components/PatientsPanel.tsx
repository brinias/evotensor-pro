import React from 'react'
import { Card, CardHeader, CardTitle, CardContent } from './ui/card'
import { Button } from './ui/button'
import { Label } from './ui/label'
import type { Project } from '../types'

export function PatientsPanel(
  { project, onOpenFile, csvBuffer, onSaveAs }:
  { project: Project | null, onOpenFile: (id: string) => void, csvBuffer: string, onSaveAs: () => void }
) {
  const files = (project?.files || []).filter(f => /\.json$/i.test(f.name))

  return (
    <Card>
      <CardHeader><CardTitle>Disease/Patient JSON files</CardTitle></CardHeader>
      <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-1 max-h-[360px] overflow-auto pr-1">
          <Label className="text-xs block mb-1">
            <span title="Only .json files are listed here.">Files</span>
          </Label>
          {files.map((f) => (
            <div key={f.id} className="flex items-center justify-between gap-2 px-2 py-1 rounded hover:bg-gray-50">
              <button className="text-left flex-1 truncate" onClick={() => onOpenFile(f.id)} title="Open this file in the main editor">
                <div className="text-sm font-medium truncate">{f.name}</div>
                <div className="text-xs text-gray-500">{(f.size / 1024).toFixed(1)} KB • {new Date(f.createdAt).toLocaleString()}</div>
              </button>
            </div>
          ))}
          {!files.length && <div className="text-sm text-gray-500">No JSON files yet.</div>}
        </div>
        <div className="space-y-2">
          <div className="text-sm text-gray-600">Open a JSON file to preview/edit in the main editor. You can also save the current editor buffer as a new file.</div>
          <Button size="sm" onClick={onSaveAs}>
            <span title="Save the current editor content as a new file under this project">Save current buffer as…</span>
          </Button>
          <pre className="bg-gray-50 p-3 rounded-lg overflow-auto text-xs max-h-[260px]" title="Preview (first ~5000 chars) of the current editor buffer">
            {(csvBuffer || '').slice(0, 5000) || 'Open a file from the left…'}
          </pre>
        </div>
      </CardContent>
    </Card>
  )
}
