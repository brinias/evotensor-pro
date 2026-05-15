import React, { useEffect, useMemo, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from './ui/card'
import { Label } from './ui/label'
import { Input } from './ui/input'
import { Button } from './ui/button'
import { api } from '../api'
import type { Project } from '../types'

type SimParams = {
  peptidesFileId: string
  plugin: string              // 'mrsa_demo' | 'ecoli_uti_demo' | 'cancer_invitro_demo' | `file:<id>`
  dose_uM: number
  interval_h: number
  n_doses: number
  duration_h: number
  outName: string
}

export function SimulatorPanel(
  { project, onOpenFile }: { project: Project | null, onOpenFile: (fileId: string) => void }
) {
  const token = useMemo(() => localStorage.getItem('pep_token') || '', [])
  const [params, setParams] = useState<SimParams>({
    peptidesFileId: '',
    plugin: 'mrsa_demo',
    dose_uM: 32,
    interval_h: 12,
    n_doses: 2,
    duration_h: 24,
    outName: 'simulation.csv',
  })
  const [jsonPlugins, setJsonPlugins] = useState<{ id: string, name: string }[]>([])

  useEffect(() => {
    if (!project) return
    const firstCSV = (project.files || []).find(f => /\.csv$/i.test(f.name))
    if (firstCSV && !params.peptidesFileId) setParams(p => ({ ...p, peptidesFileId: firstCSV.id }))
    const jsons = (project.files || []).filter(f => /\.json$/i.test(f.name)).map(f => ({ id: f.id, name: f.name }))
    setJsonPlugins(jsons)
  }, [project])

  const run = async () => {
    if (!project) return alert('Pick a project first.')
    if (!params.peptidesFileId) return alert('Select an input CSV.')
    const payload = { projectId: project.id, params }
    const res = await api.simulate(token, payload)
    if (res.fileId) onOpenFile(res.fileId)
    alert(res.message || 'OK')
  }

  return (
    <Card>
      <CardHeader><CardTitle>Lab Simulation</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Predictions CSV to simulate on (from pep_product).">Input CSV</span></Label>
          <select className="col-span-2 h-10 border rounded-xl px-3" value={params.peptidesFileId} onChange={e => setParams({ ...params, peptidesFileId: e.target.value })} title="Choose the predictions CSV">
            <option value="">— choose —</option>
            {project?.files.filter(f => /\.csv$/i.test(f.name)).map(f => <option key={f.id} value={f.id}>{f.name}</option>)}
          </select>
        </div>

        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Pick a builtin disease model or a JSON file you saved.">Plugin</span></Label>
          <select className="col-span-2 h-10 border rounded-xl px-3" value={params.plugin} onChange={e => setParams({ ...params, plugin: e.target.value })} title="Disease behavior definition">
            <option value="mrsa_demo">mrsa_demo (builtin)</option>
            <option value="ecoli_uti_demo">ecoli_uti_demo (builtin)</option>
            <option value="cancer_invitro_demo">cancer_invitro_demo (builtin)</option>
            {jsonPlugins.length > 0 && <option disabled>────────</option>}
            {jsonPlugins.map(p => <option key={p.id} value={`file:${p.id}`}>{p.name}</option>)}
          </select>
        </div>

        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Bolus dose per administration (μM).">Dose (μM)</span></Label>
          <Input className="col-span-2" value={params.dose_uM} onChange={e => setParams({ ...params, dose_uM: parseFloat(e.target.value) || 0 })} title="e.g., 32 μM" />
        </div>
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Time between doses (hours).">Interval (h)</span></Label>
          <Input className="col-span-2" value={params.interval_h} onChange={e => setParams({ ...params, interval_h: parseFloat(e.target.value) || 0 })} title="e.g., 12 h" />
        </div>
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Number of doses to administer."># Doses</span></Label>
          <Input className="col-span-2" value={params.n_doses} onChange={e => setParams({ ...params, n_doses: parseInt(e.target.value) || 0 })} title="e.g., 2" />
        </div>
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Total simulation horizon (hours).">Duration (h)</span></Label>
          <Input className="col-span-2" value={params.duration_h} onChange={e => setParams({ ...params, duration_h: parseFloat(e.target.value) || 0 })} title="e.g., 24 h" />
        </div>
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Filename for the result CSV saved under the project.">Output name</span></Label>
          <Input className="col-span-2" value={params.outName} onChange={e => setParams({ ...params, outName: e.target.value })} title="e.g., ecoli_sim.csv" />
        </div>

        <div className="pt-2">
          <Button onClick={run}>
            <span title="Run labsim_demo.py with the selected settings">Run simulation</span>
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
