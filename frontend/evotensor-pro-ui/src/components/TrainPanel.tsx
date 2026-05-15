import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Label } from './ui/label'
import { Input } from './ui/input'
import { Button } from './ui/button'
import { Brain, Play } from 'lucide-react'
import type { Project, TrainSpec } from '../types'

export function TrainPanel({ project, spec, onSpecChange, onRun, busy }:
  { project: Project | null; spec: TrainSpec; onSpecChange: (s: TrainSpec)=>void; onRun: ()=>void; busy: boolean }) {
  const files = project?.files || []
  return (
    <Card>
      <CardHeader className='pb-2'><CardTitle className='text-base flex items-center gap-2'><Brain className='h-5 w-5'/> Train a head</CardTitle></CardHeader>
      <CardContent className='grid grid-cols-1 md:grid-cols-2 gap-4'>
        <div className='space-y-3'>
          <div className='grid grid-cols-3 gap-2 items-center'>
            <Label>Type</Label>
            <select className='col-span-2 h-10 border rounded-xl px-3' value={spec.type} onChange={e=>onSpecChange({ ...spec, type: e.target.value as any })}>
              <option value='classification'>classification</option>
              <option value='regression'>regression</option>
            </select>
          </div>
          <div className='grid grid-cols-3 gap-2 items-center'>
            <Label>Task name</Label>
            <Input className='col-span-2' value={spec.task_name} onChange={e=>onSpecChange({ ...spec, task_name: e.target.value })} />
          </div>
          <div className='grid grid-cols-3 gap-2 items-center'>
            <Label>Dataset</Label>
            <select className='col-span-2 h-10 border rounded-xl px-3' value={spec.datasetId} onChange={e=>{
              const f = files.find(x => x.id === e.target.value); onSpecChange({ ...spec, datasetId: e.target.value, datasetName: f?.name || '' })
            }}>
              <option value=''>Select file</option>
              {files.filter(f=>f.kind==='csv').map(f => (<option key={f.id} value={f.id}>{f.name}</option>))}
            </select>
          </div>
          <div className='grid grid-cols-3 gap-2 items-center'>
            <Label>seq_col</Label>
            <Input className='col-span-2' value={spec.seq_col} onChange={e=>onSpecChange({ ...spec, seq_col: e.target.value })} />
          </div>
          {spec.type === 'classification' ? (
            <div className='grid grid-cols-3 gap-2 items-center'>
              <Label>label_col</Label>
              <Input className='col-span-2' value={spec.label_col} onChange={e=>onSpecChange({ ...spec, label_col: e.target.value })} />
            </div>
          ) : (
            <div className='grid grid-cols-3 gap-2 items-center'>
              <Label>target_col</Label>
              <Input className='col-span-2' value={spec.target_col} onChange={e=>onSpecChange({ ...spec, target_col: e.target.value })} />
            </div>
          )}
        </div>
        <div className='space-y-3'>
          <div className='grid grid-cols-3 gap-2 items-center'>
            <Label>feat_mode</Label>
            <select className='col-span-2 h-10 border rounded-xl px-3' value={spec.feat_mode} onChange={e=>onSpecChange({ ...spec, feat_mode: e.target.value as any })}>
              <option value='fuse'>fuse</option>
              <option value='embed'>embed</option>
              <option value='scalars'>scalars</option>
            </select>
          </div>
          {spec.type === 'classification' && <>
            <div className='grid grid-cols-3 gap-2 items-center'>
              <Label>calibration</Label>
              <select className='col-span-2 h-10 border rounded-xl px-3' value={spec.calibration} onChange={e=>onSpecChange({ ...spec, calibration: e.target.value as any })}>
                <option value='sigmoid'>sigmoid</option>
                <option value='platt'>platt</option>
                <option value='isotonic'>isotonic</option>
                <option value='none'>none</option>
              </select>
            </div>
            <div className='grid grid-cols-3 gap-2 items-center'>
              <Label>cv_folds</Label>
              <Input className='col-span-2' value={spec.cv_folds} onChange={e=>onSpecChange({ ...spec, cv_folds: parseInt(e.target.value)||5 })} />
            </div>
            <div className='grid grid-cols-3 gap-2 items-center'>
              <Label>precision_target</Label>
              <Input className='col-span-2' value={spec.precision_target} onChange={e=>onSpecChange({ ...spec, precision_target: parseFloat(e.target.value)||0.8 })} />
            </div>
          </>}
          <div className='pt-2'><Button onClick={onRun} disabled={busy}><Play className='h-4 w-4 mr-1'/>{busy? 'Training...':'Train head'}</Button></div>
        </div>
      </CardContent>
    </Card>
  )
}
