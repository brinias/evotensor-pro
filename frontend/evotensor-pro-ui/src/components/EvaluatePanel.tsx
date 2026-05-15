import React, { useEffect, useMemo, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Label } from './ui/label'
import { Button } from './ui/button'
import type { Project, EvalSpec } from '../types'
import { Play } from 'lucide-react'

type Props = {
  project: Project | null
  spec: EvalSpec
  onSpecChange: (s: EvalSpec) => void
  onRun: () => void
  busy: boolean
  // new: function from parent to fetch CSV headers for a fileId
  getFileColumns: (fileId: string) => Promise<string[]>
}

export function EvaluatePanel({ project, spec, onSpecChange, onRun, busy, getFileColumns }: Props) {
  const files = project?.files || []
  const [predCols, setPredCols] = useState<string[]>([])
  const [gtCols, setGtCols] = useState<string[]>([])

  // Load columns when predictions file changes
  useEffect(() => {
    let canceled = false
    ;(async () => {
      if (!spec.predFileId) { setPredCols([]); return }
      const cols = await getFileColumns(spec.predFileId)
      if (canceled) return
      setPredCols(cols)

      // Auto-pick defaults / propagate if current selections missing
      const next = { ...spec }
      const firstProb = cols.find(c => c.endsWith('_prob'))
      if (!cols.includes(spec.prob_col) && firstProb) {
        next.prob_col = firstProb
        const base = firstProb.replace(/_prob$/,'')
        const candLabel = `${base}_label`
        const candDecision = `${base}_decision`
        if (cols.includes(candLabel)) next.label_col = candLabel
        if (cols.includes(candDecision)) next.decision_col = candDecision
      }
      if (!cols.includes(next.label_col)) {
        const sameBase = next.prob_col?.replace(/_prob$/,'')
        const fallbackLbl = sameBase ? `${sameBase}_label` : cols.find(c => c.endsWith('_label'))
        if (fallbackLbl && cols.includes(fallbackLbl)) next.label_col = fallbackLbl
      }
      if (!cols.includes(next.decision_col)) {
        const sameBase = next.prob_col?.replace(/_prob$/,'')
        const fallbackDec = sameBase ? `${sameBase}_decision` : cols.find(c => c.endsWith('_decision'))
        if (fallbackDec && cols.includes(fallbackDec)) next.decision_col = fallbackDec
      }
      if (JSON.stringify(next) !== JSON.stringify(spec)) onSpecChange(next)
    })()
    return () => { canceled = true }
  }, [spec.predFileId])

  // Load columns when GT file changes
  useEffect(() => {
    let canceled = false
    ;(async () => {
      if (!spec.gtFileId) { setGtCols([]); return }
      const cols = await getFileColumns(spec.gtFileId)
      if (canceled) return
      setGtCols(cols)

      const next = { ...spec }
      if (!cols.includes(spec.seq_col) && cols.includes('sequence')) next.seq_col = 'sequence'
      if (!cols.includes(spec.gt_label_col) && cols.includes('label')) next.gt_label_col = 'label'
      if (JSON.stringify(next) !== JSON.stringify(spec)) onSpecChange(next)
    })()
    return () => { canceled = true }
  }, [spec.gtFileId])

  // Helpers
  const probOptions = useMemo(() => predCols.filter(c => c.endsWith('_prob')), [predCols])
  const labelOptions = useMemo(() => predCols.filter(c => c.endsWith('_label')), [predCols])
  const decisionOptions = useMemo(() => predCols.filter(c => c.endsWith('_decision')), [predCols])

  const onPickProb = (val: string) => {
    const base = val.replace(/_prob$/,'')
    const updates: Partial<EvalSpec> = { prob_col: val }
    const lbl = `${base}_label`
    const dec = `${base}_decision`
    if (labelOptions.includes(lbl)) updates.label_col = lbl
    if (decisionOptions.includes(dec)) updates.decision_col = dec
    onSpecChange({ ...spec, ...updates })
  }

  const onPickLabel = (val: string) => {
    const base = val.replace(/_label$/,'')
    const updates: Partial<EvalSpec> = { label_col: val }
    const prob = `${base}_prob`
    const dec = `${base}_decision`
    if (probOptions.includes(prob)) updates.prob_col = prob
    if (decisionOptions.includes(dec)) updates.decision_col = dec
    onSpecChange({ ...spec, ...updates })
  }

  const onPickDecision = (val: string) => {
    const base = val.replace(/_decision$/,'')
    const updates: Partial<EvalSpec> = { decision_col: val }
    const prob = `${base}_prob`
    const lbl = `${base}_label`
    if (probOptions.includes(prob)) updates.prob_col = prob
    if (labelOptions.includes(lbl)) updates.label_col = lbl
    onSpecChange({ ...spec, ...updates })
  }

  return (
    <Card>
      <CardHeader className='pb-2'><CardTitle className='text-base'>Evaluate (ground truth)</CardTitle></CardHeader>
      <CardContent className='grid grid-cols-1 md:grid-cols-2 gap-4'>
        <div className='space-y-3'>
          <div className='grid grid-cols-3 gap-2 items-center'>
            <Label>Predictions file</Label>
            <select className='col-span-2 h-10 border rounded-xl px-3' value={spec.predFileId} onChange={e=>onSpecChange({ ...spec, predFileId: e.target.value })}>
              <option value=''>Select file</option>
              {files.filter(f=>f.kind==='csv').map(f => (<option key={f.id} value={f.id}>{f.name}</option>))}
            </select>
          </div>

          <div className='grid grid-cols-3 gap-2 items-center'>
            <Label>prob_col</Label>
            <select className='col-span-2 h-10 border rounded-xl px-3' value={spec.prob_col} onChange={e => onPickProb(e.target.value)}>
              {!probOptions.includes(spec.prob_col) && <option value={spec.prob_col}>{spec.prob_col || '—'}</option>}
              {probOptions.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>

          <div className='grid grid-cols-3 gap-2 items-center'>
            <Label>label_col</Label>
            <select className='col-span-2 h-10 border rounded-xl px-3' value={spec.label_col} onChange={e => onPickLabel(e.target.value)}>
              {!labelOptions.includes(spec.label_col) && <option value={spec.label_col}>{spec.label_col || '—'}</option>}
              {labelOptions.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
        </div>

        <div className='space-y-3'>
          <div className='grid grid-cols-3 gap-2 items-center'>
            <Label>decision_col</Label>
            <select className='col-span-2 h-10 border rounded-xl px-3' value={spec.decision_col} onChange={e => onPickDecision(e.target.value)}>
              {!decisionOptions.includes(spec.decision_col) && <option value={spec.decision_col}>{spec.decision_col || '—'}</option>}
              {decisionOptions.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>

          <div className='grid grid-cols-3 gap-2 items-center'>
            <Label>Ground truth file</Label>
            <select className='col-span-2 h-10 border rounded-xl px-3' value={spec.gtFileId} onChange={e=>onSpecChange({ ...spec, gtFileId: e.target.value })}>
              <option value=''>Select file</option>
              {files.filter(f=>f.kind==='csv').map(f => (<option key={f.id} value={f.id}>{f.name}</option>))}
            </select>
          </div>

          <div className='grid grid-cols-3 gap-2 items-center'>
            <Label>seq_col</Label>
            <select className='col-span-2 h-10 border rounded-xl px-3' value={spec.seq_col} onChange={e=>onSpecChange({ ...spec, seq_col: e.target.value })}>
              {!gtCols.includes(spec.seq_col) && <option value={spec.seq_col}>{spec.seq_col || '—'}</option>}
              {gtCols.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>

          <div className='grid grid-cols-3 gap-2 items-center'>
            <Label>gt_label_col</Label>
            <select className='col-span-2 h-10 border rounded-xl px-3' value={spec.gt_label_col} onChange={e=>onSpecChange({ ...spec, gt_label_col: e.target.value })}>
              {!gtCols.includes(spec.gt_label_col) && <option value={spec.gt_label_col}>{spec.gt_label_col || '—'}</option>}
              {gtCols.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>

          <div className='pt-2'><Button onClick={onRun} disabled={busy}><Play className='h-4 w-4 mr-1'/>Run evaluate</Button></div>
        </div>
      </CardContent>
    </Card>
  )
}
