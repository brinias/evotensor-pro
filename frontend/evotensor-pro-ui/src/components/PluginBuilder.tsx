import React, { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from './ui/card'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Button } from './ui/button'
import { Save } from 'lucide-react'
import { api } from '../api'
import type { Project } from '../types'

type TPlugin = {
  name: string
  type: 'bacteria' | 'cancer' | string
  readout: string
  growth_model?: string
  params: { r_per_h?: number; K?: number; N0?: number; Viab0?: number }
  pk: { model: string; kel_per_h: number }
  pd: {
    ec50: { from: string[]; min_uM?: number; max_uM?: number; mic_divisor?: number }
    kmax?: { base: number; scale_prob: number; from?: string[] }
    hill: { base: number; from?: string[] }
    emax?: { base: number; from?: string[] } // cancer only
  }
  safety: { tox_from: string[]; alpha: number; beta: number }
  endpoint_hours: number
}

export function PluginBuilder({ project, onSaved }: { project: Project | null; onSaved: () => void }) {
  const [filename, setFilename] = useState('plugin.json')
  const [plugin, setPlugin] = useState<TPlugin>({
    name: '',
    type: 'bacteria',
    readout: 'CFU_over_time',
    growth_model: 'logistic',
    params: { r_per_h: 0.1, K: 1e9, N0: 1e6 },
    pk: { model: 'one_compartment', kel_per_h: 0.05 },
    pd: {
      ec50: { from: ['MIC_pred_uM','AMP_proxy','AMP_activity_prob'], min_uM: 0.25, max_uM: 256, mic_divisor: 4 },
      kmax: { base: 0.025, scale_prob: 0.045, from: ['AMP_activity_prob','amp_head_prob','prob'] },
      hill: { base: 1.3 },
      // emax: { base: 0.9 }, // only if type==='cancer'
    },
    safety: { tox_from: ['Aggregation_*','gravy_heur'], alpha: 0.2, beta: 1.0 },
    endpoint_hours: 24,
  })

  const update = (path: string[], value: any) => {
    setPlugin(prev => {
      const copy: any = JSON.parse(JSON.stringify(prev))
      let ref = copy
      for (let i = 0; i < path.length - 1; i++) {
        const k = path[i]
        ref[k] = { ...ref[k] }
        ref = ref[k]
      }
      ref[path[path.length - 1]] = value
      return copy
    })
  }

  const parseList = (s: string) => s.split(',').map(x => x.trim()).filter(Boolean)

  const save = async () => {
    if (!project) return alert('Select a project first.')
    const token = localStorage.getItem('pep_token') || ''
    const text = JSON.stringify(plugin, null, 2)
    await api.uploadCSV(token, project.id, filename, text)
    onSaved?.()
    alert(`Saved: ${filename}`)
  }

  return (
    <Card>
      <CardHeader><CardTitle>Create Disease Plugin (JSON).</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Filename to store under the project.">File name</span></Label>
          <Input className="col-span-2" value={filename} onChange={e => setFilename(e.target.value.replace(/\s+/g, '_'))} title="Use .json extension" />
        </div>

        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Human-readable name of this disease configuration.">name</span></Label>
          <Input className="col-span-2" value={plugin.name} onChange={e => update(['name'], e.target.value)} title="e.g., E.coli UTI (lab study)" />
        </div>

        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Model type: bacteria (CFU) or cancer (viability).">type</span></Label>
          <Input className="col-span-2" value={plugin.type} onChange={e => update(['type'], e.target.value)} title="bacteria | cancer" />
        </div>

        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Primary readout axis for plots/results.">readout</span></Label>
          <Input className="col-span-2" value={plugin.readout} onChange={e => update(['readout'], e.target.value)} title="e.g., CFU_over_time or Viability_vs_time" />
        </div>

        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Growth model keyword (informational for now).">growth_model</span></Label>
          <Input className="col-span-2" value={plugin.growth_model || ''} onChange={e => update(['growth_model'], e.target.value)} title="logistic (typical)" />
        </div>

        {/* Params */}
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Bacteria: intrinsic growth rate per hour.">params.r_per_h</span></Label>
          <Input type="number" className="col-span-2" value={plugin.params.r_per_h ?? 0} onChange={e => update(['params','r_per_h'], parseFloat(e.target.value)||0)} title="Higher → faster growth (bacteria only)" />
        </div>
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Bacteria: carrying capacity (maximum CFU).">params.K</span></Label>
          <Input type="number" className="col-span-2" value={plugin.params.K ?? 0} onChange={e => update(['params','K'], parseFloat(e.target.value)||0)} title="e.g., 1e9 (bacteria only)" />
        </div>
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Bacteria: initial CFU.">params.N0</span></Label>
          <Input type="number" className="col-span-2" value={plugin.params.N0 ?? 0} onChange={e => update(['params','N0'], parseFloat(e.target.value)||0)} title="e.g., 1e6 (bacteria only)" />
        </div>
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Cancer: initial viability (fraction).">params.Viab0</span></Label>
          <Input type="number" className="col-span-2" value={plugin.params.Viab0 ?? 1.0} onChange={e => update(['params','Viab0'], parseFloat(e.target.value)||0)} title="Typically starts at 1.0 (cancer only)" />
        </div>

        {/* PK */}
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Pharmacokinetics model keyword.">pk.model</span></Label>
          <Input className="col-span-2" value={plugin.pk.model} onChange={e => update(['pk','model'], e.target.value)} title="one_compartment" />
        </div>
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Elimination rate constant per hour; larger → faster decay.">pk.kel_per_h</span></Label>
          <Input type="number" className="col-span-2" value={plugin.pk.kel_per_h} onChange={e => update(['pk','kel_per_h'], parseFloat(e.target.value)||0)} title="e.g., 0.05 per hour" />
        </div>

        {/* PD - EC50 */}
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Priority list of columns (or prefixes with *) to derive EC50 (μM).">pd.ec50.from</span></Label>
          <Input className="col-span-2" value={plugin.pd.ec50.from.join(',')} onChange={e => update(['pd','ec50','from'], parseList(e.target.value))} title='E.g., "MIC_pred_uM, MIC_*, AMP_activity_prob"' />
        </div>
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Minimum EC50 (μM) used when mapping from probability.">pd.ec50.min_uM</span></Label>
          <Input type="number" className="col-span-2" value={plugin.pd.ec50.min_uM ?? 0.25} onChange={e => update(['pd','ec50','min_uM'], parseFloat(e.target.value)||0)} title="Default 0.25 μM" />
        </div>
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Maximum EC50 (μM) used when mapping from probability.">pd.ec50.max_uM</span></Label>
          <Input type="number" className="col-span-2" value={plugin.pd.ec50.max_uM ?? 256} onChange={e => update(['pd','ec50','max_uM'], parseFloat(e.target.value)||0)} title="Default 256 μM" />
        </div>
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="EC50 ≈ MIC / mic_divisor when EC50 source is a MIC in μM.">pd.ec50.mic_divisor</span></Label>
          <Input type="number" className="col-span-2" value={plugin.pd.ec50.mic_divisor ?? 4} onChange={e => update(['pd','ec50','mic_divisor'], parseFloat(e.target.value)||0)} title="Default 4.0" />
        </div>

        {/* PD - Kmax */}
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Base kill rate constant per hour.">pd.kmax.base</span></Label>
          <Input type="number" className="col-span-2" value={plugin.pd.kmax?.base ?? 0} onChange={e => update(['pd','kmax','base'], parseFloat(e.target.value)||0)} title="Bacteria only" />
        </div>
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Scale factor multiplied by a probability score (0..1).">pd.kmax.scale_prob</span></Label>
          <Input type="number" className="col-span-2" value={plugin.pd.kmax?.scale_prob ?? 0} onChange={e => update(['pd','kmax','scale_prob'], parseFloat(e.target.value)||0)} title="Bacteria only" />
        </div>
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Priority list of probability columns (0..1) to drive kmax.">pd.kmax.from</span></Label>
          <Input className="col-span-2" value={(plugin.pd.kmax?.from || []).join(',')} onChange={e => update(['pd','kmax','from'], parseList(e.target.value))} title='E.g., "amp_head_prob, AMP_activity_prob, prob"' />
        </div>

        {/* PD - Hill + Emax */}
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Default Hill slope for concentration-response curve.">pd.hill.base</span></Label>
          <Input type="number" className="col-span-2" value={plugin.pd.hill.base} onChange={e => update(['pd','hill','base'], parseFloat(e.target.value)||0)} title="Typical range 1.0–2.0" />
        </div>
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Optional column(s) to read Hill directly per peptide.">pd.hill.from</span></Label>
          <Input className="col-span-2" value={(plugin.pd.hill.from || []).join(',')} onChange={e => update(['pd','hill','from'], parseList(e.target.value))} title="If provided, overrides base where present" />
        </div>
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="(Cancer) Max fractional effect (0..1).">pd.emax.base</span></Label>
          <Input type="number" className="col-span-2" value={plugin.pd.emax?.base ?? 0.9} onChange={e => update(['pd','emax','base'], parseFloat(e.target.value)||0)} title="Cancer model only" />
        </div>
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="(Cancer) Column(s) to read Emax directly per peptide.">pd.emax.from</span></Label>
          <Input className="col-span-2" value={(plugin.pd.emax?.from || []).join(',')} onChange={e => update(['pd','emax','from'], parseList(e.target.value))} title="Cancer model only" />
          </div>

        {/* Safety */}
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Columns (or prefixes with *) contributing to toxicity penalty.">safety.tox_from</span></Label>
          <Input className="col-span-2" value={plugin.safety.tox_from.join(',')} onChange={e => update(['safety','tox_from'], parseList(e.target.value))} title='E.g., "cpp_binary_is_toxic, Aggregation_*"' />
        </div>
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Scaling of toxicity signal.">safety.alpha</span></Label>
          <Input type="number" className="col-span-2" value={plugin.safety.alpha} onChange={e => update(['safety','alpha'], parseFloat(e.target.value)||0)} title="Higher → larger penalty" />
        </div>
        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Sharpening factor inside sigmoid(tox * beta).">safety.beta</span></Label>
          <Input type="number" className="col-span-2" value={plugin.safety.beta} onChange={e => update(['safety','beta'], parseFloat(e.target.value)||0)} title="Higher → steeper response" />
        </div>

        <div className="grid grid-cols-3 gap-2 items-center">
          <Label><span title="Default simulation horizon if duration is not provided.">endpoint_hours</span></Label>
          <Input type="number" className="col-span-2" value={plugin.endpoint_hours} onChange={e => update(['endpoint_hours'], parseFloat(e.target.value)||0)} title="Hours" />
        </div>

        <div className="pt-2">
          <Button onClick={save}>
            <Save className="h-4 w-4 mr-1" />
            <span title="Store this JSON under the current project">Save JSON</span>
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
