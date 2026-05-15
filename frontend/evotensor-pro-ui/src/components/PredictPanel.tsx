import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Label } from './ui/label'
import { Input } from './ui/input'
import { Slider } from './ui/slider'
import { Switch } from './ui/switch'
import { Button } from './ui/button'
import { Brain, Play } from 'lucide-react'
import type { PredictParams, Tier } from '../types'

export function PredictPanel({ tier, params, onParamsChange, onRun, busy }:
  { tier: Tier; params: PredictParams; onParamsChange: (p: PredictParams)=>void; onRun: ()=>void; busy: boolean }) {
  return (
    <Card>
      <CardHeader className='pb-2'><CardTitle className='text-base flex items-center gap-2'><Brain className='h-5 w-5'/> Predict ({tier})</CardTitle></CardHeader>
      <CardContent className='grid grid-cols-1 md:grid-cols-2 gap-4'>
        <div className='space-y-3'>
          <div className='grid grid-cols-3 gap-2 items-center'>
            <Label>Decision policy</Label>
            <select className='col-span-2 h-10 border rounded-xl px-3' value={params.decision_policy} onChange={e=>onParamsChange({ ...params, decision_policy: e.target.value as any })}>
              <option value='none'>none</option>
              <option value='conformal'>conformal</option>
              <option value='margin_only'>margin_only</option>
            </select>
          </div>

          <div className='grid grid-cols-3 gap-2 items-center'>
            <Label>Threshold</Label>
            <div className='col-span-2 flex items-center gap-3'>
              <Slider value={[Math.round((params.thr_override ?? 0.5)*100)]} onValueChange={(v)=>onParamsChange({ ...params, thr_override: v[0]/100 })} />
              <Input className='w-20' value={(params.thr_override ?? 0.5).toFixed(2)} onChange={e=>onParamsChange({ ...params, thr_override: parseFloat(e.target.value)||0 })} />
            </div>
          </div>

          <div className='grid grid-cols-3 gap-2 items-center'>
            <Label>Label margin</Label>
            <Input className='col-span-2' value={params.label_margin} onChange={e=>onParamsChange({ ...params, label_margin: parseFloat(e.target.value)||0 })} />
          </div>

          <div className='grid grid-cols-3 gap-2 items-center'>
            <Label>qhat_mult</Label>
            <Input className='col-span-2' value={params.qhat_mult} onChange={e=>onParamsChange({ ...params, qhat_mult: parseFloat(e.target.value)||0 })} />
          </div>

          <div className='grid grid-cols-3 gap-2 items-center'>
            <Label>MutScan</Label>
            <select className='col-span-2 h-10 border rounded-xl px-3' value={params.mutscan} onChange={e=>onParamsChange({ ...params, mutscan: e.target.value as any })}>
              <option value='none'>none</option>
              <option value='fast'>fast</option>
              <option value='full'>full</option>
            </select>
          </div>

          <div className='grid grid-cols-3 gap-2 items-center'>
            <Label>MutScan budget</Label>
            <Input className='col-span-2' value={params.mutscan_budget} onChange={e=>onParamsChange({ ...params, mutscan_budget: parseInt(e.target.value)||0 })} />
          </div>
        </div>

        <div className='space-y-3'>
          {[
            ['use_project_heads','Use project heads_out (else default /heads)'],
            ['no_borderline','Only positive/negative labels'],
            ['force_unreliable_heads','Run even if gated by ECE/n'],
            ['ignore_conformal','Disable conformal abstention'],
            ['abstain_on_guard','Abstain on guard triggers'],
          ].map(([key, hint]) => (
            <div key={key} className='flex items-center justify-between border rounded-lg px-3 py-2'>
              <div><div className='font-medium text-sm'>{key}</div><div className='text-xs text-gray-500'>{hint}</div></div>
              <Switch checked={!!(params as any)[key]} onCheckedChange={(v)=>onParamsChange({ ...params, [key]: v } as PredictParams)} />
            </div>
          ))}
          <div className='pt-2'><Button onClick={onRun} disabled={busy}><Play className='h-4 w-4 mr-1'/>{busy? 'Running...':'Run predict'}</Button></div>
        </div>
      </CardContent>
    </Card>
  )
}
