import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Button } from './components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card'
import { Input } from './components/ui/input'
import { Label } from './components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs'
import { CSVGrid } from './components/CSVGrid'
import { PredictPanel } from './components/PredictPanel'
import { TrainPanel } from './components/TrainPanel'
import { ResultsCard } from './components/ResultsCard'
import { HeadsLibrary } from './components/HeadsLibrary'
import { EvaluatePanel } from './components/EvaluatePanel'
import { Topbar } from './components/Topbar'
import { Download, FileSpreadsheet, FolderPlus, Plus, Save, Trash2, Pencil, RefreshCw } from 'lucide-react'
import { api } from './api'
import { downloadText } from './utils'
import type { PredictParams, Project, Tier, TrainSpec, EvalSpec, FileRef } from './types'
import { JsonEditor } from './components/JsonEditor'

// ΝΕΑ components για Plugins / Simulator / Patients
import { PluginBuilder } from './components/PluginBuilder'
import { SimulatorPanel } from './components/SimulatorPanel'
import { PatientsPanel } from './components/PatientsPanel'

export default function App() {
  const [auth, setAuth] = useState<{ token: string; email: string } | null>(null)
  const [projects, setProjects] = useState<Project[]>([])
  const [current, setCurrent] = useState<Project | null>(null)
  const [activeTab, setActiveTab] = useState<
    'datasets'|'predict'|'train'|'evaluate'|'heads'|'plugins'|'simulate'|'patients'
  >('datasets')

  // Dataset editor / selection
  const [csvBuffer, setCsvBuffer] = useState<string>('sequence\n')
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null)

  // ML params/specs
  const [tier, setTier] = useState<Tier>('premium')
  const [predictParams, setPredictParams] = useState<PredictParams>({
    thr_override: 0.5, decision_policy: 'none', qhat_mult: 1.0, label_margin: 0.15,
    no_borderline: true, force_unreliable_heads: true, ignore_conformal: true, abstain_on_guard: false,
    mutscan: 'none', mutscan_budget: 60, use_project_heads: false,
  })

  const [trainSpec, setTrainSpec] = useState<TrainSpec>({
    type: 'classification', task_name: 'AMP_demo_head', datasetId: '', datasetName: '', seq_col: 'sequence', label_col: 'label',
    feat_mode: 'fuse', calibration: 'sigmoid', cv_folds: 5, precision_target: 0.8,
  })

  const [evalSpec, setEvalSpec] = useState<EvalSpec>({
    predFileId: '', gtFileId: '', prob_col: 'amp_head_prob', label_col: 'amp_head_label',
    decision_col: 'amp_head_decision', seq_col: 'sequence', gt_label_col: 'label'
  })

  const [predictResultCSV, setPredictResultCSV] = useState<string | null>(null)
  const [evalOut, setEvalOut] = useState<string>('')
  const [log, setLog] = useState('')
  const [busy, setBusy] = useState(false)

  // --- auth & bootstrap ---
  useEffect(() => {
    const token = localStorage.getItem('pep_token')
    const email = localStorage.getItem('pep_email')
    if (token && email) setAuth({ token, email })
  }, [])

  useEffect(() => {
    if (!auth) return
    ;(async () => {
      await refreshProjects()
      // pick the first project if none selected
      if (!current && projects.length) setCurrent(projects[0])
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [auth])

  // --- helpers ---
  const refreshProjects = async () => {
    if (!auth) return
    const list = await api.listProjects(auth.token)
    setProjects(list)
    if (current) {
      const cur = list.find(p => p.id === current.id) || null
      setCurrent(cur)
    } else if (list.length) {
      setCurrent(list[0])
    }
  }

  const onLogin = async (email: string, password: string) => {
    const res = await api.login(email, password)
    localStorage.setItem('pep_token', res.token)
    localStorage.setItem('pep_email', res.user.email)
    setAuth({ token: res.token, email: res.user.email })
  }

  const onCreateProject = async () => {
    if (!auth) return
    const name = prompt('New project name', `project_${new Date().toISOString().slice(0,10)}`)?.trim()
    if (!name) return
    const p = await api.createProject(auth.token, name)
    await refreshProjects()
    setCurrent(p)
  }

  const onDeleteProject = async (projectId: string) => {
    if (!auth) return
    if (!confirm('Are you sure you want to delete this project? This action cannot be undone.')) return
    await api.deleteProject(auth.token, projectId)
    if (current?.id === projectId) setCurrent(null)
    await refreshProjects()
  }

  // File ops
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const onUploadClick = () => fileInputRef.current?.click()

  const onUploadCSV = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!auth || !current) return
    const file = e.target.files?.[0]
    if (!file) return
    const text = await file.text()
    await api.uploadCSV(auth.token, current.id, file.name, text)
    await refreshProjects()
    e.target.value = ''
  }

  const onLoadFile = async (fileId: string) => {
    if (!auth || !current) return
    const text = await api.downloadCSV(auth.token, current.id, fileId)
    setSelectedFileId(fileId)
    setCsvBuffer(text)
  }

  const onDeleteFile = async (fileId: string) => {
    if (!auth || !current) return
    if (!confirm('Delete this file?')) return
    await api.deleteFile(auth.token, current.id, fileId)
    await refreshProjects()
    if (selectedFileId === fileId) {
      setSelectedFileId(null)
      setCsvBuffer('sequence\n')
    }
  }

  const onSaveGridAs = async () => {
    if (!auth || !current) return
    const defaultName = selectedFileId ? (current.files.find(f => f.id === selectedFileId)?.name || 'edited.csv') : 'dataset.csv'
    const name = prompt('Save as filename', defaultName)?.trim()
    if (!name) return
    await api.uploadCSV(auth.token, current.id, name, csvBuffer)
    await refreshProjects()
  }

  const onRenameFile = async (fileId: string) => {
    if (!auth || !current) return
    const f = current.files.find(x => x.id === fileId)
    const newName = prompt('Rename file to:', f?.name || '')?.trim()
    if (!newName || newName === f?.name) return
    // typed as any to avoid TS complaining if api.renameFile is not typed
    await (api as any).renameFile?.(auth.token, current.id, fileId, newName)
    await refreshProjects()
  }

  const downloadFile = async (file: FileRef) => {
    if (!auth || !current) return
    const text = await api.downloadCSV(auth.token, current.id, file.id)
    downloadText(file.name, text)
  }

  // Util: get first CSV row as headers (used by future Evaluate dropdowns)
  const getFileColumns = async (fileId: string): Promise<string[]> => {
    if (!auth || !current) return []
    const text = await api.downloadCSV(auth.token, current.id, fileId)
    const firstLine = (text.split(/\r?\n/).find(($1: string) => $1.trim().length > 0) || '')
    const delim = firstLine.includes('\t') ? '\t' : ','
    return firstLine.split(delim).map(($1: string) => $1.trim()).filter(Boolean)
  }

  // --- actions: predict/train/evaluate ---
  const runPredict = async () => {
    if (!auth || !current) return
    setBusy(true)
    try {
      // If a dataset file is selected, use that; otherwise save the Grid buffer first
      let datasetId = selectedFileId

      const nonEmptyRows = csvBuffer.trim().split(/\r?\n/).filter(r => r.trim().length > 0)
      if (!datasetId && nonEmptyRows.length > 1) {
        const tempName = `grid_${Date.now()}.csv`
        await api.uploadCSV(auth.token, current.id, tempName, csvBuffer)
        await refreshProjects()
        // Pick the newest file with that name
        const fresh = (await api.listProjects(auth.token)).find(p => p.id === current.id)
        const candidate = fresh?.files.filter(f => f.name === tempName).sort((a,b)=>+new Date(b.createdAt)-+new Date(a.createdAt))[0]
        if (candidate) datasetId = candidate.id
      }

      if (!datasetId) {
        alert('Select a dataset file or fill the grid and click Predict again.')
        return
      }

      const outCSV = await api.predict(
        auth.token,
        current.id,
        current?.tier || tier,
        predictParams,
        datasetId
      )
      setPredictResultCSV(outCSV)
      await refreshProjects() // ensure new prediction file appears immediately
      setActiveTab('predict')
    } catch (e: any) {
      alert(e.message)
    } finally {
      setBusy(false)
    }
  }

  const runTrain = async () => {
    if (!auth || !current) return
    if (!trainSpec.datasetId && !trainSpec.datasetName) { alert('Select a dataset'); return }
    setBusy(true)
    try {
      const res = await api.train(auth.token, current.id, trainSpec)
      setLog(prev => prev + `\n${new Date().toLocaleTimeString()}: ${res.message || 'ok'}`)
      await refreshProjects()
    } catch (e: any) {
      alert(e.message)
    } finally {
      setBusy(false)
    }
  }

  const runEvaluate = async () => {
    if (!auth || !current) return
    if (!evalSpec.predFileId || !evalSpec.gtFileId) { alert('Select both predictions and GT files'); return }
    setBusy(true)
    try {
      const txt = await api.evaluate(auth.token, current.id, evalSpec)
      setEvalOut(String(txt))
      await refreshProjects()
      setActiveTab('evaluate')
    } catch (e: any) {
      alert(e.message)
    } finally {
      setBusy(false)
    }
  }

  if (!auth) return <Login onLogin={onLogin} />

  const selectedFile = current?.files.find(f => f.id === selectedFileId) || null
const isJSON = (selectedFile?.name?.toLowerCase().endsWith('.json')) || (/^\s*[\{\[]/.test(csvBuffer || ''))

  return (
    <div className='min-h-screen bg-gradient-to-b from-white to-gray-50'>
      <Topbar email={auth.email} onLogout={() => { localStorage.clear(); location.reload() }} />
      <div className='mx-auto max-w-[1400px] px-4 py-6 grid grid-cols-12 gap-4'>

        {/* Left: Projects */}
        <div className='col-span-12 lg:col-span-3'>
          <Card className='sticky top-4'>
            <CardHeader className='pb-2'>
              <CardTitle className='text-lg flex items-center gap-2'><FolderPlus className='h-5 w-5'/> Projects</CardTitle>
            </CardHeader>
            <CardContent className='space-y-3'>
              <div className='flex items-center gap-2'>
                <Button size='sm' onClick={onCreateProject}><Plus className='h-4 w-4 mr-1'/>New</Button>
                <select className='h-9 border rounded-xl px-3' value={current?.tier || tier} onChange={(e)=>{
                  if (!current) return
                  const next = { ...current, tier: e.target.value as Tier }
                  setCurrent(next)
                  api.saveProject(auth.token, next as any)
                }}>
                  <option value='free'>Free</option>
                  <option value='premium'>Premium</option>
                </select>
              </div>

              <div className='space-y-2 max-h-[280px] overflow-auto pr-2'>
                {projects.map(p => (
                  <div key={p.id} className={`border rounded-xl p-2 ${current?.id===p.id?'border-gray-900':'border-gray-200'}`}>
                    <div className='flex items-center justify-between gap-2'>
                      <button className='text-left truncate font-medium' onClick={()=>setCurrent(p)}>{p.name}</button>
                      <Button size='icon' variant='destructive' onClick={()=>onDeleteProject(p.id)}><Trash2 className='h-4 w-4'/></Button>
                    </div>
                    <div className='text-xs text-gray-500 mt-1'>Files: {p.files?.length || 0} · Heads: {p.heads?.length || 0}</div>
                  </div>
                ))}
                {projects.length===0 && <div className='text-sm text-gray-500'>No projects yet.</div>}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right: Workspace */}
        <div className='col-span-12 lg:col-span-9 space-y-4'>
          {/* Dataset editor */}
          <Card>
            <CardHeader className='pb-3'><CardTitle className='text-lg flex items-center gap-2'><FileSpreadsheet className='h-5 w-5'/> Dataset Editor</CardTitle></CardHeader>
            <CardContent className='space-y-3'>
              <div className='flex flex-wrap items-center gap-2'>
                <Button variant='outline' size='sm' onClick={()=>{ setCsvBuffer('sequence\n'); setSelectedFileId(null) }}>New blank</Button>
<Button variant='outline' size='sm' onClick={()=>{ setCsvBuffer('{\n  "name": "",\n  "type": "bacteria"\n}\n'); setSelectedFileId(null) }}>
  New JSON
</Button>

                <Button variant='outline' size='sm' onClick={()=>downloadText(selectedFile?.name || 'dataset.csv', csvBuffer)}><Download className='h-4 w-4 mr-1'/>Export CSV</Button>
                <Button size='sm' onClick={onSaveGridAs}><Save className='h-4 w-4 mr-1'/>Save</Button>

                <div className='ml-auto flex items-center gap-2'>
                  
<input ref={fileInputRef} type='file' accept='.csv,.json,.txt' className='hidden' onChange={onUploadCSV} />

                  <Button variant='outline' size='sm' onClick={onUploadClick}><Plus className='h-4 w-4 mr-1'/>Upload CSV</Button>
                </div>
              </div>
             


{isJSON ? (
  <JsonEditor value={csvBuffer} onChange={setCsvBuffer} />
) : (
  <CSVGrid csv={csvBuffer} onChange={setCsvBuffer} />
)}


              <div className='text-xs text-gray-500'>Tip: Paste from Excel/Sheets directly to the table above.</div>
            </CardContent>
          </Card>

          {/* Run/Train/Evaluate */}
          <Card>
            <CardHeader className='pb-3'><CardTitle className='text-lg'>Run / Train / Evaluate / Simulate</CardTitle></CardHeader>
            <CardContent>
              <Tabs value={activeTab} onValueChange={setActiveTab as any}>






<TabsList className='flex gap-2'>
  {['datasets','predict','train','evaluate','heads','plugins','simulate','patients'].map((t) => {
    // βάλε εδώ ό,τι custom labels θες
    const CUSTOM_LABELS: Record<string, string> = {
      simulate: 'Lab Simulation',
      plugins: 'Create Disease File',
      patients: 'Disease files',
    }
    const defaultLabel = t[0].toUpperCase() + t.slice(1)
    const label = CUSTOM_LABELS[t] ?? defaultLabel

    return (
      <TabsTrigger
        key={t}
        value={t as any}
        active={activeTab === t}
        onClick={() => setActiveTab(t as any)}
      >
        {label}
      </TabsTrigger>
    )
  })}
</TabsList>









                {/* DATASETS TAB */}
                <TabsContent hidden={activeTab!=='datasets'} className='pt-4 space-y-3'>
                  <div className='flex items-center justify-between'>
                    <p className='text-sm text-gray-600'>Load/edit CSV and then click Predict / Train / Evaluate.</p>
                    <Button variant='outline' size='sm' onClick={refreshProjects}><RefreshCw className='h-4 w-4 mr-1'/>Refresh</Button>
                  </div>

                  <div className='overflow-auto border rounded-xl'>
                    <table className='w-full text-sm'>
                      <thead className='bg-white sticky top-0'>
                        <tr>
                          {['Name','Size','Created','Actions'].map(h => (
                            <th key={h} className='text-left px-3 py-2 border-b font-medium'>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {current?.files?.length ? current.files.map(f => (
                          <tr key={f.id} className='hover:bg-gray-50'>
                            <td className='px-3 py-2 border-b font-medium truncate'>{f.name}</td>
                            <td className='px-3 py-2 border-b'>{(f.size/1024).toFixed(1)} KB</td>
                            <td className='px-3 py-2 border-b'>{new Date(f.createdAt).toLocaleString()}</td>
                            <td className='px-3 py-2 border-b'>
                              <div className='flex flex-wrap gap-1'>
                                <Button size='sm' variant='ghost' onClick={()=>onLoadFile(f.id)}>Open</Button>
                                <Button size='icon' variant='ghost' onClick={()=>onRenameFile(f.id)}><Pencil className='h-4 w-4'/></Button>
                                <Button size='icon' variant='ghost' onClick={()=>downloadFile(f)}><Download className='h-4 w-4'/></Button>
                                <Button size='icon' variant='destructive' onClick={()=>onDeleteFile(f.id)}><Trash2 className='h-4 w-4'/></Button>
                              </div>
                            </td>
                          </tr>
                        )) : (
                          <tr><td className='px-3 py-3 text-gray-500' colSpan={4}>No files yet.</td></tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </TabsContent>

                {/* PREDICT TAB */}
                <TabsContent hidden={activeTab!=='predict'} className='pt-4 space-y-4'>
                  <PredictPanel tier={current?.tier || tier} params={predictParams} onParamsChange={setPredictParams} onRun={runPredict} busy={busy}/>
                  {predictResultCSV && <ResultsCard csv={predictResultCSV}/>}                
                </TabsContent>

                {/* TRAIN TAB */}
                <TabsContent hidden={activeTab!=='train'} className='pt-4 space-y-4'>
                  <TrainPanel project={current} spec={trainSpec} onSpecChange={setTrainSpec} onRun={runTrain} busy={busy} />
                  <Card>
                    <CardHeader className='pb-2'><CardTitle className='text-base'>Logs</CardTitle></CardHeader>
                    <CardContent><pre className='text-xs bg-gray-50 rounded-lg p-3 max-h-[220px] overflow-auto whitespace-pre-wrap'>{log || '—'}</pre></CardContent>
                  </Card>
                </TabsContent>

                {/* EVALUATE TAB */}
                <TabsContent hidden={activeTab!=='evaluate'} className='pt-4 space-y-4'>
                  <EvaluatePanel project={current} spec={evalSpec} onSpecChange={setEvalSpec} onRun={runEvaluate} busy={busy} getFileColumns={getFileColumns} />
                  <Card>
                    <CardHeader className='pb-2'><CardTitle className='text-base'>Output</CardTitle></CardHeader>
                    <CardContent><pre className='text-xs bg-gray-50 rounded-lg p-3 max-h-[320px] overflow-auto whitespace-pre-wrap'>{evalOut || '—'}</pre></CardContent>
                  </Card>
                </TabsContent>

                {/* HEADS TAB */}
                <TabsContent hidden={activeTab!=='heads'} className='pt-4'>
                  <HeadsLibrary
                    project={current}
                    onDelete={async (headId) => {
                      if (!auth || !current) return
                      // αν δεν έχεις backend route για delete head, βγάλτο αυτό το κομμάτι
                      try {
                        await api.deleteHead(auth.token, current.id, headId)
                        await refreshProjects()
                      } catch {
                        // optional: alert('Delete head not supported on backend')
                      }
                    }}
                  />
                </TabsContent>

                {/* PLUGINS TAB (builder για JSON plugin) */}
                <TabsContent hidden={activeTab!=='plugins'} className='pt-4'>
<PluginBuilder project={current} onSaved={refreshProjects} />
                </TabsContent>

                {/* SIMULATE TAB (τρέχει labsim_demo.py) */}



<TabsContent hidden={activeTab!=='simulate'} className='pt-4'>
  <SimulatorPanel
    project={current}
    onOpenFile={async (fid: string) => {
      if (!auth || !current) return
      await refreshProjects()      // 1) φέρε φρέσκια λίστα αρχείων
      await onLoadFile(fid)        // 2) φόρτωσε το νέο CSV στον editor
      setActiveTab('datasets')     // 3) γύρνα στο datasets για να το δεις
    }}
  />
</TabsContent>






                {/* PATIENTS TAB (browser/preview/save-as) */}
                <TabsContent hidden={activeTab!=='patients'} className='pt-4'>
                  <PatientsPanel
                    project={current}
onOpenFile={(fid: string)=>onLoadFile(fid)}
                   
                    csvBuffer={csvBuffer}
                    onSaveAs={onSaveGridAs}
                  />
                </TabsContent>

              </Tabs>
            </CardContent>
          </Card>
        </div>

      </div>
    </div>
  )
}

function Login({ onLogin }: { onLogin: (email: string, password: string) => void }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  return (
    <div className='min-h-screen grid place-items-center bg-gray-50'>
      <Card className='w-full max-w-md'>
        <CardHeader><CardTitle className='text-lg'>Sign in</CardTitle></CardHeader>
        <CardContent className='space-y-3'>
          <div>
            <Label>Email</Label>
            <Input type='email' value={email} onChange={e=>setEmail(e.target.value)} placeholder='you@lab.org' />
          </div>
          <div>
            <Label>Password</Label>
            <Input type='password' value={password} onChange={e=>setPassword(e.target.value)} placeholder='••••••••' />
          </div>
          <Button className='w-full' onClick={()=>onLogin(email, password)}>Continue</Button>
        </CardContent>
      </Card>
    </div>
  )
}
