// src/api.ts
import type { PredictParams, Project, TrainSpec, EvalSpec } from './types'


import { auth } from './firebase'
import { signInWithEmailAndPassword } from 'firebase/auth'

const AUTH_MODE = import.meta.env.VITE_AUTH_MODE || 'firebase'


async function getAuthHeader(): Promise<Record<string, string>> {
  if (AUTH_MODE === 'dev') return { Authorization: 'Bearer dev-token' }

  const user = auth.currentUser
  if (!user) throw new Error('Not logged in')
  const idToken = await user.getIdToken(true) // force refresh
  return { Authorization: `Bearer ${idToken}` }
}


async function fetchJSON<T>(url: string, opts: RequestInit = {}): Promise<T> {
  const res = await fetch(url, opts)
  if (!res.ok) throw new Error(await res.text())
  return res.json() as Promise<T>
}

async function fetchText(url: string, opts: RequestInit = {}): Promise<string> {
  const res = await fetch(url, opts)
  if (!res.ok) throw new Error(await res.text())
  return res.text()
}

export const api = {
  // --- Auth ---
async login(email: string, password: string): Promise<{ token: string; user: { id: string; email: string } }> {
  if (AUTH_MODE === 'dev') {
    return { token: 'dev-token', user: { id: 'local-dev', email: email || 'local-dev@evotensor.local' } }
  }

  const cred = await signInWithEmailAndPassword(auth, email, password)
  const token = await cred.user.getIdToken(true)
  return { token, user: { id: cred.user.uid, email: cred.user.email || email } }
},

  // --- Projects ---
async listProjects(_: string): Promise<Project[]> {
  return fetchJSON<Project[]>('/api/projects', {
    headers: { ...(await getAuthHeader()) },
  })
},

async createProject(_: string, name: string): Promise<Project> {
  return fetchJSON<Project>('/api/projects', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(await getAuthHeader()) },
    body: JSON.stringify({ name }),
  })
},


  async saveProject(token: string, project: Project): Promise<Project> {
    return fetchJSON<Project>(`/api/projects/${project.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify(project),
    })
  },

  async deleteProject(token: string, projectId: string): Promise<{ ok: boolean }> {
    return fetchJSON<{ ok: boolean }>(`/api/projects/${projectId}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` },
    })
  },

  // --- Files ---
  async uploadCSV(token: string, projectId: string, name: string, text: string): Promise<{ id: string; name: string; size: number; createdAt: string }> {
    const res = await fetch('/api/files/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ projectId, name, text }),
    })
    if (!res.ok) throw new Error(await res.text())
    const data = await res.json()
    return data.file || data
  },

async downloadCSV(_: string, projectId: string, fileId: string): Promise<string> {
  const url = `/api/download?projectId=${encodeURIComponent(projectId)}&fileId=${encodeURIComponent(fileId)}`
  return fetchText(url, { headers: { ...(await getAuthHeader()) } })
},


  async deleteFile(token: string, projectId: string, fileId: string): Promise<{ ok: boolean }> {
    return fetchJSON<{ ok: boolean }>(`/api/files/${fileId}?projectId=${encodeURIComponent(projectId)}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` },
    })
  },

  async renameFile(token: string, projectId: string, fileId: string, newName: string): Promise<{ ok: boolean }> {
    return fetchJSON<{ ok: boolean }>('/api/files/rename', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ projectId, fileId, newName }),
    })
  },

  // --- Predict / Train / Evaluate / Heads ---
  async predict(
    token: string,
    projectId: string,
    tier: 'free' | 'premium',
    params: PredictParams,
    datasetFileId: string
  ): Promise<string> {
    return fetchText('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ projectId, datasetFileId, tier, params }),
    })
  },

  async train(token: string, projectId: string, spec: TrainSpec): Promise<{ ok: boolean; message: string }> {
    return fetchJSON<{ ok: boolean; message: string }>('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ projectId, spec }),
    })
  },

  async evaluate(token: string, projectId: string, spec: EvalSpec): Promise<string> {
    return fetchText('/api/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ projectId, spec }),
    })
  },

  async deleteHead(token: string, projectId: string, headId: string): Promise<{ ok: boolean }> {
    return fetchJSON<{ ok: boolean }>(`/api/heads/${encodeURIComponent(headId)}?projectId=${encodeURIComponent(projectId)}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` },
    })
  },

  // --- Simulate (labsim_demo.py) ---
  async simulate(
    token: string,
    payload: { projectId: string; params: {
      peptidesFileId: string
      plugin: string // 'mrsa_demo' | 'ecoli_uti_demo' | 'cancer_invitro_demo' | `file:<id>`
      dose_uM: number
      interval_h: number
      n_doses: number
      duration_h: number
      outName: string
    } }
  ): Promise<{ message: string; fileId?: string }> {
    return fetchJSON<{ message: string; fileId?: string }>('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify(payload),
    })
  },
}
