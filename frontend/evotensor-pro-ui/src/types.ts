export type Tier = 'free' | 'premium'
export type FileRef = { id: string; name: string; kind: 'csv'; size: number; createdAt: string; text?: string }
export type Head = { id: string; task_name: string; createdAt: string; type: 'classification' | 'regression' }
export type Project = { id: string; name: string; createdAt: string; files: FileRef[]; tier: Tier; heads?: Head[] }
export type PredictParams = {
  thr_override?: number | null
  decision_policy: 'none' | 'conformal' | 'margin_only'
  qhat_mult: number
  label_margin: number
  no_borderline: boolean
  force_unreliable_heads: boolean
  ignore_conformal: boolean
  abstain_on_guard: boolean
  mutscan: 'none' | 'fast' | 'full'
  mutscan_budget: number
  use_project_heads: boolean
}
export type TrainSpec = {
  type: 'classification' | 'regression'
  task_name: string
  datasetId: string
  datasetName: string
  seq_col: string
  label_col?: string
  target_col?: string
  feat_mode: 'embed' | 'scalars' | 'fuse'
  calibration?: 'none' | 'platt' | 'sigmoid' | 'isotonic'
  cv_folds?: number
  precision_target?: number
}
export type EvalSpec = {
  predFileId: string
  gtFileId: string
  prob_col: string
  label_col: string
  decision_col: string
  seq_col: string
  gt_label_col: string
}
