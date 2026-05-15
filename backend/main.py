import os, json, time, uuid, shutil, subprocess, sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request, Body
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from fastapi import UploadFile, File

# --- Runtime paths ---
# Deployment paths can be supplied through environment variables. Local
# development defaults to the repository root, derived from this file.
BASE_DIR = Path(os.environ.get("EVOTENSOR_BASE_DIR", Path(__file__).resolve().parents[1])).resolve()
VOLUMES_DIR = Path(os.environ.get("EVOTENSOR_VOLUMES_DIR", BASE_DIR / "volumes")).resolve()

PROJECTS_ROOT = str(Path(os.environ.get("EVOTENSOR_PROJECTS_ROOT", VOLUMES_DIR / "projects")).resolve())
PEP_PRODUCT_SCRIPT_PATH = str(Path(os.environ.get("PEP_PRODUCT_SCRIPT_PATH", BASE_DIR / "pep_product.py")).resolve())
GROUNDTRUTH_SCRIPT_PATH = str(Path(os.environ.get("GROUNDTRUTH_SCRIPT_PATH", BASE_DIR / "groundtruth.py")).resolve())
DEFAULT_HEADS_PATH = str(Path(os.environ.get("DEFAULT_HEADS_PATH", BASE_DIR / "heads")).resolve())
SPECIALISTS_PATH = str(Path(os.environ.get("SPECIALISTS_PATH", BASE_DIR / "specialists")).resolve())
TIMEOUT_SEC = int(os.environ.get("EVOTENSOR_TIMEOUT_SEC", "7200"))  # 2 hours for safety
LABSIM_SCRIPT_PATH = str(Path(os.environ.get("LABSIM_SCRIPT_PATH", BASE_DIR / "labsim_demo.py")).resolve())
os.makedirs(PROJECTS_ROOT, exist_ok=True)



app = FastAPI(title="Evotensor PRO API - Final No Docker Version")

###-----Firebase-----#####
import os, time
from fastapi import Header, Depends
import firebase_admin
from firebase_admin import credentials, auth, firestore

FIREBASE_CRED = os.environ.get("FIREBASE_CRED", str(BASE_DIR / "secrets" / "firebase.json"))
TOOL_KEY = os.environ.get("TOOL_KEY", "evotensor_pro")
AUTH_MODE = os.environ.get("EVOTENSOR_AUTH_MODE", "firebase").lower()



PROJECTS_BASE = str(Path(os.environ.get("EVOTENSOR_PROJECTS_BASE", VOLUMES_DIR / "projects_by_user")).resolve())
DEMO_PROJECT_ID = "demo_default"
DEMO_PROJECT_NAME = "Evotensor Demo"
DEMO_PROJECT_ENABLED = os.environ.get("EVOTENSOR_SEED_DEMO", "1").lower() not in {"0", "false", "no"}
DEMO_FILES = [
    ("demo/evaluation/amp_predictions.csv", "amp_predictions.csv"),
    ("demo/evaluation/amp_ground_truth.csv", "amp_ground_truth.csv"),
    ("demo/ecoli_uti/for_ecoli_lab_test.csv", "for_ecoli_lab_test.csv"),
    ("demo/ecoli_uti/ecoli_disease_plugin.json", "ecoli_disease_plugin.json"),
    ("demo/ecoli_uti/ecoli_lab_sim.csv", "ecoli_lab_sim.csv"),
]

def projects_root_for(user_uid: str) -> str:
    root = os.path.join(PROJECTS_BASE, user_uid)
    os.makedirs(root, exist_ok=True)
    return root

if AUTH_MODE == "firebase" and not os.path.exists(FIREBASE_CRED):
    raise RuntimeError(
        f"Firebase service account not found at {FIREBASE_CRED}. "
        "Set FIREBASE_CRED or use EVOTENSOR_AUTH_MODE=dev for local API-only testing."
    )

if AUTH_MODE == "firebase" and not firebase_admin._apps:
    firebase_admin.initialize_app(credentials.Certificate(FIREBASE_CRED))

db = firestore.client() if AUTH_MODE == "firebase" else None

def get_bearer_token(authorization: str | None = Header(default=None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    raise HTTPException(status_code=401, detail="Bad Authorization header")



def require_firebase_user(token: str = Depends(get_bearer_token)) -> dict:
    if AUTH_MODE == "dev":
        return {"uid": "local-dev", "email": "local-dev@evotensor.local"}

    try:
        decoded = auth.verify_id_token(token)
        return decoded
    except Exception as e:
        print("VERIFY_FAIL:", repr(e), flush=True)
        raise HTTPException(status_code=401, detail="Invalid token")



def require_tool_access(decoded: dict = Depends(require_firebase_user)) -> dict:
    uid = decoded["uid"]
    if AUTH_MODE == "dev":
        return {
            "uid": uid,
            "email": decoded.get("email", "local-dev@evotensor.local"),
            "profile": {"tools": {TOOL_KEY: True}, "enabled": True},
            "claims": decoded,
        }

    ref = db.collection("users").document(uid).get()
    if not ref.exists:
        # user έκανε signup αλλά δεν έχει Firestore doc
        raise HTTPException(status_code=403, detail="No access (not registered)")
    data = ref.to_dict() or {}
    if not data.get("enabled", True):
        raise HTTPException(status_code=403, detail="Account disabled")
    tools = (data.get("tools") or {})
    if tools.get(TOOL_KEY) is not True:
        raise HTTPException(status_code=403, detail="No access to tool")
    return {"uid": uid, "email": data.get("email") or decoded.get("email", ""), "profile": data, "claims": decoded}




@app.post("/api/request_access")
def request_access(decoded: dict = Depends(require_firebase_user)):
    uid = decoded["uid"]
    email = decoded.get("email","")
    ref = db.collection("users").document(uid)
    if not ref.get().exists:
        ref.set({
            "email": email,
            "enabled": True,
            "tools": {TOOL_KEY: False},
            "tier": "free",
            "status": "pending",
            "createdAt": int(time.time())
        })
    return {"ok": True, "status": "pending"}



###-----Firebase-----#####








# --- Pydantic Models ---
class User(BaseModel): id: str; email: str
class LoginReq(BaseModel): email: str; password: str
class ProjectFile(BaseModel): id: str; name: str; kind: str = "csv"; size: int; createdAt: str
class Head(BaseModel): id: str; task_name: str; createdAt: str; type: str
class Project(BaseModel): id: str; name: str; createdAt: str; tier: str = "free"; files: List[ProjectFile] = []; heads: List[Head] = []


#-simulate
class SimParams(BaseModel):
    peptidesFileId: str
    plugin: str                 # π.χ. 'mrsa_demo' | 'ecoli_uti_demo' | 'file:<id>'
    dose_uM: float = 32.0
    interval_h: float = 12.0
    n_doses: int = 2
    duration_h: float = 24.0
    outName: str = "simulation.csv"

class SimulateReq(BaseModel):
    projectId: str
    params: SimParams


def _write_temp_text(text: str, suffix: str = ".json") -> str:
    os.makedirs("/tmp/evotensor", exist_ok=True)
    p = os.path.join("/tmp/evotensor", f"tmp_{uuid.uuid4().hex}{suffix}")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p







#----simulate over

class TrainSpec(BaseModel):
    type: str = "classification"
    task_name: str
    datasetId: str
    datasetName: Optional[str] = ""
    seq_col: str = "sequence"
    label_col: Optional[str] = "label"
    target_col: Optional[str] = None   # ✅ NEW
    feat_mode: str = "fuse"
    calibration: str = "sigmoid"
    cv_folds: int = 5
    precision_target: float = 0.8



class PredictParams(BaseModel):
    thr_override: Optional[float] = 0.5
    decision_policy: str = "none"
    qhat_mult: float = 1.0
    label_margin: float = 0.15
    no_borderline: bool = False
    force_unreliable_heads: bool = True
    ignore_conformal: bool = True
    abstain_on_guard: bool = False
    mutscan: str = "none"
    mutscan_budget: int = 60
    use_project_heads: bool = False

class PredictReq(BaseModel):
    projectId: str
    datasetFileId: Optional[str] = None  # FRONTEND στέλνει αυτό
    tier: str = "premium"
    params: PredictParams

class EvaluateReq(BaseModel):
    projectId: str
    spec: dict

class CreateProjectReq(BaseModel):
    name: str

class RenameReq(BaseModel):
    projectId: str
    fileId: str
    newName: str


# --- Helper Functions ---
def proj_dir(uid: str, pid: str) -> str:
    return os.path.join(projects_root_for(uid), pid)

def proj_meta_path(uid: str, pid: str) -> str:
    return os.path.join(proj_dir(uid, pid), "project.json")

def run_local_script(args: list, timeout: int, env: dict = None) -> subprocess.CompletedProcess:
    print("RUN (direct):", " ".join(args), flush=True)
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, env=process_env)

def load_project(uid: str, pid: str) -> Project:
    p_path = proj_meta_path(uid, pid)
    if not os.path.exists(p_path): 
        raise HTTPException(404, "Project not found")
    with open(p_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Project(**data)

def save_project(uid: str, project: Project):
    d = proj_dir(uid, project.id)
    os.makedirs(d, exist_ok=True)
    # heads φορτώνονται δυναμικά, δεν τα γράφουμε στο JSON
    project_dict = project.model_dump(exclude={'heads'}) if hasattr(project, 'model_dump') else project.dict(exclude={'heads'})
    with open(proj_meta_path(uid, project.id), "w", encoding="utf-8") as f:
        json.dump(project_dict, f, ensure_ascii=False, indent=2)

def ensure_demo_project(uid: str) -> None:
    if not DEMO_PROJECT_ENABLED:
        return

    base = proj_dir(uid, DEMO_PROJECT_ID)
    files_dir = os.path.join(base, "files")
    runs_dir = os.path.join(base, "runs")
    heads_dir = os.path.join(base, "heads_out")
    os.makedirs(files_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(heads_dir, exist_ok=True)

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    existing = None
    if os.path.exists(proj_meta_path(uid, DEMO_PROJECT_ID)):
        try:
            existing = load_project(uid, DEMO_PROJECT_ID)
        except Exception:
            existing = None

    known_by_name = {f.name: f for f in existing.files} if existing else {}
    files: List[ProjectFile] = list(existing.files) if existing else []

    for rel_path, display_name in DEMO_FILES:
        src = BASE_DIR / rel_path
        if not src.is_file() or display_name in known_by_name:
            continue
        fid = f"demo_{Path(display_name).stem.replace('-', '_')}"
        dst = file_realpath(uid, DEMO_PROJECT_ID, fid)
        shutil.copyfile(src, dst)
        files.append(ProjectFile(
            id=fid,
            name=display_name,
            size=os.path.getsize(dst),
            createdAt=now,
        ))

    project = existing or Project(id=DEMO_PROJECT_ID, name=DEMO_PROJECT_NAME, createdAt=now, tier="premium")
    project.name = DEMO_PROJECT_NAME
    project.tier = "premium"
    project.files = files
    save_project(uid, project)

def list_projects(uid: str) -> List[Project]:
    root = projects_root_for(uid)
    ensure_demo_project(uid)
    out = []
    if not os.path.isdir(root): return []
    
    for name in sorted(os.listdir(root)):
        pd = os.path.join(root, name)
        if not os.path.isdir(pd): continue
        meta_path = os.path.join(pd, "project.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    p = Project(**json.load(f))
                    # --- ALWAYS READ HEADS FROM DISK ---
                    p.heads = []
                    heads_dir = os.path.join(pd, "heads_out")
                    if os.path.isdir(heads_dir):
                        for fname in os.listdir(heads_dir):
                            if fname.endswith(".joblib") or fname.endswith(".pkl"):
                                stat = os.stat(os.path.join(heads_dir, fname))
                                task_name = fname.replace(".joblib", "").replace(".pkl", "")
                                p.heads.append(Head(
                                    id=fname,
                                    task_name=task_name,
                                    createdAt=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(stat.st_mtime))),
                                    type="classification"
                                ))
                    out.append(p)
            except Exception as e:
                print(f"Error loading project {name}: {e}")
                continue
    return out

def find_file_meta(p: Project, file_id: str):
    for f in p.files:
        if f.id == file_id: return f
    raise HTTPException(404, "File not found")

def file_realpath(uid: str, pid: str, file_id: str) -> str:
    return os.path.join(proj_dir(uid, pid), "files", f"{file_id}.csv")


# --- API Endpoints ---

@app.post("/api/login")
def login(req: LoginReq):
    return {"token": f"t_{uuid.uuid4().hex}", "user": User(id="u1", email=req.email).dict()}

@app.get("/api/projects")
def api_list_projects(user_data=Depends(require_tool_access)):
    uid = user_data["uid"] # ΠΑΙΡΝΟΥΜΕ ΤΟ UID
    def norm(p: Project): return p.model_dump() if hasattr(p, 'model_dump') else p.dict()
    # ΠΕΡΝΑΜΕ ΤΟ UID ΣΤΟ LIST_PROJECTS
    return [norm(p) for p in list_projects(uid)]

@app.post("/api/projects")
async def api_create_project(request: Request, user_data=Depends(require_tool_access), name: str | None = Form(None), payload: CreateProjectReq | None = None):
    uid = user_data["uid"] # ΠΑΙΡΝΟΥΜΕ ΤΟ UID
    
    if payload and payload.name:
        name = payload.name
    if not name:
        try:
            data = await request.json()
            name = data.get("name")
        except Exception:
            pass
    if not name: raise HTTPException(422, "Field 'name' required")

    pid = f"p_{int(time.time())}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    p = Project(id=pid, name=name, createdAt=now)
    
    # ΧΡΗΣΗ ΤΟΥ path ΜΕ ΒΑΣΗ ΤΟ UID
    base = proj_dir(uid, pid)
    for d in ["files", "runs", "heads_out"]:
        os.makedirs(os.path.join(base, d), exist_ok=True)
    
    save_project(uid, p) # SAVE ΜΕ UID
    return p.model_dump() if hasattr(p, 'model_dump') else p.dict()

@app.put("/api/projects/{pid}")
def api_update_project(pid: str, project: Project, user_data=Depends(require_tool_access)):
    uid = user_data["uid"]
    if pid != project.id: raise HTTPException(400, "pid mismatch")
    load_project(uid, pid) # ΕΛΕΓΧΟΣ ΥΠΑΡΞΗΣ
    save_project(uid, project)
    return project.model_dump() if hasattr(project, 'model_dump') else project.dict()

@app.delete("/api/projects/{projectId}")
def api_delete_project(projectId: str, user_data=Depends(require_tool_access)):
    uid = user_data["uid"]
    p_dir = proj_dir(uid, projectId)
    if not os.path.isdir(p_dir): raise HTTPException(404, "Project not found")
    try:
        shutil.rmtree(p_dir)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, f"Failed to delete project: {e}")

@app.delete("/api/files/{fileId}")
def api_delete_file(fileId: str, projectId: str = Query(...), user_data=Depends(require_tool_access)):
    uid = user_data["uid"]
    p = load_project(uid, projectId)
    p.files = [f for f in p.files if f.id != fileId]
    save_project(uid, p)
    path = file_realpath(uid, projectId, fileId)
    if os.path.exists(path):
        os.remove(path)
    return {"ok": True}

@app.post("/api/files/rename")
def api_files_rename(payload: RenameReq, user_data=Depends(require_tool_access)):
    uid = user_data["uid"]
    p = load_project(uid, payload.projectId)
    new_name = (payload.newName or "").strip()
    if not new_name:
        raise HTTPException(400, "Empty name")
    found = False
    for f in p.files:
        if f.id == payload.fileId:
            if "." not in new_name:
                _, ext = os.path.splitext(f.name)
                if ext:
                    new_name += ext
            f.name = new_name
            found = True
            break
    if not found:
        raise HTTPException(404, "File not found")
    save_project(uid, p)
    return {"ok": True}

@app.post("/api/upload")
async def api_upload(projectId: str = Form(...), file: UploadFile = File(...), user_data=Depends(require_tool_access)):
    uid = user_data["uid"]
    p = load_project(uid, projectId)
    data = await file.read()
    fid = f"f_{uuid.uuid4().hex}"
    
    # ΓΡΑΦΟΥΜΕ ΣΤΟ Path ΤΟΥ ΧΡΗΣΤΗ
    with open(file_realpath(uid, projectId, fid), "wb") as f:
        f.write(data)
        
    meta = ProjectFile(
        id=fid, name=file.filename, size=len(data),
        createdAt=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    p.files.append(meta); save_project(uid, p)
    return meta.dict()

@app.post("/api/files/upload")
def api_files_upload(payload: dict = Body(...), user_data=Depends(require_tool_access)):
    uid = user_data["uid"]
    projectId = payload.get("projectId")
    name = payload.get("name") or f"dataset_{int(time.time())}.csv"
    text = payload.get("text") or ""
    
    p = load_project(uid, projectId)
    fid = f"f_{uuid.uuid4().hex}"
    
    with open(file_realpath(uid, projectId, fid), "w", encoding="utf-8") as f:
        f.write(text)
        
    meta = ProjectFile(
        id=fid, name=name, size=len(text.encode("utf-8")),
        createdAt=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    p.files.append(meta); save_project(uid, p)
    return {"ok": True, "file": meta.dict()}

@app.get("/api/download", response_class=PlainTextResponse)
def api_download(projectId: str = Query(...), fileId: str = Query(...), user_data=Depends(require_tool_access)):
    uid = user_data["uid"]
    p = load_project(uid, projectId)
    find_file_meta(p, fileId)
    path = file_realpath(uid, projectId, fileId)
    if not os.path.exists(path):
        raise HTTPException(404, "file not on disk")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/files/download", response_class=PlainTextResponse)
def api_files_download(projectId: str = Query(...), fileId: str = Query(...), user_data=Depends(require_tool_access)):
    return api_download(projectId=projectId, fileId=fileId, user_data=user_data)

@app.delete("/api/heads/{headId}")
def api_delete_head(headId: str, projectId: str = Query(...), user_data=Depends(require_tool_access)):
    uid = user_data["uid"]
    p = load_project(uid, projectId)
    hdir = os.path.join(proj_dir(uid, p.id), "heads_out")
    path = os.path.join(hdir, headId)
    if not os.path.isfile(path):
        base = os.path.join(hdir, headId)
        tried = False
        for ext in (".joblib", ".pkl"):
            alt = base + ext
            if os.path.isfile(alt):
                path = alt
                tried = True
                break
        if not tried and not os.path.isfile(path):
            raise HTTPException(404, "Head not found")
    try:
        os.remove(path)
    except Exception as e:
        raise HTTPException(500, f"Failed to delete head: {e}")
    return {"ok": True}

@app.post("/api/predict", response_class=PlainTextResponse)
def api_predict(req: PredictReq, user_data=Depends(require_tool_access)):
    uid = user_data["uid"]
    p = load_project(uid, req.projectId)
    
    meta = find_file_meta(p, req.datasetFileId) if req.datasetFileId else sorted(p.files, key=lambda x: x.createdAt)[-1]
    in_csv_path = file_realpath(uid, p.id, meta.id)
    
    run_id = f"r_{int(time.time())}"
    out_csv_path = os.path.join(proj_dir(uid, p.id), "runs", f"{run_id}.csv")

    # Χρήση heads από το dir του χρήστη αν ζητηθεί
    heads_dir = os.path.join(proj_dir(uid, p.id), "heads_out") if req.params.use_project_heads else DEFAULT_HEADS_PATH

    args = [
        sys.executable, PEP_PRODUCT_SCRIPT_PATH, "predict",
        "--tier", req.tier,
        "--input_csv", in_csv_path,
        "--seq_col", "sequence",
        "--out_csv", out_csv_path,
        "--heads_dir", heads_dir
    ]
    args += ["--thr_override", str(req.params.thr_override if req.params.thr_override is not None else 0.5),
             "--decision_policy", req.params.decision_policy]
    args += ["--qhat_mult", str(req.params.qhat_mult),
             "--label_margin", str(req.params.label_margin),
             "--mutscan", req.params.mutscan,
             "--mutscan_budget", str(req.params.mutscan_budget)]
    if req.params.force_unreliable_heads: args.append("--force_unreliable_heads")
    if req.params.ignore_conformal: args.append("--ignore_conformal")
    if req.params.no_borderline: args.append("--no_borderline")
    if req.params.abstain_on_guard: args.append("--abstain_on_guard")

    specialist_env = {
        "SPECIALISTS_MANIFEST": os.path.join(SPECIALISTS_PATH, "specialists_manifest.json"),
        "SPECIALIST_PKLS_GLOB": os.path.join(SPECIALISTS_PATH, "specialist_*.pkl")
    }
    proc = run_local_script(args, timeout=TIMEOUT_SEC, env=specialist_env)
    if proc.returncode != 0:
        error_details = (proc.stdout or "") + (proc.stderr or "")
        raise HTTPException(500, f"Predict failed: {error_details}")

    try:
        with open(out_csv_path, "r", encoding="utf-8") as f:
            csv_text = f.read()
        fid = f"f_{uuid.uuid4().hex}"
        pred_file_meta = ProjectFile(
            id=fid, name=f"preds_{run_id}.csv",
            size=os.path.getsize(out_csv_path),
            createdAt=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        shutil.copy(out_csv_path, file_realpath(uid, p.id, fid))
        p.files.append(pred_file_meta); save_project(uid, p)
        return csv_text
    except FileNotFoundError:
        raise HTTPException(500, "Prediction script ran but did not produce an output file.")

@app.post("/api/train")
def api_train(payload: dict, user_data=Depends(require_tool_access)):
    uid = user_data["uid"]
    projectId = payload.get("projectId")
    spec = TrainSpec(**payload.get("spec", {}))
    p = load_project(uid, projectId)

    meta = find_file_meta(p, spec.datasetId)
    in_csv_path = file_realpath(uid, p.id, meta.id)
    heads_out_dir = os.path.join(proj_dir(uid, p.id), "heads_out")

    is_reg = (spec.type or "").lower() == "regression"
    subcmd = "train_reg" if is_reg else "train"

    args = [sys.executable, PEP_PRODUCT_SCRIPT_PATH, subcmd, "--task_name", spec.task_name]
    args += ["--dataset_csv", in_csv_path, "--seq_col", spec.seq_col]
    args += ["--feat_mode", spec.feat_mode, "--heads_dir", heads_out_dir]
    args += ["--cv_folds", str(spec.cv_folds)]

    if is_reg:
        if not spec.target_col:
            raise HTTPException(400, "target_col is required for regression")
        args += ["--target_col", spec.target_col]
    else:
        if not spec.label_col:
            raise HTTPException(400, "label_col is required for classification")
        args += ["--label_col", spec.label_col]
        args += ["--calibration", spec.calibration]
        args += ["--precision_target", str(spec.precision_target)]

    specialist_env = {
        "SPECIALISTS_MANIFEST": os.path.join(SPECIALISTS_PATH, "specialists_manifest.json"),
        "SPECIALIST_PKLS_GLOB": os.path.join(SPECIALISTS_PATH, "specialist_*.pkl")
    }
    proc = run_local_script(args, timeout=TIMEOUT_SEC, env=specialist_env)
    if proc.returncode != 0:
        error_details = (proc.stdout or "") + (proc.stderr or "")
        raise HTTPException(500, f"Train failed: {error_details}")

    return {"ok": True, "message": (proc.stdout or "")[-2000:]}

@app.post("/api/evaluate", response_class=PlainTextResponse)
def api_evaluate(payload: EvaluateReq, user_data=Depends(require_tool_access)):
    uid = user_data["uid"]
    p = load_project(uid, payload.projectId); spec = payload.spec
    pred_path = file_realpath(uid, p.id, spec["predFileId"])
    gt_path = file_realpath(uid, p.id, spec["gtFileId"])
    
    args = [sys.executable, GROUNDTRUTH_SCRIPT_PATH,
            "--pred_csv", pred_path, "--gt_csv", gt_path,
            "--prob_col", spec.get("prob_col", "amp_head_prob"),
            "--label_col", spec.get("label_col", "amp_head_label"),
            "--decision_col", spec.get("decision_col", "amp_head_decision"),
            "--seq_col", spec.get("seq_col", "sequence"),
            "--gt_label_col", spec.get("gt_label_col", "label")]
            
    proc = run_local_script(args, timeout=TIMEOUT_SEC)
    if proc.returncode != 0:
        error_details = (proc.stdout or "") + (proc.stderr or "")
        raise HTTPException(500, f"Evaluate failed: {error_details}")
    return proc.stdout or "OK"

@app.post("/api/simulate")
def api_simulate(req: SimulateReq, user_data=Depends(require_tool_access)):
    uid = user_data["uid"]
    p = load_project(uid, req.projectId)
    par = req.params

    in_path = file_realpath(uid, p.id, par.peptidesFileId)
    if not os.path.exists(in_path):
        raise HTTPException(404, "Input CSV not found")

    plugin_arg = par.plugin
    if plugin_arg.startswith("file:"):
        plug_id = plugin_arg.split(":", 1)[1]
        plug_path = file_realpath(uid, p.id, plug_id)
        if not os.path.exists(plug_path):
            raise HTTPException(404, "Plugin file not found")
        with open(plug_path, "r", encoding="utf-8") as f:
            plug_text = f.read()
        plugin_arg = _write_temp_text(plug_text, ".json")

    out_fid = f"f_{uuid.uuid4().hex}"
    out_disk_path = file_realpath(uid, p.id, out_fid)
    os.makedirs(os.path.dirname(out_disk_path), exist_ok=True)

    args = [
        sys.executable, LABSIM_SCRIPT_PATH,
        "--peptides_csv", in_path,
        "--plugin", plugin_arg,
        "--dose_uM", str(par.dose_uM),
        "--interval_h", str(par.interval_h),
        "--n_doses", str(par.n_doses),
        "--duration_h", str(par.duration_h),
        "--out_csv", out_disk_path,
    ]

    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "Simulation timed out")
    if proc.returncode != 0:
        raise HTTPException(500, f"Simulation failed:\\nSTDOUT:\\n{proc.stdout}\\nSTDERR:\\n{proc.stderr}")

    size = os.path.getsize(out_disk_path) if os.path.exists(out_disk_path) else 0
    meta = ProjectFile(
        id=out_fid,
        name=(par.outName or "simulation.csv"),
        size=size,
        createdAt=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    p.files.append(meta)
    save_project(uid, p)
    return {"message": "OK", "fileId": out_fid}
