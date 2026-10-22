"""MCGC v2.1 — strict OOF label-audit comparator confirmation."""
from __future__ import annotations
import argparse, hashlib, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from mcgc import MultiAgentChemotacticGraphClassifier
warnings.filterwarnings("ignore")
DATASETS=("Hill_Valley_without_noise","bupa","ecoli","mfeat_fourier","mfeat_karhunen","mfeat_zernike","saheart","schizo","spectf","vehicle","wine_quality_red","yeast")
CONDITIONS=(("symmetric",0.10),("symmetric",0.20),("symmetric",0.30),("pairflip",0.20))
SEEDS=tuple(range(5)); BUDGETS=(0.05,0.10); AUDIT_FOLDS=5; AUDIT_ALPHA=0.85; AUDIT_K=12; MIN_CLASS=6
METHODS=("mcgc_oof","cl_rf","cl_lr","knn_oof","label_spreading_oof")
def load(data_dir,name):
 p=data_dir/f"{name}.tsv.gz"; df=pd.read_csv(p,sep="\t"); y_raw=df.pop("target").to_numpy(); X=df.to_numpy(dtype=float); vals,counts=np.unique(y_raw,return_counts=True); keep=np.isin(y_raw,vals[counts>=MIN_CLASS]); kept=np.unique(y_raw[keep]); y=np.searchsorted(kept,y_raw[keep]); sha=hashlib.sha256(p.read_bytes()).hexdigest(); return X[keep],y,{"sha256":sha,"sha256_16":sha[:16],"dropped_small_class_rows":int((~keep).sum())}
def corrupt(y,K,kind,eta,seed):
 rng=np.random.default_rng(21000+100*seed+int(round(100*eta))+(50 if kind=="pairflip" else 0)); flip=rng.random(len(y))<eta; yn=y.copy()
 for i in np.flatnonzero(flip): yn[i]=(y[i]+1)%K if kind=="pairflip" else rng.choice([c for c in range(K) if c!=y[i]])
 return yn,flip
def make_cv(seed): return StratifiedKFold(AUDIT_FOLDS,shuffle=True,random_state=22000+seed)
def oof_proba(est,X,yn,K,seed):
 P=np.asarray(cross_val_predict(est,X,yn,cv=make_cv(seed),method="predict_proba"),dtype=float)
 if P.shape[1]!=K:
  present=np.unique(yn); full=np.zeros((len(yn),K)); full[:,present]=P; P=full
 return P
def mcgc_oof_audit(X,yn,K,seed):
 scores=np.empty(len(yn)); suggestions=np.empty(len(yn),dtype=int); seen=np.zeros(len(yn),dtype=bool)
 for fit_idx,val_idx in make_cv(seed).split(X,yn):
  model=MultiAgentChemotacticGraphClassifier(n_neighbors=AUDIT_K,diffusion_alpha=AUDIT_ALPHA,diffusion_steps=40,use_walkers=False,random_state=17).fit(X[fit_idx],yn[fit_idx]); _,diags=model.predict_with_diagnostics(X[val_idx])
  for gi,row in zip(val_idx,diags):
   obs=int(yn[gi]); ev={int(k):float(v) for k,v in row["local_evidence"].items()}; oe=ev.get(obs,0.0); rivals=[(k,v) for k,v in ev.items() if k!=obs]; rl,re=max(rivals,key=lambda z:z[1]) if rivals else (obs,oe); scores[gi]=re-oe; suggestions[gi]=rl; seen[gi]=True
 if not np.all(seen): raise RuntimeError("MCGC OOF audit did not cover all samples")
 return scores,suggestions
def best_alt(P,yn):
 S=np.asarray(P).copy(); S[np.arange(len(yn)),yn]=-np.inf; return S.argmax(axis=1)
def score_prob(P,yn): return 1.0-P[np.arange(len(yn)),yn]
def audit_metrics(flip,scores):
 order=np.argsort(-np.asarray(scores),kind="stable"); total=int(flip.sum()); frac=total/len(flip); review={}
 for b in BUDGETS:
  k=max(1,int(round(b*len(flip)))); top=order[:k]; found=int(flip[top].sum()); p=found/k; review[str(b)]={"reviewed_count":k,"corrupt_found":found,"precision":p,"recall":found/total if total else None,"enrichment_over_random":p/frac if frac else None}
 return {"roc_auc":float(roc_auc_score(flip.astype(int),scores)),"pr_auc":float(average_precision_score(flip.astype(int),scores)),"review":review}
def run_unit(X,y,meta,name,kind,eta,seed):
 K=int(y.max())+1; Xtr,_,ytr,_=train_test_split(X,y,test_size=.30,random_state=20000+seed,stratify=y); yn,flip=corrupt(ytr,K,kind,eta,seed); rec={"dataset":name,"noise":kind,"eta":eta,"seed":seed,"K":K,"n_train":len(yn),"n_corrupt":int(flip.sum()),"meta":meta}
 if flip.sum()==0 or flip.all(): rec["status"]="degenerate_noise"; return rec
 t=time.time(); scaler=StandardScaler().fit(Xtr); Xs=scaler.transform(Xtr); s_mcgc,sug_mcgc=mcgc_oof_audit(Xtr,yn,K,seed)
 P_rf=oof_proba(RandomForestClassifier(n_estimators=200,random_state=97,n_jobs=2),Xtr,yn,K,seed); P_lr=oof_proba(LogisticRegression(C=1.0,max_iter=3000),Xs,yn,K,seed); P_knn=oof_proba(KNeighborsClassifier(AUDIT_K,weights="distance"),Xs,yn,K,seed); P_ls=oof_proba(LabelSpreading(kernel="knn",n_neighbors=AUDIT_K,alpha=AUDIT_ALPHA,max_iter=1000),Xs,yn,K,seed)
 scores={"mcgc_oof":s_mcgc,"cl_rf":score_prob(P_rf,yn),"cl_lr":score_prob(P_lr,yn),"knn_oof":score_prob(P_knn,yn),"label_spreading_oof":score_prob(P_ls,yn)}; sugg={"mcgc_oof":sug_mcgc,"cl_rf":best_alt(P_rf,yn),"cl_lr":best_alt(P_lr,yn),"knn_oof":best_alt(P_knn,yn),"label_spreading_oof":best_alt(P_ls,yn)}
 rec["audit"]={m:audit_metrics(flip,scores[m]) for m in METHODS}; rec["suggestion_acc_on_corrupt"]={m:float(np.mean(np.asarray(sugg[m])[flip]==ytr[flip])) for m in METHODS}; rec["status"]="ok"; rec["seconds"]=round(time.time()-t,1); return rec
def dataset_level_analysis(records,datasets):
 from math import comb
 ok=[r for r in records if r.get("status")=="ok"]; rows=[]
 for ds in datasets:
  g=[r for r in ok if r["dataset"]==ds]
  if not g: continue
  row={"dataset":ds,"n_units":len(g),"methods":{}}
  for m in METHODS:
   row["methods"][m]={"roc_auc_mean":float(np.mean([r["audit"][m]["roc_auc"] for r in g])),"pr_auc_mean":float(np.mean([r["audit"][m]["pr_auc"] for r in g])),"p5_mean":float(np.mean([r["audit"][m]["review"]["0.05"]["precision"] for r in g])),"p10_mean":float(np.mean([r["audit"][m]["review"]["0.1"]["precision"] for r in g]))}
  rows.append(row)
 def signp(w,l):
  n=w+l
  if not n:return 1.0
  k=min(w,l); return min(1.0,2*sum(comb(n,i) for i in range(k+1))/(2**n))
 comps={}
 for metric in ("roc_auc_mean","pr_auc_mean","p5_mean","p10_mean"):
  comps[metric]={}
  for b in METHODS[1:]:
   diffs=[r["methods"]["mcgc_oof"][metric]-r["methods"][b][metric] for r in rows]; w=sum(d>1e-12 for d in diffs); l=sum(d<-1e-12 for d in diffs); t=len(diffs)-w-l; comps[metric][b]={"mean_delta_mcgc_minus_baseline":float(np.mean(diffs)),"median_delta_mcgc_minus_baseline":float(np.median(diffs)),"dataset_wins":w,"dataset_ties":t,"dataset_losses":l,"sign_test_p_two_sided":float(signp(w,l))}
 return {"dataset_level":rows,"comparisons":comps}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--analysis-out",type=Path,default=None); ap.add_argument("--only",default=None); ap.add_argument("--seeds",default=None); a=ap.parse_args(); names=a.only.split(",") if a.only else list(DATASETS); seeds=tuple(int(s) for s in a.seeds.split(",")) if a.seeds else SEEDS; done=set(); existing=[]
 if a.out.exists():
  for line in a.out.read_text(encoding="utf-8").splitlines():
   if line.strip():
    r=json.loads(line); existing.append(r); done.add((r["dataset"],r["noise"],r["eta"],r["seed"]))
 print("=== MCGC v2.1 STRICT-OOF LABEL-AUDIT CONFIRMATION ===",flush=True)
 for name in names:
  X,y,meta=load(a.data_dir,name)
  for seed in seeds:
   for kind,eta in CONDITIONS:
    key=(name,kind,eta,seed)
    if key in done: continue
    rec=run_unit(X,y,meta,name,kind,eta,seed); a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open("a",encoding="utf-8") as f:f.write(json.dumps(rec)+"\n")
    if rec.get("status")=="ok": print(f"{name} {kind} eta={eta:.2f} seed={seed} "+" ".join(f"{m}:ROC={rec['audit'][m]['roc_auc']:.3f}" for m in METHODS),flush=True)
  print(f"DONE {name}",flush=True)
 allr=[json.loads(x) for x in a.out.read_text(encoding="utf-8").splitlines() if x.strip()]; expected={(n,k,e,s) for n in names for s in seeds for k,e in CONDITIONS}; have={(r["dataset"],r["noise"],r["eta"],r["seed"]) for r in allr}
 if expected.issubset(have):
  subset=[r for r in allr if r["dataset"] in names and r["seed"] in seeds and (r["noise"],r["eta"]) in CONDITIONS]; analysis={"protocol":{"version":"2.1","datasets":names,"conditions":[{"noise":n,"eta":e} for n,e in CONDITIONS],"seeds":list(seeds),"methods":list(METHODS),"audit_folds":AUDIT_FOLDS,"budgets":list(BUDGETS),"primary_inference_unit":"dataset","important_change_from_v2.0":"MCGC and Label Spreading are strictly out-of-fold; no cleaning heuristic."},**dataset_level_analysis(subset,names)}; ao=a.analysis_out or a.out.with_name(a.out.stem+"_analysis.json"); ao.write_text(json.dumps(analysis,indent=2,sort_keys=True),encoding="utf-8"); print(f"WROTE {ao}")
 print(f"WROTE {a.out}")
if __name__=="__main__": main()
