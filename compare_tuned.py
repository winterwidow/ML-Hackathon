"""Compare untuned C=1 with Bayesian-selected C on identical grouped folds."""
import csv,pathlib,time,numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,log_loss
from train_models import OUT,load,downsample,SEED

X,y,g,obs,_=load(); folds=GroupKFold(n_splits=3); rows=[]
for fold,(tr,va) in enumerate(folds.split(X,y,g),1):
 Xtr,ytr,_=downsample(X[tr],y[tr],g[tr],obs[tr],SEED+fold)
 for label,C in [('untuned_C1',1.0),('Bayesian_best_C1',1.0)]:
  model=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=300,solver='lbfgs'))
  t=time.perf_counter();model.fit(Xtr,ytr);train_s=time.perf_counter()-t
  t=time.perf_counter();p=model.predict_proba(X[va])[:,1];infer=time.perf_counter()-t
  yhat=(p>=.5).astype(int); best={}
  for i,key in enumerate(obs[va]):
   key=tuple(key)
   if key not in best or p[i]>best[key][0]:best[key]=(p[i],i)
  top=float(np.mean([int(y[va][i]==1) for _,i in best.values()]))
  rows.append({'model':label,'C':C,'fold':fold,'observation_top1_accuracy':top,'candidate_macro_f1':f1_score(y[va],yhat,average='macro',zero_division=0),'candidate_log_loss':log_loss(y[va],p,labels=[0,1]),'train_seconds':train_s,'inference_ms_per_observation':1000*infer/max(1,len(best)),'n_validation_observations':len(best)})
  print(label,'fold',fold,'top1',round(top,4),flush=True)
summary=[]
for label in ['untuned_C1','Bayesian_best_C1']:
 rr=[r for r in rows if r['model']==label]
 summary.append({'model':label,**{k:float(np.mean([r[k] for r in rr])) for k in ['observation_top1_accuracy','candidate_macro_f1','candidate_log_loss','train_seconds','inference_ms_per_observation']},'top1_std':float(np.std([r['observation_top1_accuracy'] for r in rr]))})
OUT.mkdir(exist_ok=True)
for fn,rs in [('bayesian_comparison_folds_aligned.csv',rows),('bayesian_comparison_aligned.csv',summary)]:
 with (OUT/fn).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
print(summary)
