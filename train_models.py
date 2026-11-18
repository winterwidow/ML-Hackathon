"""Initial grouped holdout comparison for PathSnap candidate scoring.
Labels are heuristic route-to-GPS alignments; metrics are provisional.
Run from the PathSnap project root: python train_models.py
"""
import csv, pathlib, random, time
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

ROOT=pathlib.Path(__file__).resolve().parent
DATA=ROOT/'features'/'train_candidates_aligned.csv'
OUT=ROOT/'features'
# Never feed alignment-only fields (route_alignment_dist_m / aligned_route_arc_id)
# into the model: those depend on server-side reference labels and are unavailable at inference.
FEATURES=['dt_s','gps_speed_mps','candidate_dist_m','road_length_m','gps_move_bearing_rad','road_bearing_rad','bearing_diff_rad']
SEED=42

# Keep every mapped positive and sample up to four negatives per observation.
def load():
    train_groups=set(); alltrips=set(); records=[]; groups=[]; obs=[]; y=[]; positive=[]
    with DATA.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['label_known']!='1': continue
            vals=[]
            for k in FEATURES:
                v=r[k]
                vals.append(float(v) if v!='' else 0.0)
            if not np.isfinite(vals).all(): continue
            records.append(vals); groups.append(r['trip_id']); obs.append((r['trip_id'],int(r['obs_idx']))); y.append(int(r['is_route_label'])); positive.append(r['aligned_route_arc_id'])
    return np.asarray(records,dtype=np.float32),np.asarray(y,dtype=np.int8),np.asarray(groups),np.asarray(obs,dtype=object),np.asarray(positive)

def downsample(X,y,groups,obs,seed=SEED):
    rng=random.Random(seed); bins={}
    for i,key in enumerate(obs): bins.setdefault(tuple(key),[]).append(i)
    keep=[]
    for ids in bins.values():
        pos=[i for i in ids if y[i]==1]; neg=[i for i in ids if y[i]==0]
        keep.extend(pos); keep.extend(rng.sample(neg,min(4,len(neg))))
    keep=np.asarray(keep,dtype=np.int64)
    return X[keep],y[keep],groups[keep]

def score(model,X,y,obs,route_ids):
    t=time.perf_counter(); p=model.predict_proba(X)[:,1]; elapsed=time.perf_counter()-t
    pred=(p>=0.5).astype(int)
    # Choose the highest-scoring candidate for each labeled GPS observation.
    best={}
    for i,key in enumerate(obs):
        key=tuple(key)
        if key not in best or p[i]>best[key][0]: best[key]=(p[i],i)
    top=[]
    for key,(_,i) in best.items(): top.append(int(y[i]==1))
    return {'candidate_accuracy':accuracy_score(y,pred),'candidate_macro_f1':f1_score(y,pred,average='macro',zero_division=0),'candidate_log_loss':log_loss(y,p,labels=[0,1]),'observation_top1_accuracy':float(np.mean(top)),'inference_ms_per_candidate':1000*elapsed/len(y),'inference_ms_per_observation':1000*elapsed/max(1,len(best)),'n_val_candidates':len(y),'n_val_observations':len(best)}

def main():
    if not DATA.exists(): raise SystemExit(f'Missing {DATA}; run pathsnap_person1.py first.')
    X,y,g,obs,routes=load(); trips=sorted(set(g)); rng=random.Random(SEED); rng.shuffle(trips)
    nval=max(1,round(.2*len(trips))); valtrips=set(trips[:nval]); valmask=np.isin(g,list(valtrips)); trmask=~valmask
    Xtr,ytr,gtr=downsample(X[trmask],y[trmask],g[trmask],obs[trmask]); Xv,yv,gv,obsv,rv=X[valmask],y[valmask],g[valmask],obs[valmask],routes[valmask]
    models={
      'Logistic Regression':make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=300,solver='lbfgs')),
      'Decision Tree':DecisionTreeClassifier(max_depth=16,min_samples_leaf=20,class_weight='balanced',random_state=SEED),
      'Random Forest':RandomForestClassifier(n_estimators=80,max_depth=16,min_samples_leaf=5,max_features='sqrt',class_weight='balanced_subsample',n_jobs=-1,random_state=SEED),
      'Extra Trees':ExtraTreesClassifier(n_estimators=80,max_depth=16,min_samples_leaf=5,max_features='sqrt',class_weight='balanced',n_jobs=-1,random_state=SEED),
      'Hist Gradient Boosting':HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=31,l2_regularization=1.0,random_state=SEED)
    }
    print(f'Known candidate rows: {len(y):,}; training trips: {len(set(g[trmask]))}; validation trips: {sorted(valtrips)}')
    print(f'Sampled training rows: {len(ytr):,}; holdout candidate rows: {len(yv):,}')
    results=[]
    for name,model in models.items():
        print('Fitting',name,flush=True); start=time.perf_counter(); model.fit(Xtr,ytr); train_s=time.perf_counter()-start
        m=score(model,Xv,yv,obsv,rv); m.update(model=name,train_seconds=train_s,training_rows=len(ytr),positive_train=int(ytr.sum()),positive_validation=int(yv.sum()))
        results.append(m); print(name, {k:round(v,4) if isinstance(v,float) else v for k,v in m.items()},flush=True)
    OUT.mkdir(exist_ok=True)
    fields=list(results[0])
    with (OUT/'initial_model_comparison_aligned.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
    with (OUT/'model_split_aligned.txt').open('w',encoding='utf-8') as f:
        f.write('Seed 42; 80%/20% grouped trip holdout. Validation trips: '+', '.join(sorted(valtrips))+'\n')
        f.write('Source labels are heuristic spatial route alignment; results are provisional. Student-test and evaluator-test partitions were not used.\n')
    print('Saved features/initial_model_comparison_aligned.csv and features/model_split_aligned.txt')
if __name__=='__main__': main()
