"""Five-fold group-aware CV by trip for PathSnap's initial model comparison."""
import csv, pathlib, time, random
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from train_models import DATA, OUT, FEATURES, SEED, load, downsample

def metrics(model,X,y,obs):
    start=time.perf_counter(); prob=model.predict_proba(X)[:,1]; infer=time.perf_counter()-start
    hard=(prob>=.5).astype(int); best={}
    for i,k in enumerate(obs):
        k=tuple(k)
        if k not in best or prob[i]>best[k][0]: best[k]=(prob[i],i)
    top=float(np.mean([int(y[i]==1) for _,i in best.values()]))
    return {'candidate_accuracy':accuracy_score(y,hard),'candidate_macro_f1':f1_score(y,hard,average='macro',zero_division=0),'candidate_log_loss':log_loss(y,prob,labels=[0,1]),'observation_top1_accuracy':top,'inference_ms_per_observation':1000*infer/max(1,len(best)),'n_val_observations':len(best)}

def make_models():
    return {
      'Logistic Regression':make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=300,solver='lbfgs')),
      'Decision Tree':DecisionTreeClassifier(max_depth=16,min_samples_leaf=20,class_weight='balanced',random_state=SEED),
      'Random Forest':RandomForestClassifier(n_estimators=80,max_depth=16,min_samples_leaf=5,max_features='sqrt',class_weight='balanced_subsample',n_jobs=-1,random_state=SEED),
      'Extra Trees':ExtraTreesClassifier(n_estimators=80,max_depth=16,min_samples_leaf=5,max_features='sqrt',class_weight='balanced',n_jobs=-1,random_state=SEED),
      'Hist Gradient Boosting':HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=31,l2_regularization=1.0,random_state=SEED)
    }

def main():
    X,y,g,obs,_=load(); splitter=GroupKFold(n_splits=5); folds=[]
    for fold,(tr,va) in enumerate(splitter.split(X,y,g),1):
        Xtr,ytr,gtr=downsample(X[tr],y[tr],g[tr],obs[tr],SEED+fold)
        print(f'Fold {fold}: train trips={len(set(g[tr]))}, validation trips={len(set(g[va]))}, sampled train={len(ytr):,}, validation={len(va):,}',flush=True)
        for name,model in make_models().items():
            start=time.perf_counter(); model.fit(Xtr,ytr); train_s=time.perf_counter()-start
            m=metrics(model,X[va],y[va],obs[va]); m.update(model=name,fold=fold,train_seconds=train_s)
            folds.append(m); print(name, 'top1',round(m['observation_top1_accuracy'],4),'macroF1',round(m['candidate_macro_f1'],4),flush=True)
    OUT.mkdir(exist_ok=True)
    with (OUT/'group_cv_folds_aligned.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(folds[0]));w.writeheader();w.writerows(folds)
    fields=['model','observation_top1_mean','observation_top1_std','candidate_macro_f1_mean','candidate_macro_f1_std','candidate_log_loss_mean','candidate_log_loss_std','train_seconds_mean','inference_ms_per_observation_mean']
    summary=[]
    for name in dict.fromkeys(r['model'] for r in folds):
        rr=[r for r in folds if r['model']==name]
        summary.append({'model':name,'observation_top1_mean':np.mean([r['observation_top1_accuracy'] for r in rr]),'observation_top1_std':np.std([r['observation_top1_accuracy'] for r in rr]),'candidate_macro_f1_mean':np.mean([r['candidate_macro_f1'] for r in rr]),'candidate_macro_f1_std':np.std([r['candidate_macro_f1'] for r in rr]),'candidate_log_loss_mean':np.mean([r['candidate_log_loss'] for r in rr]),'candidate_log_loss_std':np.std([r['candidate_log_loss'] for r in rr]),'train_seconds_mean':np.mean([r['train_seconds'] for r in rr]),'inference_ms_per_observation_mean':np.mean([r['inference_ms_per_observation'] for r in rr])})
    with (OUT/'group_cv_summary_aligned.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary)
    with (OUT/'group_cv_protocol_aligned.txt').open('w',encoding='utf-8') as f:
        f.write('5-fold GroupKFold, groups=trip_id, data=80 training trips only. Up to four negative candidates sampled per labeled observation in each training fold, random seed 42+fold. Validation rows are not downsampled. Labels are heuristic route alignment; no student-test or evaluator-test data used.\n')
    for r in summary: print(r,flush=True)
    print('Saved features/group_cv_folds_aligned.csv and features/group_cv_summary_aligned.csv',flush=True)
if __name__=='__main__':main()
