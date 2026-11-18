"""Small Gaussian-process Bayesian search for Logistic Regression C.
Uses 3-fold GroupKFold by trip and only train_candidates.csv.
The objective is mean observation-level top-1 accuracy from heuristic labels.
"""
import csv, pathlib, time, warnings
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from train_models import DATA, OUT, load, downsample, SEED

N_TRIALS=10
INITIAL_LOG10_C=[-3.0,-1.0,0.0,1.0]

def top1(model,X,y,obs):
    p=model.predict_proba(X)[:,1]; best={}
    for i,key in enumerate(obs):
        key=tuple(key)
        if key not in best or p[i]>best[key][0]: best[key]=(p[i],i)
    return float(np.mean([int(y[i]==1) for _,i in best.values()]))

def objective(logc, folds):
    C=10.0**float(logc); scores=[]; train_times=[]
    for k,(Xtr,ytr,Xv,yv,obsv) in enumerate(folds,1):
        model=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=300,solver='lbfgs'))
        st=time.perf_counter();model.fit(Xtr,ytr);train_times.append(time.perf_counter()-st)
        scores.append(top1(model,Xv,yv,obsv))
    return {'C':C,'log10_C':float(logc),'mean_top1':float(np.mean(scores)),'std_top1':float(np.std(scores)),'fold_scores':scores,'mean_train_seconds':float(np.mean(train_times))}

def propose(xs,ys,rng):
    x=np.asarray(xs).reshape(-1,1); y=np.asarray(ys)
    gp=GaussianProcessRegressor(kernel=ConstantKernel(1.0)*Matern(length_scale=1.5,nu=2.5)+WhiteKernel(noise_level=1e-5),normalize_y=True,n_restarts_optimizer=2,random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');gp.fit(x,y)
    grid=np.linspace(-3,3,241);mu,std=gp.predict(grid.reshape(-1,1),return_std=True)
    best=max(ys); imp=mu-best-0.01; z=imp/np.maximum(std,1e-9); ei=imp*norm.cdf(z)+std*norm.pdf(z)
    for xx in xs: ei[np.argmin(abs(grid-xx))]=-1
    return float(grid[int(np.argmax(ei))])

def main():
    X,y,g,obs,_=load(); splitter=GroupKFold(n_splits=3); folds=[]
    for k,(tr,va) in enumerate(splitter.split(X,y,g),1):
        Xtr,ytr,_=downsample(X[tr],y[tr],g[tr],obs[tr],SEED+k)
        folds.append((Xtr,ytr,X[va],y[va],obs[va]))
        print(f'Prepared fold {k}: training={len(ytr):,}, validation={len(va):,}, validation trips={len(set(g[va]))}',flush=True)
    xs=[];ys=[];rows=[]
    for i in range(N_TRIALS):
        x=INITIAL_LOG10_C[i] if i<len(INITIAL_LOG10_C) else propose(xs,ys,np.random.default_rng(SEED+i))
        result=objective(x,folds);result['trial']=i+1;rows.append(result);xs.append(x);ys.append(result['mean_top1'])
        print(f"Trial {i+1}/{N_TRIALS}: C={result['C']:.6g}, CV top1={result['mean_top1']:.4f} +/- {result['std_top1']:.4f}",flush=True)
    OUT.mkdir(exist_ok=True)
    with (OUT/'bayesian_trials_aligned.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['trial','C','log10_C','mean_top1','std_top1','mean_train_seconds','fold_scores'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({**r,'fold_scores':';'.join(f'{v:.6f}' for v in r['fold_scores'])})
    best=max(rows,key=lambda r:r['mean_top1'])
    with (OUT/'bayesian_best_aligned.txt').open('w',encoding='utf-8') as f:
        f.write(f"Best C: {best['C']:.8g}\nBest 3-fold grouped CV observation top-1 accuracy: {best['mean_top1']:.6f} +/- {best['std_top1']:.6f}\nTrials: {N_TRIALS}\nSearch: Gaussian-process Bayesian optimization over log10(C) in [-3,3], 4 initial points then expected improvement.\nObjective: mean observation-level top-1 accuracy.\nPartitions: training trips only; student-test and evaluator-test were not used. Labels are heuristic route alignments.\n")
    print('Best:',best,flush=True);print('Saved features/bayesian_trials_aligned.csv and features/bayesian_best_aligned.txt',flush=True)
if __name__=='__main__':main()
