"""Viterbi route decoding and classifier-only comparison on student-test trips."""
import csv,math,os,pathlib,collections,joblib,numpy as np
from train_models import ROOT,FEATURES
OUT=ROOT/'features'
CAND=pathlib.Path(os.getenv('PATHSNAP_STUDENT_CANDIDATES',str(OUT/'student_test_candidates_aligned.csv')))
MODEL=pathlib.Path(os.getenv('PATHSNAP_MODEL_PATH',str(ROOT/'models'/'logistic_candidate_model.joblib')))
DATA=pathlib.Path(os.getenv('PATHSNAP_DATA_ROOT',str(ROOT/'data')))/'student_test'
GAP_BREAK_S=60.0

def continuity(path,arcs,obs_by_idx):
    idxs=sorted(path)
    pairs=[(path[i],path[i+1]) for i in idxs if i+1 in path and i+1 in obs_by_idx and obs_by_idx[i+1]['dt']<=GAP_BREAK_S]
    if not pairs:return float('nan')
    return sum(a==b or arcs[a][1]==arcs[b][0] for a,b in pairs)/len(pairs)

def decode(obs,arcs):
    # Break at missing observations and long GPS gaps; never force transitions over outages.
    segments=[];current=[];last_idx=None
    for item in obs:
        if not item['states']:
            if current: segments.append(current);current=[]
            last_idx=item['idx'];continue
        if current and (item['idx']!=last_idx+1 or item['dt']>GAP_BREAK_S):segments.append(current);current=[]
        current.append(item);last_idx=item['idx']
    if current:segments.append(current)
    decoded={}
    for seq in segments:
        scores=[];backs=[]
        for t,item in enumerate(seq):
            states=item['states'];em=np.log(np.maximum(item['probs'],1e-12))
            if t==0:scores.append(em);backs.append(None);continue
            prev=seq[t-1]['states'];prevscore=scores[-1];new=np.empty(len(states));back=np.empty(len(states),dtype=int)
            for j,b in enumerate(states):
                best=-float('inf');best_i=0
                for i,a in enumerate(prev):
                    if a==b:trans=0.0
                    elif arcs[a][1]==arcs[b][0]:trans=-1.0
                    else:trans=-5.0
                    val=prevscore[i]+trans
                    if val>best:best=val;best_i=i
                new[j]=best+em[j];back[j]=best_i
            scores.append(new);backs.append(back)
        pos=int(np.argmax(scores[-1]));path=[None]*len(seq)
        for t in range(len(seq)-1,-1,-1):
            path[t]=seq[t]['states'][pos]
            if t:pos=int(backs[t][pos])
        for item,aid in zip(seq,path):decoded[item['idx']]=aid
    return decoded

def main():
    bundle=joblib.load(MODEL);model=bundle['model'];feat=bundle['features'];
    grouped=collections.defaultdict(list)
    with CAND.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            vals=[float(r[k] or 0) for k in feat]; grouped[r['trip_id']].append((int(r['obs_idx']),int(r['candidate_arc_id']),vals,int(r['is_route_label']),int(r['label_known']),float(r['dt_s'] or 0)))
    out=OUT/'student_test_hmm_predictions_aligned.csv';summary=[];all_rows=[]
    for tid,rows in grouped.items():
        d=DATA/tid; arcs=[[int(v) for v in l.split()] for l in (d/f'{tid}.arcs').read_text().splitlines() if l.strip()]
        X=np.asarray([r[2] for r in rows],dtype=np.float32);prob=model.predict_proba(X)[:,1]
        byobs=collections.defaultdict(list)
        for r,p in zip(rows,prob):byobs[r[0]].append((r[1],p,r[3],r[4],r[5]))
        obs=[];classifier={};truth={};known={}
        for idx,cands in sorted(byobs.items()):
            states=[x[0] for x in cands];ps=np.asarray([x[1] for x in cands]);obs.append({'idx':idx,'states':states,'probs':ps,'dt':cands[0][4]})
            classifier[idx]=states[int(np.argmax(ps))]
            pos=[x[0] for x in cands if x[2]==1]
            if pos:truth[idx]=pos[0];known[idx]=True
        decoded=decode(obs,arcs)
        evalidx=[i for i in truth if i in decoded]
        baseacc=float(np.mean([classifier[i]==truth[i] for i in evalidx]));hmmacc=float(np.mean([decoded[i]==truth[i] for i in evalidx]))
        obs_by_idx={x['idx']:x for x in obs}
        sr={'trip_id':tid,'labeled_observations':len(evalidx),'classifier_top1_accuracy':baseacc,'hmm_top1_accuracy':hmmacc,'classifier_continuity':continuity(classifier,arcs,obs_by_idx),'hmm_continuity':continuity(decoded,arcs,obs_by_idx)};summary.append(sr);print(sr,flush=True)
        for i in sorted(decoded):all_rows.append({'trip_id':tid,'obs_idx':i,'classifier_arc_id':classifier[i],'hmm_arc_id':decoded[i],'reference_arc_id':truth.get(i,''),'label_known':int(i in truth)})
    with out.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    with (OUT/'student_test_hmm_summary_aligned.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print('macro means:',{k:float(np.nanmean([r[k] for r in summary])) for k in ['classifier_top1_accuracy','hmm_top1_accuracy','classifier_continuity','hmm_continuity']})
if __name__=='__main__':main()
