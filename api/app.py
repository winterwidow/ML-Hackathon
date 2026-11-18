"""FastAPI server for PathSnap. Run with uvicorn api.app:app --host 0.0.0.0 --port 8000."""
import os, pathlib, sys, csv, math
import joblib, numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pathsnap_person1 import read_rows, project, build_segments, spatial_index, nearby, bearing, wrap_angle
from hmm_decode import decode, GAP_BREAK_S

PARTITION=os.getenv('PATHSNAP_PARTITION','student_test')
DATA_ROOT=pathlib.Path(os.getenv('PATHSNAP_DATA_ROOT',str(ROOT/'data')))
MODEL_PATH=pathlib.Path(os.getenv('PATHSNAP_MODEL_PATH',str(ROOT/'models'/'extra_trees_candidate_model.joblib')))
BUNDLE=joblib.load(MODEL_PATH)
MODEL=BUNDLE['model']; FEATURES=BUNDLE['features']
app=FastAPI(title='PathSnap API',version='0.1.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])
app.mount('/static',StaticFiles(directory=pathlib.Path(__file__).parent/'static'),name='static')
class PredictRequest(BaseModel): trip_id:str
@app.get('/')
def home():return FileResponse(pathlib.Path(__file__).parent/'static'/'index.html')
@app.get('/health')
def health():return {'status':'ok','partition':PARTITION,'model':'Extra Trees (160 trees, depth 16, leaf 10)'}
@app.get('/trips')
def trips():
    folder=DATA_ROOT/PARTITION
    if not folder.exists():raise HTTPException(503,f'Data partition not found: {folder}')
    return {'trips':[p.name for p in sorted(folder.iterdir()) if p.is_dir()]}

def infer(folder):
    tid=folder.name
    track=[[float(v) for v in r] for r in read_rows(folder/f'{tid}.track',3)]
    nodes=[[float(v) for v in r] for r in read_rows(folder/f'{tid}.nodes',2)]
    arcs=[[int(v) for v in r] for r in read_rows(folder/f'{tid}.arcs',2)]
    lat0=sum(p[1] for p in track)/len(track); points=[(*project(lon,lat,lat0),t) for lon,lat,t in track]
    seg=build_segments(nodes,arcs,lat0); grid=spatial_index(seg); features=[]; grouped=[]
    for i,(x,y,t) in enumerate(points):
        prev=points[i-1] if i else None; nxt=points[i+1] if i+1<len(points) else None
        dt=(t-prev[2]) if prev else 0.0
        move=bearing(x-prev[0],y-prev[1]) if prev and dt>0 else (bearing(nxt[0]-x,nxt[1]-y) if nxt else 0.0)
        speed=math.hypot(x-prev[0],y-prev[1])/dt if prev and dt>0 else 0.0
        cand=nearby(x,y,grid,radius=80.0)
        if not cand:cand=nearby(x,y,grid,radius=500.0)
        if not cand:grouped.append({'idx':i,'states':[],'probs':np.array([]),'dt':dt});continue
        vals=[]
        for dist,s in cand:
            aid,_,_,dx,dy,den,length,road_b=s
            row={'dt_s':dt,'gps_speed_mps':speed,'candidate_dist_m':dist,'road_length_m':length,'gps_move_bearing_rad':move,'road_bearing_rad':road_b,'bearing_diff_rad':abs(wrap_angle(move-road_b))}
            vals.append([row[k] for k in FEATURES]);features.append((i,aid,row))
        prob=MODEL.predict_proba(np.asarray(vals,dtype=np.float32))[:,1]
        # nearby() returns (distance, segment_tuple); the decoder needs trip-local arc IDs.
        grouped.append({'idx':i,'states':[int(z[1][0]) for z in cand],'probs':prob,'dt':dt})
    decoded=decode(grouped,arcs)
    pred=[decoded.get(i) for i in range(len(track))]
    route_coords=[]
    for aid in pred:
        if aid is None:continue
        u,v=arcs[aid];route_coords.extend([[nodes[u][0],nodes[u][1]],[nodes[v][0],nodes[v][1]]])
    result={'trip_id':tid,'predicted_arc_ids':pred,'predicted_route_coordinates':route_coords,'gps_coordinates':[[p[0],p[1]] for p in track],'observations':len(track)}
    # Reference labels are read only after inference, server-side, solely for evaluator scoring.
    routefile=folder/f'{tid}.route'
    if routefile.exists():
        from pathsnap_person1 import process_trip
        labelrows,label_summary=process_trip(folder,PARTITION); truth={}
        for r in labelrows:
            if r['label_known'] and r['is_route_label']:truth.setdefault(r['obs_idx'],r['candidate_arc_id'])
        # Count every route-aligned GPS observation. Missing candidates or decoded states
        # count as misses instead of silently disappearing from the score denominator.
        scored=label_summary['gps_observations']-label_summary['unlabeled_observations']
        correct=sum(pred[i]==aid for i,aid in truth.items() if i<len(pred) and pred[i] is not None)
        result['accuracy']=correct/scored if scored else None
        result['scored_observations']=scored
        result['candidate_covered_observations']=len(truth)
        result['candidate_misses']=max(0,scored-len(truth))
    return result
@app.post('/predict')
def predict(req:PredictRequest):
    partition_dir=(DATA_ROOT/PARTITION).resolve()
    folder=(partition_dir/req.trip_id).resolve()
    if folder.parent!=partition_dir or not folder.is_dir():raise HTTPException(404,'Unknown trip ID')
    try:return infer(folder)
    except Exception as e:raise HTTPException(500,str(e))