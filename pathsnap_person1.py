"""PathSnap Person 1 starter: trip parsing, route alignment, candidate features.
Uses only Python's standard library. Run:
  python pathsnap_person1.py DATA_ROOT OUTPUT_DIR [train|student_test]
DATA_ROOT contains train/ and student_test/ trip folders.
"""
from __future__ import annotations
import csv, math, pathlib, sys
from bisect import bisect_left
from collections import defaultdict

CELL_M = 100.0
CANDIDATE_RADIUS_M = 80.0
MAX_CANDIDATES = 8
LABEL_MAX_DISTANCE_M = 50.0


def read_rows(path, width):
    rows=[]
    with open(path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip(): continue
            parts=line.split()
            if len(parts)!=width: raise ValueError(f'{path}:{line_no}: expected {width} columns, got {len(parts)}')
            rows.append(parts)
    return rows


def bearing(dx, dy):
    # clockwise from north, radians
    return math.atan2(dx, dy)


def wrap_angle(a):
    return (a+math.pi)%(2*math.pi)-math.pi


def project(lon, lat, lat0):
    # Local equirectangular projection, in metres; suitable for each small trip area.
    return (lon*111320.0*math.cos(math.radians(lat0)), lat*110574.0)


def build_segments(nodes, arcs, lat0):
    pts=[project(lon,lat,lat0) for lon,lat in nodes]
    out=[]
    for aid,(u,v) in enumerate(arcs):
        x1,y1=pts[u]; x2,y2=pts[v]; dx=x2-x1; dy=y2-y1
        out.append((aid,x1,y1,dx,dy,dx*dx+dy*dy,math.hypot(dx,dy),bearing(dx,dy)))
    return out


def spatial_index(segments):
    grid=defaultdict(list)
    for s in segments:
        _,x,y,dx,dy,*_=s; x2=x+dx; y2=y+dy
        for gx in range(math.floor(min(x,x2)/CELL_M), math.floor(max(x,x2)/CELL_M)+1):
            for gy in range(math.floor(min(y,y2)/CELL_M), math.floor(max(y,y2)/CELL_M)+1): grid[(gx,gy)].append(s)
    return grid


def point_segment_distance(x,y,s):
    _,x1,y1,dx,dy,den,*_=s
    t=max(0.0,min(1.0,((x-x1)*dx+(y-y1)*dy)/den)) if den else 0.0
    return math.hypot(x-(x1+t*dx),y-(y1+t*dy))


def point_segment_fraction(x,y,s):
    _,x1,y1,dx,dy,den,*_=s
    return max(0.0,min(1.0,((x-x1)*dx+(y-y1)*dy)/den)) if den else 0.0


def nearby(x,y,grid,radius=CANDIDATE_RADIUS_M,limit=MAX_CANDIDATES):
    gx=math.floor(x/CELL_M); gy=math.floor(y/CELL_M); cells=math.ceil(radius/CELL_M)+1
    found={}
    for a in range(gx-cells,gx+cells+1):
        for b in range(gy-cells,gy+cells+1):
            for s in grid.get((a,b),()):
                if s[0] not in found: found[s[0]]=s
    scored=[(point_segment_distance(x,y,s),s) for s in found.values()]
    scored=[z for z in scored if z[0]<=radius]
    scored.sort(key=lambda z:z[0])
    return scored[:limit] if limit is not None else scored


def align_route_observations(points, route, route_segments, route_grid):
    """Greedy monotone alignment with a distance-based progress window.

    The progress window prevents a GPS point near a later crossing of the route
    from jumping hundreds of route arcs forward and stranding later observations.
    """
    positions=defaultdict(list)
    for i,aid in enumerate(route): positions[aid].append(i)
    cumulative=[0.0]
    for seg in route_segments: cumulative.append(cumulative[-1]+seg[6])
    aligned=[]; last=0; last_progress=0.0
    for i,(x,y,t) in enumerate(points):
        prev=points[i-1] if i else None
        dt=max(0.0,t-prev[2]) if prev else 0.0
        displacement=math.hypot(x-prev[0],y-prev[1]) if prev else 0.0
        max_progress=max(100.0,displacement*2.0+30.0*dt)
        # Bound pathological timestamp gaps without letting alignment jump to the
        # end of a route in one step.
        max_progress=min(max_progress,2000.0)
        choices=nearby(x,y,route_grid,radius=200.0,limit=None)
        valid=[]
        for dist,seg in choices:
            poslist=positions[seg[0]]
            k=0 if i==0 else bisect_left(poslist,last)
            while k<len(poslist):
                pos=poslist[k]
                progress=cumulative[pos]+point_segment_fraction(x,y,seg)*seg[6]
                advance=progress-last_progress
                if i==0 or (-20.0<=advance<=max_progress):
                    score=dist+0.05*abs(advance-displacement)
                    valid.append((score,dist,pos,progress))
                    break
                if advance>max_progress:break
                k+=1
        if valid:
            _,dist,pos,progress=min(valid)
            aligned.append((pos,dist));last=pos;last_progress=max(last_progress,progress)
        else:
            aligned.append((None,float('inf')))
    return aligned


def load_trip(folder):
    tid=folder.name
    track=[[float(v) for v in r] for r in read_rows(folder/f'{tid}.track',3)]
    nodes=[[float(v) for v in r] for r in read_rows(folder/f'{tid}.nodes',2)]
    arcs=[[int(v) for v in r] for r in read_rows(folder/f'{tid}.arcs',2)]
    route=[int(r[0]) for r in read_rows(folder/f'{tid}.route',1)]
    if not track or not nodes or not arcs or not route: raise ValueError(f'{tid}: empty required file')
    if any(not(-180<=lon<=180 and -90<=lat<=90 and math.isfinite(t)) for lon,lat,t in track): raise ValueError(f'{tid}: invalid GPS row')
    if any(u<0 or v<0 or u>=len(nodes) or v>=len(nodes) for u,v in arcs): raise ValueError(f'{tid}: arc endpoint outside nodes')
    if any(a<0 or a>=len(arcs) for a in route): raise ValueError(f'{tid}: route arc id outside arcs')
    if any(b[2]<=a[2] for a,b in zip(track,track[1:])): raise ValueError(f'{tid}: non-increasing timestamps')
    return track,nodes,arcs,route


def process_trip(folder, partition):
    track,nodes,arcs,route=load_trip(folder)
    lat0=sum(p[1] for p in track)/len(track)
    points=[(*project(lon,lat,lat0),t) for lon,lat,t in track]
    allseg=build_segments(nodes,arcs,lat0)
    route_seg=[allseg[a] for a in route]
    route_grid=spatial_index(route_seg)
    aligned=align_route_observations(points,route,route_seg,route_grid)
    # Use a grid for candidate road arcs; keep candidate IDs trip-local.
    road_grid=spatial_index(allseg)
    rows=[]
    for i,(x,y,t) in enumerate(points):
        prev=points[i-1] if i else None; nxt=points[i+1] if i+1<len(points) else None
        dt=(t-prev[2]) if prev else 0.0
        move_b=bearing(x-prev[0],y-prev[1]) if prev and dt>0 else (bearing(nxt[0]-x,nxt[1]-y) if nxt else 0.0)
        move_dist=math.hypot(x-prev[0],y-prev[1]) if prev else 0.0
        speed=move_dist/dt if dt>0 else 0.0
        route_pos, route_dist=aligned[i]
        label=route[route_pos] if route_pos is not None and route_dist<=LABEL_MAX_DISTANCE_M else None
        candidates=nearby(x,y,road_grid)
        for dist,s in candidates:
            aid,_,_,dx,dy,den,length,road_b=s
            rows.append({'partition':partition,'trip_id':folder.name,'obs_idx':i,'candidate_arc_id':aid,'gps_time_s':t,'dt_s':dt,'gps_speed_mps':speed,'gps_move_bearing_rad':move_b,'candidate_dist_m':dist,'road_length_m':length,'road_bearing_rad':road_b,'bearing_diff_rad':abs(wrap_angle(move_b-road_b)),'aligned_route_arc_id':label if label is not None else '', 'label_known':int(label is not None),'is_route_label':int(label is not None and aid==label),'route_alignment_dist_m':route_dist if math.isfinite(route_dist) else ''})
    return rows, {'trip_id':folder.name,'partition':partition,'gps_observations':len(track),'road_nodes':len(nodes),'road_arcs':len(arcs),'route_arcs':len(route),'candidate_rows':len(rows),'unlabeled_observations':sum(a is None or d>LABEL_MAX_DISTANCE_M for a,d in aligned)}


def main():
    if len(sys.argv)<3: raise SystemExit(__doc__)
    root=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); part=sys.argv[3] if len(sys.argv)>3 else 'train'
    if part not in ('train','student_test'): raise SystemExit('partition must be train or student_test')
    folders=sorted(p for p in (root/part).iterdir() if p.is_dir())
    out.mkdir(parents=True,exist_ok=True); summaries=[]; fieldnames=None
    filename=sys.argv[4] if len(sys.argv)>4 else f'{part}_candidates.csv'
    dest=out/filename
    with open(dest,'w',newline='',encoding='utf-8') as f:
        writer=None
        for folder in folders:
            rows,summary=process_trip(folder,part); summaries.append(summary)
            if writer is None and rows: fieldnames=list(rows[0]); writer=csv.DictWriter(f,fieldnames=fieldnames); writer.writeheader()
            if writer: writer.writerows(rows)
            print(summary)
    print(f'Wrote {dest}; {sum(s["candidate_rows"] for s in summaries)} candidate rows across {len(folders)} trips.')
    print('Label caution: route alignment is a spatial monotone heuristic; inspect unlabeled rates and maps before training.')

if __name__=='__main__': main()
