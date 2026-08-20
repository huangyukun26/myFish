from __future__ import annotations

import hashlib, json, zipfile
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path("runs/structural_backbones_20260713")
OUT = Path("runs/cloud_20260714/seen_tri_router")

def fold(s):
    return int.from_bytes(hashlib.sha1(s.split()[0].encode()).digest()[:4], "little") % 4

def topk(x):
    if "logits" in x: return x["logits"].float().topk(20, 1)
    return x["topk_values"].float(), x["topk_indices"].long()

def feats(base, tri, da, db, counts):
    bv, bi = topk(base); tv, ti = topk(tri); av, ai = topk(da); dv, di = topk(db)
    bz=(bv-bv.mean(1,keepdim=True))/bv.std(1,keepdim=True).clamp_min(1e-6)
    tz=(tv-tv.mean(1,keepdim=True))/tv.std(1,keepdim=True).clamp_min(1e-6)
    bp,tp,ap,dp=bi[:,0],ti[:,0],ai[:,0],di[:,0]
    br=bi.eq(tp[:,None]); tr=ti.eq(bp[:,None])
    same=torch.tensor([base["classes"][int(x)].split()[0]==base["classes"][int(y)].split()[0] for x,y in zip(bp,tp)])
    x=torch.stack([
      bv[:,0]-bv[:,1],bv[:,0]-bv[:,4],bz[:,0],bz[:,0]-bz[:,1],
      tv[:,0]-tv[:,1],tv[:,0]-tv[:,4],tz[:,0],tz[:,0]-tz[:,1],
      br.any(1).float(),br.float().argmax(1)/20,tr.any(1).float(),tr.float().argmax(1)/20,
      tp.eq(ap).float(),tp.eq(dp).float(),ap.eq(dp).float(),bp.eq(ap).float(),bp.eq(dp).float(),same.float(),
      torch.log1p(counts[bp].float()),torch.log1p(counts[tp].float()),(counts[tp]<=2).float(),(counts[tp]<=5).float()
    ],1)
    return x.numpy(),bp,tp,br.any(1),same

def mdl(): return HistGradientBoostingRegressor(max_iter=200,learning_rate=.035,max_depth=3,min_samples_leaf=35,l2_regularization=8,random_state=2027)
def threshold(s,y):
    qs=np.unique(np.quantile(s,np.linspace(.45,.995,160)))
    return float(max(qs,key=lambda t:(y[s>=t].sum(),-(s>=t).sum())))
def jsonl(path,classes):
    pos={x:i for i,x in enumerate(classes)}; ids=[]; ii=[]; vv=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        r=json.loads(line);ids.append(r["image_id"]);ii.append([pos[x] for x in r["predictions"]]);vv.append(r["scores"])
    return {"classes":classes,"image_ids":ids,"topk_indices":torch.tensor(ii),"topk_values":torch.tensor(vv)}
def package(cur,ids,classes,current,cand,mask,name):
    p=current.clone();p[mask]=cand[mask]; d=dict(cur)
    for i,x in zip(ids,p):d[i]=classes[int(x)]
    q=OUT/name;q.mkdir(parents=True,exist_ok=True); f=q/"prediction.json";f.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding="utf-8")
    with zipfile.ZipFile(q/"submission.zip","w",zipfile.ZIP_DEFLATED) as z:z.write(f,"prediction.json")
    return int(mask.sum())

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base=torch.load(ROOT/"concat_balanced_gate/fixed_fusion_logits.pt",map_location="cpu",weights_only=False)
    tri=torch.load(ROOT/"triview_concat_gate/paired_random2027/fixed_fusion_taxon_logits.pt",map_location="cpu",weights_only=False)
    da=torch.load(ROOT/"dino_metric_full_holdout/prediction/test_seen_metric_seed2027_topk.pt",map_location="cpu",weights_only=False)
    db=torch.load(ROOT/"dino_metric_full_holdout/prediction/test_seen_metric_seed2028_topk.pt",map_location="cpu",weights_only=False)
    counts=base["full_class_counts"].long(); x,b,c,intop,same=feats(base,tri,da,db,counts); truth=base["class_ids"].long()
    y=(c.eq(truth).long()-b.eq(truth).long()).numpy(); groups=np.array([fold(s) for s in base["labels"]]); oof=np.zeros(len(y)); ts=[]; rows=[]
    for f in range(4):
      tr,te=groups!=f,groups==f;m=mdl().fit(x[tr],y[tr]);t=threshold(m.predict(x[tr]),y[tr]);s=m.predict(x[te]);take=s>=t;oof[te]=s;ts.append(t)
      rows.append({"fold":f,"changed":int(take.sum()),"net":int(y[te][take].sum()),"wins":int((y[te][take]==1).sum()),"losses":int((y[te][take]==-1).sum())})
    take=oof>=np.array([ts[f] for f in groups]); guard=intop.numpy()&same.numpy(); strict=take&guard
    summary={"oof":{"changed":int(take.sum()),"net":int(y[take].sum()),"folds":rows},"strict_oof":{"changed":int(strict.sum()),"net":int(y[strict].sum()),"wins":int((y[strict]==1).sum()),"losses":int((y[strict]==-1).sum())}}
    torch.save({"oof_score":torch.from_numpy(oof),"thresholds":ts,"groups":torch.from_numpy(groups),
                "base":b,"alternate":c,"truth":truth,"guard":torch.from_numpy(guard)}, OUT/"oof_router_scores.pt")
    m=mdl().fit(x,y)
    pb=torch.load(OUT.parent/"seen_dino_router/public_fusion.pt",map_location="cpu",weights_only=False)
    pda=torch.load(ROOT/"dino_metric_full_prediction/test_seen_metric_seed2027_topk.pt",map_location="cpu",weights_only=False)
    pdb=torch.load(ROOT/"dino_metric_full_prediction/test_seen_metric_seed2028_topk.pt",map_location="cpu",weights_only=False)
    pt=jsonl(Path("runs/cloud_20260714/triview_public_direct/topk.jsonl"),list(base["classes"])); px,pb0,pc,pintop,psame=feats(pb,pt,pda,pdb,counts);score=m.predict(px)
    cur=json.loads(Path("runs/submission_20260702_seen_router_unseen_pair_o70species_avg_letterbox/prediction.json").read_text(encoding="utf-8")); pos={x:i for i,x in enumerate(base["classes"])}; current=torch.tensor([pos[cur[i]] for i in pda["image_ids"]]); agree=current.eq(pb0); pg=pintop&psame
    masks={"medium":torch.from_numpy(score>=np.median(ts))&agree,"strict":torch.from_numpy(score>=np.median(ts))&agree&pg,"strict_high":torch.from_numpy(score>=max(ts))&agree&pg}
    summary["thresholds"]=ts;summary["public"]={n:package(cur,pda["image_ids"],base["classes"],current,pc,q,n) for n,q in masks.items()}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
