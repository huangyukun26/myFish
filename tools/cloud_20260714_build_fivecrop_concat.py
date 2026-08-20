import argparse
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--five',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
b=torch.load(a.base,map_location='cpu',weights_only=False);f=torch.load(a.five,map_location='cpu',weights_only=False);pos={x:i for i,x in enumerate(f['image_ids'])};missing=[x for x in b['image_ids'] if x not in pos]
if missing:raise RuntimeError(f'{len(missing)} missing ids, first={missing[:3]}')
idx=torch.tensor([pos[x] for x in b['image_ids']]);out=dict(b);out['features']=torch.cat([b['features'].float(),f['features'][idx].float()],dim=1);out['fivecrop_source']=str(a.five);out['component_dims']=[b['features'].shape[1],f['features'].shape[1]];a.out.parent.mkdir(parents=True,exist_ok=True);torch.save(out,a.out);print({'rows':len(idx),'dim':out['features'].shape[1],'out':str(a.out)})
