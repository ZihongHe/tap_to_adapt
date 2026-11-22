"""Extract frozen, multiscale DistilHuBERT features for the public pilot only."""
import argparse
import csv
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModel

MODEL='ntu-spml/distilhubert'
REVISION='fa87d96265d6b7af66e112faff6ff44df419cec9'


def main():
    """Read the public manifest and write pooled acoustic features."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,default=Path('data/pilot_1000'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Output already exists')
    extractor=AutoFeatureExtractor.from_pretrained(MODEL,revision=REVISION)
    model=AutoModel.from_pretrained(MODEL,revision=REVISION).to(args.device).eval()
    with (args.corpus/'labels.csv').open(encoding='utf-8',newline='') as f:
        records=list(csv.DictReader(f))
    features=[]
    with torch.inference_mode():
        for row in records:
            waveform,sr=sf.read(args.corpus/row['file'],dtype='float32')
            if sr!=16000 or waveform.shape!=(240000,):raise ValueError('Unexpected pilot waveform format')
            inputs=extractor(waveform,sampling_rate=sr,return_tensors='pt')
            # Match the original cache's CUDA mixed-precision inference;
            # CPU extraction remains float32 and can differ numerically.
            with torch.autocast(device_type='cuda',dtype=torch.float16,
                                enabled=args.device.startswith('cuda')):
                hidden=model(**{k:v.to(args.device) for k,v in inputs.items()}).last_hidden_state[0].float()
            n=len(hidden);vectors=[]
            for seconds in (15,8,4,2,1,.5,.25):
                count=max(1,min(n,round(seconds*n/15)))
                part=hidden[-count:]
                vectors.extend((part.mean(0),part.std(0,unbiased=False)))
            features.append(torch.cat(vectors).cpu().numpy())
    with args.output.open('xb') as f:
        np.savez_compressed(f,features=np.stack(features),
            labels=np.array([int(r['label']) for r in records]),
            event_indices=np.array([int(r['event_index']) for r in records]))
    print('Pilot features written.')


if __name__=='__main__':main()
