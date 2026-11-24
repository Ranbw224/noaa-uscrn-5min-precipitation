"""Load a saved local research model through its portable, hash-verified index.

Load joblib files only from this known, reviewed release: pickle-based models
are executable objects, not a safe format for arbitrary untrusted uploads.
"""
from pathlib import Path
import argparse
import hashlib
import json
import joblib

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('release',type=Path)
    p.add_argument('--variable',default='precipitation')
    p.add_argument('--track',default='official_hourly')
    args=p.parse_args()
    models=json.loads((args.release/'model_metadata.json').read_bytes())
    entry=next(m for m in models if m['variable']==args.variable and m['track']==args.track)
    path=args.release/entry['path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
        raise ValueError('Model hash mismatch')
    bundle=joblib.load(path)
    print(json.dumps({'variable':args.variable,'track':args.track,'feature_names':bundle['feature_names'],
                      'training_rows':entry['training_rows'],'actual_training_stations':len(entry['training_wbans'])},indent=2))

if __name__=='__main__':
    main()
