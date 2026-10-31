"""Sequential delivery: verified Phase 4 -> Phase 7 -> Phase 5. No broker calls."""
import argparse
import json
from pathlib import Path
import time
from continuous_hedging.config import FitConfig
from continuous_hedging.phase7 import run as bridge
from continuous_hedging.phase5 import run as robustness
from continuous_hedging.experiment_utils import dump,sha


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase4',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--daily',type=Path,default=root/'data/raw/upstox/nifty_daily_to_20261029.parquet')
    parser.add_argument('--wait-phase4',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    verification=args.phase4/'verification.json'
    if args.wait_phase4:
        print('Waiting for the running Phase 4 experiment to finish',flush=True)
        deadline=time.monotonic()+43200
        while not verification.exists():
            if time.monotonic()>deadline:raise TimeoutError('Phase 4 did not finish in the bounded wait')
            time.sleep(5)
    if not verification.exists() or not json.loads(verification.read_text())['mvp_gates_pass']:
        raise ValueError('Phase 4 must have passed before the bridge is run')
    dump(args.output/'delivery_plan.json',{'phase_order':[4,7,5],'phase4':str(args.phase4.resolve()),
         'phase4_frozen_manifest_sha256':sha(args.phase4/'frozen_before_test.json'),'daily_source':str(args.daily.resolve())})
    print('Starting Phase 7 matched rounding bridge',flush=True)
    bridge(args.output/'phase7',args.phase4,FitConfig())
    print('Starting Phase 5 robustness and daily-bootstrap research',flush=True)
    robustness(args.output/'phase5',args.daily,FitConfig(batch=2048))
    print('All requested experiment stages finished; reporting and verification artifacts are available',flush=True)


if __name__=='__main__':main()
