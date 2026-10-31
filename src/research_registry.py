"""Explicit registry for corrected research artifacts; does not promote legacy models."""
import argparse
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import importlib.metadata


ROOT=Path(__file__).resolve().parents[1]


def sha256(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def freeze(output):
    if output.exists():raise FileExistsError("Preserving frozen registry")
    stage=ROOT/'data/research/expanded_20261030'
    files=[*sorted((stage/'fresh_evaluation').glob('*_frozen_model.joblib')),
           stage/'data/processed/options_signal_eligible.parquet',stage/'data/processed/options_features.parquet',
           stage/'data/processed/ml_dataset.parquet']
    for directory in ('deep_hedging_part2_final_20261031','historical_hedging_part3_final_20261031'):
        p=ROOT/'data/research'/directory
        files.extend(sorted(p.glob('*/policy_seed_*.pt')))
    descriptor=ROOT/'research_runs.json'
    if descriptor.exists():
        runs=json.loads(descriptor.read_text())
        files.append(descriptor)
        extension_report=ROOT/'RESEARCH_PHASES_4_7_5.md'
        if extension_report.exists():files.append(extension_report)
        continuous=ROOT/runs['continuous_phase1_3']
        if not continuous.resolve().is_relative_to((ROOT/'data/research').resolve()):
            raise ValueError('Continuous run must remain inside the research workspace')
        if (continuous/'frozen_before_test.json').exists():
            files.extend(sorted(continuous.glob('*/*.pt')))
            files.extend([continuous/'experiment_plan.json',continuous/'frozen_before_test.json',continuous/'test_comparison.csv',continuous/'verification.json'])
        for key in ('continuous_phase4','rounding_phase7','robustness_phase5'):
            if key not in runs:continue
            extension=ROOT/runs[key]
            if not extension.resolve().is_relative_to((ROOT/'data/research').resolve()):
                raise ValueError('Extension run must remain inside the research workspace')
            if not (extension/'verification.json').exists() or not (extension/'full_replay_verification.json').exists():
                raise ValueError(f'Incomplete extension: {key}')
            files.extend(sorted(p for p in extension.rglob('*') if p.is_file() and '__pycache__' not in p.parts))
            for name in ('delivery_plan.json','supplemental_protocol.json','supplemental_protocol_v2.json'):
                extra=extension.parent/name
                if extra.exists():files.append(extra)
    if not all(p.exists() for p in files) or len(files)<8:raise ValueError("Canonical research artifacts unavailable")
    payload={"created_at":datetime.now(timezone.utc).isoformat(),"stage":str(stage.relative_to(ROOT)),
             "purpose":"Corrected historical research only; no production promotion",
             "artifacts":{str(p.relative_to(ROOT)):sha256(p) for p in files},
             "source":{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'src').rglob('*.py')},
             "environment":dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata['Name']))}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,indent=2))
    return payload


def verify(file,check_source=False):
    saved=json.loads(Path(file).read_text())
    for name,expected in saved['artifacts'].items():
        if sha256(ROOT/name)!=expected:raise ValueError(f"Research artifact changed: {name}")
    if check_source:
        for name,expected in saved['source'].items():
            if sha256(ROOT/name)!=expected:raise ValueError(f"Research code changed: {name}")
    return ROOT/saved['stage']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['freeze','verify'])
    parser.add_argument('--file',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='freeze':freeze(args.file)
    else:verify(args.file)
    print(args.command,'complete:',args.file)
