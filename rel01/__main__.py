"""Command-line entry points for prepared-feature reproduction."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from .config import SPEC, ScopeError

MAIN = ('M0', 'M2', 'C1-0')
ALL = ('M0', 'M1', 'M2', 'M3', 'C1-0', 'C1-1')


def write_json(path, value):
    from .storage import atomic_json
    import numpy as np
    def convert_scalar(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f'unsupported report value {type(obj).__name__}')
    value = json.loads(json.dumps(value, default=convert_scalar))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def analyze(run_root, output):
    from .evaluation import hierarchy, paired_bootstrap, counterfactual_summary, mu3d_condition_subset
    from .routing import PRIMARY_CONTROLS
    root = Path(run_root).resolve()
    output = Path(output).resolve()
    if output.exists() or output == root or root.is_relative_to(output):
        raise ScopeError('choose a new analysis output path outside the run root and its ancestors')
    primary = defaultdict(lambda: defaultdict(list))
    fixed = defaultdict(lambda: defaultdict(list))
    source = defaultdict(lambda: defaultdict(list))
    groups = sorted(root.glob('*/GROUP_COMPLETE.json'))
    if not groups:
        raise ScopeError('no completed reproduction groups found')
    identities = set()
    branch_sets = {}
    for receipt in groups:
        group = receipt.parent
        manifest = json.loads(receipt.read_text())
        identity = (manifest['dataset'], int(manifest['outer']), int(manifest['seed']))
        if identity in identities:
            raise ScopeError('duplicate outer-fold and seed group')
        identities.add(identity)
        from .storage import sha
        for name, expected in manifest['sealed_files_sha256'].items():
            if Path(name).name != name or sha(group/name) != expected:
                raise ScopeError('prediction file integrity mismatch')
        branches = manifest.get('branches', ALL)
        if identity[0] in branch_sets and tuple(branches) != branch_sets[identity[0]]:
            raise ScopeError('model inventory differs between outer-fold or seed groups')
        branch_sets[identity[0]] = tuple(branches)
        for model in branches:
            rows = json.loads((group/f'{model}_SEALED.json').read_text())
            if any(row['model'] != model or row['control'] != PRIMARY_CONTROLS[model]
                   or row['evaluation_kind'] != 'PRIMARY' or row['dataset'] != identity[0]
                   or row['outer_fold'] != identity[1] or row['seed'] != identity[2] for row in rows):
                raise ScopeError('prediction experiment identity mismatch')
            primary[(identity[0], model)]['rows'].extend(rows)
            if model not in ('M0', 'M1'):
                for suffix, target in [('FIXED', fixed), ('SOURCE', source)]:
                    path = group/f'{model}_{suffix}_CONTROLS_SEALED.json'
                    if path.exists():
                        for mode, values in json.loads(path.read_text())['predictions'].items():
                            target[(identity[0], model)][mode].extend(values)
    datasets = sorted({key[0] for key in identities})
    for dataset in datasets:
        expected = {(dataset, outer, seed) for outer in range(5) for seed in SPEC.seeds}
        if {key for key in identities if key[0] == dataset} != expected:
            raise ScopeError('analysis requires all five outer folds and three seeds for each dataset')
    report = {'primary': {}, 'paired': {}, 'fixed_controls': {}, 'source_controls': {}, 'mu3d_conditions': {}}
    for (dataset, model), obj in primary.items():
        report['primary'][f'{dataset}/{model}'] = hierarchy(obj['rows'])
    for dataset in datasets:
        for left, right in [('M2', 'M0'), ('M2', 'C1-0'), ('M1', 'M0'), ('M3', 'M1'), ('M3', 'C1-1')]:
            if (dataset, left) in primary and (dataset, right) in primary:
                report['paired'][f'{dataset}/{left}-{right}'] = paired_bootstrap(
                    primary[(dataset, left)]['rows'], primary[(dataset, right)]['rows'])
    for name, collection in [('fixed_controls', fixed), ('source_controls', source)]:
        for (dataset, model), controls in collection.items():
            if controls:
                report[name][f'{dataset}/{model}'] = counterfactual_summary(controls)
    if 'MU3D' in datasets:
        conditions = json.loads((Path(__file__).resolve().parent/'protocol/mu3d_conditions.json').read_text())
        for group in ['A_SAME_QUERY_VALENCE', 'B_SAME_QUERY_REFERENT']:
            selected = {}
            for (dataset, model), obj in primary.items():
                if dataset == 'MU3D':
                    selected[model] = mu3d_condition_subset(obj['rows'], conditions, group)
            report['mu3d_conditions'][group] = {
                'metrics': {model: hierarchy(rows) for model, rows in selected.items()},
                'paired': {f'{a}-{b}': paired_bootstrap(selected[a], selected[b])
                           for a,b in [('M2','M0'),('M2','C1-0')] if a in selected and b in selected},
            }
    write_json(output, report)
    print(json.dumps({'report': str(output), 'groups': len(groups)}))


def main():
    parser = argparse.ArgumentParser(description='SHR-CoRel reproduction from user-prepared features')
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ['validate', 'train']:
        child = commands.add_parser(command)
        child.add_argument('--dataset', choices=['DOLOS', 'MU3D'], required=True)
        child.add_argument('--embeddings', type=Path, required=True)
        child.add_argument('--lexical', type=Path)
        if command == 'train':
            child.add_argument('--output', type=Path, required=True)
            child.add_argument('--outer', type=int, choices=range(5), nargs='+', default=list(range(5)))
            child.add_argument('--seeds', type=int, choices=SPEC.seeds, nargs='+', default=list(SPEC.seeds))
            child.add_argument('--models', choices=['main', 'all'], default='main')
            child.add_argument('--device', default='cpu')
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--run', type=Path, required=True)
    evaluation.add_argument('--output', type=Path, required=True)
    commands.add_parser('self-check')
    args = parser.parse_args()
    if args.command == 'self-check':
        import unittest
        from .config import ROOT
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.command == 'evaluate':
        analyze(args.run, args.output)
        return
    from .inputs import load_inputs
    data, scopes = load_inputs(args.dataset, args.embeddings, args.lexical)
    if args.command == 'validate':
        print(json.dumps({'dataset': data.dataset, 'samples': len(data.ids),
                          'participants': len(set(data.participants)), 'scopes': len(scopes),
                          'data_fingerprint': data.fingerprint}))
        return
    if args.models == 'all' and args.lexical is None:
        raise ScopeError('--models all requires the prepared 36-column lexical measurements')
    if len(set(args.outer)) != len(args.outer) or len(set(args.seeds)) != len(args.seeds):
        raise ScopeError('duplicate requested folds or seeds')
    from .authorization import Permit
    from .runner import run_group
    from .config import ROOT
    run_root = args.output.resolve()
    protected = [ROOT.resolve(), args.embeddings.resolve()]
    if args.lexical is not None:
        protected.append(args.lexical.resolve())
    if any(run_root == p or run_root in p.parents for p in protected) or run_root.is_relative_to(ROOT.resolve()):
        raise ScopeError('choose an output directory outside the repository and input files')
    for outer in args.outer:
        for seed in args.seeds:
            name = f'{args.dataset}_o{outer}_s{seed}'
            print(f'Running {name}', flush=True)
            run_group(data, scopes, outer, seed, run_root/name, args.device,
                      Permit(data), branches=ALL if args.models == 'all' else MAIN)
    print(json.dumps({'completed_groups': len(args.outer)*len(args.seeds), 'output': str(run_root)}))


if __name__ == '__main__':
    main()
