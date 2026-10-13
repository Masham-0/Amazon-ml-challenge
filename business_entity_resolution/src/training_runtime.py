"""Disk-backed fitting matrices and durable LightGBM training checkpoints."""
import gc
import hashlib
import json
import mmap
from pathlib import Path

import lightgbm as lgb
import numpy as np


def signature(path):
    path = Path(path)
    return [str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns]


def selected_dataset(records, mask, labels, names, params, scratch, weight=None, reference=None):
    """Avoid simultaneously retaining a whole source mapping and a heap copy."""
    scratch = Path(scratch)
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise ValueError('Cannot construct a fitting dataset with no rows')
    matrix = np.memmap(scratch, dtype=np.float32, mode='w+', shape=(len(indices), len(names)))
    try:
        for start in range(0, len(indices), 50000):
            ix = indices[start:start + 50000]
            matrix[start:start + len(ix)] = records['x'][ix]
            records._mmap.madvise(mmap.MADV_DONTNEED)
        matrix.flush()
        matrix._mmap.madvise(mmap.MADV_DONTNEED)
        dataset = lgb.Dataset(matrix, label=labels[indices],
                              weight=None if weight is None else weight[indices],
                              feature_name=names, params=params, reference=reference,
                              free_raw_data=False)
        dataset.construct()
    finally:
        del matrix
        scratch.unlink(missing_ok=True)
        gc.collect()
    return dataset


def train_checkpointed(params, dataset, rounds, directory, name, inputs,
                       valid_sets=None, patience=80, interval=100, extra_callbacks=()):
    """Resume completed boosting rounds; preserve validation patience and best round.

    A resumed fit can follow a different stochastic path from uninterrupted
    training. Its own validation result is always recomputed before selection.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    statefile = directory / f'{name}.training.json'
    protocol = {'params': params, 'rounds': rounds, 'inputs': inputs,
                'validation': bool(valid_sets), 'patience': patience,
                'lightgbm': lgb.__version__}
    # Normalize tuples and NumPy-free metadata before comparing JSON state.
    protocol = json.loads(json.dumps(protocol))
    state = json.loads(statefile.read_text()) if statefile.exists() else {}
    if state and state.get('protocol') != protocol:
        raise ValueError(f'Stale training checkpoint: {statefile}; use a new experiment directory')
    model = None
    if state:
        modelpath = directory / state['model']
        if hashlib.sha256(modelpath.read_bytes()).hexdigest() != state['sha256']:
            raise ValueError(f'Damaged training checkpoint: {modelpath}')
        model = lgb.Booster(model_file=str(modelpath))
        assert model.current_iteration() == state['iteration']
        if state['finished']:
            model.best_iteration = state['best_iteration'] + 1 if valid_sets else state['iteration']
            return model
        state['resumed'] = True
        print('Resuming', name, 'after', state['iteration'], 'boosting rounds', flush=True)
    else:
        state = {'protocol': protocol, 'iteration': 0, 'best_iteration': -1,
                 'best_loss': None, 'best_scores': [], 'resumed': False}

    def checkpoint(env):
        evaluations = env.evaluation_result_list or []
        if valid_sets:
            if len(evaluations) != 1 or evaluations[0][3]:
                raise ValueError('Checkpoint callback expects one minimized validation metric')
            loss = float(evaluations[0][2])
            if state['best_loss'] is None or loss < state['best_loss']:
                state['best_loss'] = loss
                state['best_iteration'] = env.iteration
                state['best_scores'] = [list(e) for e in evaluations]
        stop = bool(valid_sets) and env.iteration - state['best_iteration'] >= patience
        finished = stop or env.iteration + 1 >= rounds
        if (env.iteration + 1) % interval == 0 or finished:
            filename = f'{name}.{env.iteration + 1}.checkpoint.txt'
            destination = directory / filename
            temporary = directory / (filename + '.partial')
            env.model.save_model(str(temporary), num_iteration=-1)
            temporary.replace(destination)
            previous = state.get('model')
            state.update(iteration=env.iteration + 1, model=filename,
                         sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), finished=finished)
            temporary_state = statefile.with_suffix('.partial')
            temporary_state.write_text(json.dumps(state, indent=2))
            temporary_state.replace(statefile)
            if previous and previous != filename:
                (directory / previous).unlink(missing_ok=True)
            print('Saved training checkpoint:', name, state['iteration'],
                  'validation loss', state['best_loss'], flush=True)
        if finished and valid_sets:
            raise lgb.callback.EarlyStopException(state['best_iteration'], state['best_scores'])

    checkpoint.order = 20
    def release_raw(env):
        if env.iteration == env.begin_iteration:
            # Resuming requires raw values until LightGBM has initialized the
            # previous booster as predictor. Afterwards the binned data suffice.
            dataset.data = None
            for validation in valid_sets or []:
                validation.data = None
            gc.collect()
    release_raw.before_iteration = True
    release_raw.order = -20
    return lgb.train(params, dataset, num_boost_round=rounds - state['iteration'],
                     init_model=model, valid_sets=valid_sets,
                     callbacks=[release_raw, checkpoint, *extra_callbacks])
