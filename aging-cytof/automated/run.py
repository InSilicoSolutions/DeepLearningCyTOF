import subprocess

def run(phenotype):
    args = ['python','train.py',phenotype,'2>&1']
    print(args)
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with open(f'stdout/{phenotype}.txt','wb') as wf:
        wf.write(proc.stdout)

if __name__ == '__main__':
    from pathlib import Path
    from multiprocessing import Pool
    import matplotlib.pyplot as plt
    import pickle
    import numpy as np

    with open('pheno-cols.txt') as f:
        phenos = f.read().split('\n')
    pool = Pool(4)
    pool.map(run, phenos)
    pool.close()
    pool.terminate()

    results_dir = Path('result')
    results_dir.mkdir(exist_ok=True)
    results = {}
    for pheno in phenos:
        d = results_dir/pheno
        result_pkl = d/'results.pkl'
        if not result_pkl.exists():
            continue
        with open(d/'results.pkl','rb') as f:
            results[pheno] = pickle.load(f)
    phenos = sorted(results.keys())

    # Train history
    fig, axes = plt.subplots(5,8, figsize=(30,20))
    flaxes = [axis for _ in axes for axis in _]
    for i, pheno in enumerate(phenos):
        ax = flaxes[i]
        train_hist = results[pheno]['history']['mean_absolute_error']
        valid_hist = results[pheno]['history']['val_mean_absolute_error']
        ax.plot(train_hist)
        ax.plot(valid_hist)
        ax.set_title(pheno)
    fig.supylabel('Loss')
    fig.supxlabel('Epoch')
    plt.tight_layout()
    fig.savefig('training-history.png')

    # Pred vs True
    fig, axes = plt.subplots(5,8, figsize=(30,20))
    flaxes = [axis for _ in axes for axis in _]
    for i, pheno in enumerate(phenos):
        ax = flaxes[i]
        vals_true = results[pheno]['y_valid']
        vals_pred = results[pheno]['y_scores']
        ax.plot(vals_true, vals_pred, 'b.')
        limits = (min(min(vals_true), min(vals_pred)), max(max(vals_true), max(vals_pred)))
        limits = (limits[0] - 0.1*(limits[1] - limits[0]), limits[1] + 0.1*(limits[1] - limits[0]))
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_title(pheno)
        ax.plot(limits,limits,'--k')
        coef = np.polyfit(vals_true, vals_pred, 1)
        poly1d_fn = np.poly1d(coef)
        ax.plot(limits, poly1d_fn(limits), '--b')
    fig.supylabel('True Value')
    fig.supxlabel('Predicted Value')
    plt.tight_layout()
    fig.savefig('pred-vs-true.png')