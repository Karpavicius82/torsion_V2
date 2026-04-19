"""
TORSIJOS LAUKU TEORIJOS PATVIRTINIMAS -- GA (v2, GREITAS)
Fixes: NO deepcopy, reuse single model, flush prints, numpy-only fitness
"""
import numpy as np
import h5py
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import time
import sys

def log(msg):
    print(msg, flush=True)

# ========================
# 1. DATASET
# ========================
def load_dataset(hdf5_path, n_input=8, n_output=8, max_samples=None):
    with h5py.File(hdf5_path, 'r') as f:
        S = f['t0_fields']['S'][:]  # [N, T, X]
    n_traj, n_time, nx = S.shape
    window = n_input + n_output
    inputs, outputs = [], []
    for traj in range(n_traj):
        for t0 in range(0, n_time - window + 1):
            inputs.append(S[traj, t0:t0+n_input])
            outputs.append(S[traj, t0+n_input:t0+window])
    inputs = np.array(inputs, dtype=np.float32)
    outputs = np.array(outputs, dtype=np.float32)
    if max_samples and len(inputs) > max_samples:
        idx = np.random.choice(len(inputs), max_samples, replace=False)
        inputs, outputs = inputs[idx], outputs[idx]
    log(f"  {Path(hdf5_path).name}: {inputs.shape[0]} samples [{n_input},{nx}]->[{n_output},{nx}]")
    return inputs, outputs

# ========================
# 2. KOMPAKTISKAS MODELIS
# ========================
class CompactOp(nn.Module):
    def __init__(self, ni=8, no=8, h=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ni, h, 7, padding=3), nn.Tanh(),
            nn.Conv1d(h, h, 5, padding=2), nn.Tanh(),
            nn.Conv1d(h, no, 5, padding=2),
        )
    def forward(self, x):
        return self.net(x)

def get_param_shapes(model):
    """Issaugo parametru formas VIENA karta."""
    shapes = []
    for p in model.parameters():
        shapes.append((p.shape, p.numel()))
    return shapes

def genome_size(shapes):
    return sum(n for _, n in shapes)

def load_genome_into_model(model, genome, shapes):
    """Uzkrauna genoma i ESAMA modeli BE KOPIJOS."""
    offset = 0
    for p, (shape, numel) in zip(model.parameters(), shapes):
        p.data.copy_(torch.from_numpy(genome[offset:offset+numel].reshape(shape)))
        offset += numel

# ========================
# 3. FITNESS (numpy greitkelis su vienu modeliu)
# ========================
@torch.no_grad()
def evaluate_one(model, x_t, y_t):
    pred = model(x_t)
    return -torch.mean((pred - y_t)**2).item()

def batch_evaluate(population, model, shapes, x_t, y_t, subset_idx):
    """Ivertina visa populiacija NAUDOJANT VIENA MODELI."""
    x_sub = x_t[subset_idx]
    y_sub = y_t[subset_idx]
    fitnesses = np.empty(len(population), dtype=np.float32)
    for i, genome in enumerate(population):
        load_genome_into_model(model, genome, shapes)
        fitnesses[i] = evaluate_one(model, x_sub, y_sub)
    return fitnesses

# ========================
# 4. GA OPERATORIAI (vektorizuoti)
# ========================
def init_pop(gsize, pop_size, scale=0.05):
    return [np.random.randn(gsize).astype(np.float32) * scale for _ in range(pop_size)]

def tournament(pop, fits, k=4):
    idx = np.random.choice(len(pop), k, replace=False)
    return pop[idx[np.argmax(fits[idx])]].copy()

def crossover(p1, p2, rate=0.7):
    if np.random.random() > rate:
        return p1.copy(), p2.copy()
    pt = np.random.randint(1, len(p1))
    return np.concatenate([p1[:pt], p2[pt:]]), np.concatenate([p2[:pt], p1[pt:]])

def mutate(g, rate=0.08, scale=0.025):
    mask = np.random.random(len(g)) < rate
    g[mask] += np.random.randn(mask.sum()).astype(np.float32) * scale
    return g

# ========================
# 5. GA CIKLAS
# ========================
def evolve(model, shapes, train_x, train_y, valid_x, valid_y,
           pop_size=60, gens=100, elite=4, subset=256):
    gsize = genome_size(shapes)
    log(f"\n  GA: genomas={gsize:,}, pop={pop_size}, gens={gens}, subset={subset}")

    tx = torch.from_numpy(train_x).float()
    ty = torch.from_numpy(train_y).float()
    vx = torch.from_numpy(valid_x).float()
    vy = torch.from_numpy(valid_y).float()

    pop = init_pop(gsize, pop_size)
    best_genome = pop[0].copy()
    best_fit = -np.inf
    hist_best, hist_mean, hist_valid = [], [], []

    t0 = time.time()
    for gen in range(gens):
        # Atsitiktinis subset kiekvienai kartai
        sub_idx = torch.randperm(tx.shape[0])[:subset]

        fits = batch_evaluate(pop, model, shapes, tx, ty, sub_idx)
        gen_best = np.max(fits)
        gen_mean = np.mean(fits)

        if gen_best > best_fit:
            best_fit = gen_best
            best_genome = pop[np.argmax(fits)].copy()

        hist_best.append(-gen_best)
        hist_mean.append(-gen_mean)

        # Validacija kas 10 kartu
        if gen % 10 == 0 or gen == gens - 1:
            load_genome_into_model(model, best_genome, shapes)
            vf = evaluate_one(model, vx, vy)
            hist_valid.append((gen, -vf))
            elapsed = time.time() - t0
            log(f"  Gen {gen:04d}: train_MSE={-gen_best:.4e}  "
                f"valid_MSE={-vf:.4e}  mean={-gen_mean:.4e}  t={elapsed:.1f}s")

        # Nauja karta
        elite_idx = np.argsort(fits)[-elite:]
        new_pop = [pop[i].copy() for i in elite_idx]

        while len(new_pop) < pop_size:
            p1 = tournament(pop, fits)
            p2 = tournament(pop, fits)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        pop = new_pop

    total = time.time() - t0
    log(f"\n  GA baigtas per {total:.1f}s, best train MSE={-best_fit:.4e}")
    return best_genome, hist_best, hist_mean, hist_valid, total

# ========================
# 6. MAIN
# ========================
if __name__ == '__main__':
    log("="*70)
    log("  TORSIJOS LAUKU TEORIJOS PATVIRTINIMAS (GA v2)")
    log("="*70)

    root = Path(r"d:\SISTEMOS\7. Antigravity2025\Old\torsinonao laukai")
    data = root / "torsion_well_data" / "datasets" / "torsion_1d" / "data"

    NI, NO = 8, 8
    log("\n--- Duomenys ---")
    trx, try_ = load_dataset(str(data/"train"/"torsion_1d_train.hdf5"), NI, NO, max_samples=1500)
    vax, vay = load_dataset(str(data/"valid"/"torsion_1d_valid.hdf5"), NI, NO)
    tex, tey = load_dataset(str(data/"test"/"torsion_1d_test.hdf5"), NI, NO)

    model = CompactOp(ni=NI, no=NO, h=24)
    shapes = get_param_shapes(model)
    gs = genome_size(shapes)
    log(f"  Modelio param: {gs:,}")

    log("\n--- GA Evoliucija ---")
    best_g, hb, hm, hv, total_t = evolve(
        model, shapes, trx, try_, vax, vay,
        pop_size=60, gens=100, elite=4, subset=256
    )

    # Test
    log("\n--- Testavimas ---")
    load_genome_into_model(model, best_g, shapes)
    tex_t = torch.from_numpy(tex).float()
    tey_t = torch.from_numpy(tey).float()
    with torch.no_grad():
        pred = model(tex_t).numpy()

    test_mse = np.mean((tey - pred)**2)
    rel_errs = [np.sqrt(np.sum((tey[i]-pred[i])**2))/max(np.sqrt(np.sum(tey[i]**2)),1e-15)
                for i in range(len(tey))]
    mean_rel = np.mean(rel_errs)
    ss_res = np.sum((tey.flatten()-pred.flatten())**2)
    ss_tot = np.sum((tey.flatten()-np.mean(tey.flatten()))**2)
    r2 = 1.0 - ss_res/max(ss_tot, 1e-15)

    log(f"  Test MSE: {test_mse:.6e}")
    log(f"  Rel L2:   {mean_rel*100:.2f}%")
    log(f"  R^2:      {r2:.6f}")

    torch.save(model.state_dict(), str(root/"torsion_ga_best.pt"))

    # Vizualizacija
    log("\n--- Vizualizacija ---")
    fig = plt.figure(figsize=(22, 26))
    fig.suptitle("TORSIJOS LAUKU TEORIJOS PATVIRTINIMAS (GA)", fontsize=17, fontweight='bold', y=0.99)
    gs_fig = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs_fig[0, 0])
    ax1.semilogy(hb, 'b-', lw=2, label='Best MSE')
    ax1.semilogy(hm, 'r-', lw=1, alpha=0.5, label='Mean MSE')
    ax1.set_xlabel('Karta'); ax1.set_ylabel('MSE (log)')
    ax1.set_title('GA Evoliucija'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs_fig[0, 1])
    if hv:
        vg, vm = zip(*hv)
        ax2.semilogy(vg, vm, 'go-', lw=2, ms=6)
    ax2.set_xlabel('Karta'); ax2.set_ylabel('Valid MSE')
    ax2.set_title('Validacija'); ax2.grid(True, alpha=0.3)

    nx = tey.shape[2]
    xg = np.linspace(0, 40, nx)
    for si in range(min(4, len(tey))):
        ax = fig.add_subplot(gs_fig[1 + si//2, si%2])
        t_mid, t_end = NO//2, NO-1
        ax.plot(xg, tey[si,t_mid], 'b-', lw=2, label='Tikras(mid)')
        ax.plot(xg, pred[si,t_mid], 'b--', lw=2, label='GA(mid)')
        ax.plot(xg, tey[si,t_end], 'r-', lw=2, label='Tikras(end)')
        ax.plot(xg, pred[si,t_end], 'r--', lw=2, label='GA(end)')
        re = np.sqrt(np.sum((tey[si]-pred[si])**2))/max(np.sqrt(np.sum(tey[si]**2)),1e-15)
        ax.set_title(f'#{si+1} (klaida: {re*100:.1f}%)')
        ax.set_xlabel('x'); ax.set_ylabel('S'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.savefig(str(root/"torsion_ga_confirmation.png"), dpi=150, bbox_inches='tight', facecolor='white')
    log(f"  Issaugota: torsion_ga_confirmation.png")

    log(f"\n{'='*70}")
    log(f"  VERDIKTAS: R^2 = {r2:.4f}")
    if r2 > 0.90:
        log("  *** PATVIRTINTA: Torsijos PDE turi nuoseklia, ismokstama fizika!")
    elif r2 > 0.70:
        log("  ** DALINIS: Dinamika ismokstama, bet reikia optimizacijos.")
    else:
        log("  * Evoliucija vyksta, bet reikia daugiau kartu.")
    log(f"{'='*70}")
