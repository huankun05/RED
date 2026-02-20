import numpy as np
import RoleEmbedding
import RoleEmbedding_Opt_V5
import RoleEmbedding_Opt_V6
import test, tsne
import time, matplotlib.pyplot as plt
import os, shutil

# --- 全局配置 ---
INPATH, GTPATH = 'topo.txt', 'gt.txt'
DIM, D_GLOBAL = 12, 6
TURNS = 5
TQ_LIST = [1, 2, 3, 4, 5]
EXP_NAME = "RED_Comparison"

# --- 路径系统 ---
BASE_DIR = os.path.join("Experiment_Outputs", EXP_NAME)
DIRS = {k: os.path.join(BASE_DIR, k) for k in ["embeddings", "plots", "results", "temp"]}

def init_env():
    """初始化环境，确保目录洁净"""
    if os.path.exists(BASE_DIR): shutil.rmtree(BASE_DIR)
    for d in DIRS.values(): os.makedirs(d)
    force_clean()

def force_clean():
    """强力收纳：将根目录下所有实验产生的临时文件移入 temp"""
    protected = [INPATH, GTPATH, 'RED_Comparison.py', 'TotalCaller.py', 'test.py', 
                 'RoleEmbedding.py', 'RoleEmbedding_Opt_V5.py', 'RoleEmbedding_Opt_V6.py', 'tsne.py', 'Degree.py', 
                 'diameter.py', 'gt.py', 'README.txt']
    patterns = ('00', 'tmp', 'matrix_', 'orig', 'opt', 'robust', 'combined', 'embedding')
    for f in os.listdir('.'):
        if (f.startswith(patterns) or f.endswith(('.csv', '.txt', '.png', '.eps'))) and f not in protected:
            try:
                dest = os.path.join(DIRS["temp"], f)
                if os.path.exists(dest): os.remove(dest)
                shutil.move(f, dest)
            except: pass

# --- 核心实验 Phase 1: 聚类与效率对比 ---
def run_phase1_clustering_and_time():
    print("\n" + " PHASE 1: Clustering Performance & Efficiency ".center(65, "#"))
    
    # 记录生成时间
    start = time.time()
    p_orig = RoleEmbedding.RoleEmbeddingD(INPATH, os.path.join(DIRS["embeddings"], 'orig'), DIM, 6)[0]
    t_orig = time.time() - start
    
    start = time.time()
    p_V5 = RoleEmbedding_Opt_V5.RED_Optimized(INPATH, os.path.join(DIRS["embeddings"], 'V5'), DIM, 6, tq=3)[0]
    t_V5 = time.time() - start
    
    start = time.time()
    p_opt = RoleEmbedding_Opt_V6.RED_Optimized(INPATH, os.path.join(DIRS["embeddings"], 'opt'), DIM, 6, tq=3)[0]
    t_opt = time.time() - start
    
    scores_cache = {"Original RED": [], "RED V5": [], "Optimized RED": []}
    for i in range(TURNS):
        print(f" -> Clustering Turn {i+1}/{TURNS}...")
        test.test_clustering([p_orig, p_V5, p_opt], GTPATH, turn=f"cmp_{i}")
        s = test.readMatrix(f'00cmp_{i}PyCluster.txt', float)
        scores_cache["Original RED"].append(s[0])
        scores_cache["RED V5"].append(s[1])
        scores_cache["Optimized RED"].append(s[2])
    
    # 生成对齐的合并报表
    orig_avgs = np.mean(scores_cache["Original RED"], axis=0)
    V5_avgs = np.mean(scores_cache["RED V5"], axis=0)
    opt_avgs = np.mean(scores_cache["Optimized RED"], axis=0)
    metrics = ["AMI", "ARI", "V-measure", "Silhouette"]
    
    res_path = os.path.join(DIRS["results"], "RED_Summary_Report.txt")
    sep = "+" + "-"*15 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+"
    
    with open(res_path, "w", encoding='utf-8') as f:
        f.write("【实验 1：聚类性能对比总结】\n")
        f.write(sep + "\n")
        f.write(f"| {'Metric':<13} | {'Original':^10} | {'RED V5':^10} | {'Optimized':^10} | {'V5 Improve':^10} | {'V6 Improve':^10} |\n")
        f.write(sep + "\n")
        print(sep)
        print(f"| {'Metric':<13} | {'Original':^10} | {'RED V5':^10} | {'Optimized':^10} | {'V5 Improve':^10} | {'V6 Improve':^10} |")
        print(sep)
        
        for i, name in enumerate(metrics):
            V5_imp = ((V5_avgs[i] - orig_avgs[i]) / abs(orig_avgs[i]) * 100) if orig_avgs[i] != 0 else 0
            opt_imp = ((opt_avgs[i] - orig_avgs[i]) / abs(orig_avgs[i]) * 100) if orig_avgs[i] != 0 else 0
            line = f"| {name:<13} | {orig_avgs[i]:>10.4f} | {V5_avgs[i]:>10.4f} | {opt_avgs[i]:>10.4f} | {V5_imp:>+9.2f}% | {opt_imp:>+9.2f}% |\n"
            f.write(line); print(line.strip())
        
        f.write(sep + "\n\n")
        print(sep)

        f.write("【实验 2：时间效率对比】\n")
        eff_info = (f"Original Time:  {t_orig:.4f} s\n"
                    f"RED V5 Time:    {t_V5:.4f} s\n"
                    f"Optimized Time: {t_opt:.4f} s\n"
                    f"V5 Speedup:     {t_orig/t_V5:.2f} x\n"
                    f"V6 Speedup:     {t_orig/t_opt:.2f} x\n")
        f.write(eff_info); print(eff_info)
    
    force_clean()
    return p_orig, p_V5, p_opt

# --- 核心实验 Phase 2: 鲁棒性分析 (回归) ---
def run_phase2_robustness():
    print("\n" + " PHASE 2: Robustness Curve Analysis (Big Mirror) ".center(65, "#"))
    ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    orig_accs, V5_accs, opt_accs = [], [], []

    for r in ratios:
        print(f" -> Testing Noise: {int(r*100)}%...")
        combined_topo = os.path.join(DIRS["temp"], f"robust_topo_{r}.txt")
        N = test.combine_topo(INPATH, combined_topo, r)
        
        p_orig = RoleEmbedding.RoleEmbeddingD(combined_topo, os.path.join(DIRS["temp"], 'tmp_orig'), DIM, 6)[0]
        p_V5 = RoleEmbedding_Opt_V5.RED_Optimized(combined_topo, os.path.join(DIRS["temp"], 'tmp_V5'), DIM, 6, tq=3)[0]
        p_opt = RoleEmbedding_Opt_V6.RED_Optimized(combined_topo, os.path.join(DIRS["temp"], 'tmp_opt'), DIM, 6, tq=3)[0]
        
        acc_scores = test.test_role([p_orig, p_V5, p_opt], N, f"rob_{r}")
        orig_accs.append(acc_scores[0][0])
        V5_accs.append(acc_scores[1][0])
        opt_accs.append(acc_scores[2][0])

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(ratios, orig_accs, 'r-o', label='Original RED', linewidth=2)
    plt.plot(ratios, V5_accs, 'g-s', label='RED V5', linewidth=2)
    plt.plot(ratios, opt_accs, 'b-^', label='Optimized RED', linewidth=2)
    plt.xlabel('Noise Edge Ratio'); plt.ylabel('Equivalency Accuracy (Cosine)'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(DIRS["plots"], "Robustness_Comparison.png"))
    
    # 记录鲁棒性数据
    with open(os.path.join(DIRS["results"], "robustness_data.txt"), "w") as f:
        f.write("Noise_Ratio | Original_Acc | RED_V5_Acc | Optimized_Acc\n")
        for i, r in enumerate(ratios):
            f.write(f"{r:<11} | {orig_accs[i]:^12.4f} | {V5_accs[i]:^10.4f} | {opt_accs[i]:^13.4f}\n")
    force_clean()

# --- 核心实验 Phase 3: 参数敏感性 (回归) ---
def run_phase3_sensitivity():
    print("\n" + " PHASE 3: Semiclassical Step (tq) Sensitivity ".center(65, "#"))
    V5_amis, V6_amis = [], []
    for t in TQ_LIST:
        print(f" -> Testing tq={t}...")
        p_V5 = RoleEmbedding_Opt_V5.RED_Optimized(INPATH, os.path.join(DIRS["temp"], f'tq_V5_{t}'), DIM, 6, tq=t)[0]
        p_V6 = RoleEmbedding_Opt_V6.RED_Optimized(INPATH, os.path.join(DIRS["temp"], f'tq_V6_{t}'), DIM, 6, tq=t)[0]
        test.test_clustering([p_V5, p_V6], GTPATH, turn=f"tq_{t}")
        score = test.readMatrix(f'00tq_{t}PyCluster.txt', float)
        V5_amis.append(score[0, 0])
        V6_amis.append(score[1, 0])
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(TQ_LIST))
    width = 0.35
    plt.bar(x - width/2, V5_amis, width, label='RED V5', color='lightgreen', edgecolor='darkgreen')
    plt.bar(x + width/2, V6_amis, width, label='RED V6', color='skyblue', edgecolor='navy')
    plt.xlabel('tq Step'); plt.ylabel('AMI Score'); plt.title('RED: tq Sensitivity Comparison')
    plt.xticks(x, TQ_LIST); plt.legend(); plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(DIRS["plots"], "TQ_Sensitivity.png"))
    force_clean()

# --- 主控逻辑 ---
if __name__ == "__main__":
    init_env()
    
    # 1. 聚类与效率
    p_orig, p_V5, p_opt = run_phase1_clustering_and_time()
    
    # 2. 鲁棒性 (找回)
    run_phase2_robustness()
    
    # 3. 参数敏感性 (找回)
    run_phase3_sensitivity()
    
    # 4. 可视化
    print("\n" + " PHASE 4: Visualizations ".center(65, "#"))
    tsne.HighVisualize(p_orig, GTPATH)
    tsne.HighVisualize(p_V5, GTPATH)
    tsne.HighVisualize(p_opt, GTPATH)
    # 图片收纳
    for img in ["orig.png", "V5.png", "opt.png"]:
        if os.path.exists(img): shutil.move(img, os.path.join(DIRS["plots"], img))
    
    force_clean()
    print(f"\n[FINISH] 实验报告已完整生成。请查看: {DIRS['results']}")
    print(f"[FINISH] 所有分析图表存放在: {DIRS['plots']}")