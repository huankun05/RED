import sys, warnings, time, os, shutil
import numpy as np
import matplotlib.pyplot as plt
from GraphRoleMain import main_rolx
from GraphWave import main_GraphWave
from Role2vec import main_Role2vec
from node2vec import main_node2vec
import Degree, RoleEmbedding, RoleEmbedding_Opt_V5, test, tsne

warnings.filterwarnings('ignore')

# --- 实验全局配置 ---
EXP_NAME = "TotalCaller"
DIM, TURNS = 12, 5  
INPATH, GTPATH = 'topo.txt', 'gt.txt'
TQ_LIST = [1, 2, 3, 4, 5]

# --- 自动目录管理系统 ---
BASE_DIR = os.path.join("Experiment_Outputs", EXP_NAME)
DIRS = {
    "embeddings": os.path.join(BASE_DIR, "embeddings"),
    "plots": os.path.join(BASE_DIR, "plots"),
    "results": os.path.join(BASE_DIR, "results"),
    "temp": os.path.join(BASE_DIR, "temp")
}

def init_env():
    if os.path.exists(BASE_DIR): shutil.rmtree(BASE_DIR)
    for d in DIRS.values(): os.makedirs(d)
    force_clean_all()

def force_clean_all():
    """彻底收纳：清理根目录下所有实验产生的临时文件"""
    protected = [INPATH, GTPATH, 'RED_Comparison.py', 'TotalCaller.py', 'test.py', 
                 'RoleEmbedding.py', 'RoleEmbedding_Opt_V5.py', 'tsne.py', 'Degree.py', 
                 'diameter.py', 'gt.py', 'README.txt']
    patterns = ('00', 'tmp', 'matrix_', 'base_', 'main_', 'p1_', 'r_', 'rob_', 'tq_', 'combined', 'embedding')
    for f in os.listdir('.'):
        if (f.startswith(patterns) or f.endswith(('.csv', '.txt', '.png'))) and f not in protected:
            try:
                dest = os.path.join(DIRS["temp"], f)
                if os.path.exists(dest): os.remove(dest)
                shutil.move(f, dest)
            except: pass

def run_single_algo_timed(name, dim, inpath, out_prefix):
    """封装单个算法运行并计时"""
    start = time.time()
    path = ""
    if name == 'Degree':
        path = out_prefix + "Degree.txt"
        Degree.Degree(inpath, path, dim)
    elif name == 'Node2vec':
        main_node2vec.main_node2vec([inpath, dim, 80, 10, 10, 1, 2, 10], out_prefix + "node2vec")
        path = out_prefix + "node2vec2.txt" if os.path.exists(out_prefix + "node2vec2.txt") else out_prefix + "node2vec.txt"
    elif name == 'Role2vec':
        path = out_prefix + "Role2vec.txt"
        main_Role2vec.main_Role2vec(inpath, path, dim)
    elif name == 'GraphWave':
        path = out_prefix + "GraphWave.txt"
        main_GraphWave.main_GraphWave(inpath, dim, path)
    elif name == 'OriginalRED':
        path = RoleEmbedding.RoleEmbeddingD(inpath, out_prefix + "orig", dim, 6)[0]
    elif name == 'OptimizedRED':
        path = RoleEmbedding_Opt_V5.RED_Optimized(inpath, out_prefix + "opt", dim, 6, tq=1)[0]
    
    elapsed = time.time() - start
    return path, elapsed

# --- PHASE 1: 聚类基准测试 ---
def phase1_benchmark():
    print("\n" + " PHASE 1: Clustering & Efficiency Benchmark ".center(65, "#"))
    algos = ['Degree', 'Node2vec', 'Role2vec', 'GraphWave', 'OriginalRED', 'OptimizedRED']
    results = {n: [] for n in algos}
    times = {n: [] for n in algos}
    
    for i in range(TURNS):
        print(f" -> Turn {i+1}/{TURNS}")
        current_paths = []
        for name in algos:
            p, t = run_single_algo_timed(name, DIM, INPATH, os.path.join(DIRS["temp"], f"p1_{i}_"))
            current_paths.append(p)
            times[name].append(t)
        
        test.test_clustering(current_paths, GTPATH, turn="p1")
        s = test.readMatrix('00p1PyCluster.txt', float)
        for idx, name in enumerate(algos):
            results[name].append(s[idx])

    # 对齐表格生成
    sep = "+" + "-"*17 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*12 + "+"
    header = f"| {'Method':<15} | {'AMI':^8} | {'ARI':^8} | {'V-meas':^8} | {'Sil':^8} | {'Time(s)':^10} |"
    
    res_path = os.path.join(DIRS["results"], "phase1_clustering_report.txt")
    with open(res_path, "w", encoding='utf-8') as f:
        f.write("【实验描述】：对比不同算法在原始拓扑上的聚类性能及计算效率。重复运行 5 轮取均值。\n")
        f.write("【指标含义】：AMI/ARI/V-meas越高代表越接近真实角色；Sil代表簇的紧凑程度。\n\n")
        f.write(sep + "\n" + header + "\n" + sep + "\n")
        print("\n" + sep + "\n" + header + "\n" + sep)
        for name in algos:
            avg_s = np.mean(results[name], axis=0)
            avg_t = np.mean(times[name])
            row = f"| {name:<15} | {avg_s[0]:>8.4f} | {avg_s[1]:>8.4f} | {avg_s[2]:>8.4f} | {avg_s[3]:>8.4f} | {avg_t:>10.4f} |"
            print(row); f.write(row + "\n")
        print(sep); f.write(sep + "\n")
    force_clean_all()

# --- PHASE 2: 鲁棒性分析 ---
def phase2_robustness():
    print("\n" + " PHASE 2: Robustness Analysis (Mirror Test) ".center(65, "#"))
    ratios = [0.1, 0.5, 0.9, 1.3]
    algos = ['Degree', 'Node2vec', 'Role2vec', 'GraphWave', 'OriginalRED', 'OptimizedRED']
    acc_data = {n: [] for n in algos}
    time_data = {n: [] for n in algos}

    for r in ratios:
        print(f" -> Noise Level: {int(r*100)}%")
        c_path = os.path.join(DIRS["temp"], f"robust_{r}.txt")
        N = test.combine_topo(INPATH, c_path, r)
        
        paths = []
        for name in algos:
            p, t = run_single_algo_timed(name, DIM, c_path, os.path.join(DIRS["temp"], f"r_{r}_"))
            paths.append(p)
            time_data[name].append(t)
        
        scores = test.test_role(paths, N, "r")
        for idx, name in enumerate(algos):
            acc_data[name].append(scores[idx][0])

    # 对齐表格生成
    res_path = os.path.join(DIRS["results"], "phase2_robustness_report.txt")
    sep = "+" + "-"*17 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*12 + "+"
    header = f"| {'Method':<15} | {'Acc@10%':^8} | {'Acc@50%':^8} | {'Acc@90%':^8} | {'Acc@130%':^8} | {'AvgTime':^10} |"
    
    with open(res_path, "w", encoding='utf-8') as f:
        f.write("【实验描述】：分析各算法在镜像网络中的识别准确率。随着噪声比例增加，准确率越高说明算法鲁棒性越强。\n")
        f.write("【指标含义】：Acc即准确识别镜像孪生节点的比例。\n\n")
        f.write(sep + "\n" + header + "\n" + sep + "\n")
        for name in algos:
            row = f"| {name:<15} | {acc_data[name][0]:>8.3f} | {acc_data[name][1]:>8.3f} | {acc_data[name][2]:>8.3f} | {acc_data[name][3]:>8.3f} | {np.mean(time_data[name]):>10.4f} |"
            f.write(row + "\n")
        f.write(sep + "\n")
    
    plt.figure(figsize=(10, 6))
    for name in algos: plt.plot(ratios, acc_data[name], marker='o', label=name, linewidth=2)
    plt.xlabel('Noise Edge Ratio'); plt.ylabel('Cosine Accuracy'); plt.legend(bbox_to_anchor=(1.05, 1)); plt.grid(True, linestyle='--'); plt.tight_layout()
    plt.savefig(os.path.join(DIRS["plots"], "Robustness_Benchmark.png"))
    force_clean_all()

# --- PHASE 3: 参数敏感性 ---
def phase3_sensitivity():
    print("\n" + " PHASE 3: Optimized RED Parameter Sensitivity ".center(65, "#"))
    amis, times = [], []
    for t in TQ_LIST:
        print(f" -> Testing tq={t}")
        start = time.time()
        p = RoleEmbedding_Opt_V5.RED_Optimized(INPATH, os.path.join(DIRS["temp"], f"tq_{t}"), DIM, 6, tq=t)[0]
        times.append(time.time() - start)
        test.test_clustering([p], GTPATH, turn="tq")
        amis.append(test.readMatrix('00tqPyCluster.txt', float)[0,0])

    res_path = os.path.join(DIRS["results"], "phase3_sensitivity_report.txt")
    sep = "+" + "-"*12 + "+" + "-"*15 + "+" + "-"*15 + "+"
    header = f"| {'tq Step':^10} | {'AMI Score':^13} | {'Exec Time(s)':^13} |"
    
    with open(res_path, "w", encoding='utf-8') as f:
        f.write("【实验描述】：研究 Optimized RED 的核心参数 tq (半经典量子步长) 对聚类效果 AMI 的影响。\n")
        f.write(sep + "\n" + header + "\n" + sep + "\n")
        for i, t in enumerate(TQ_LIST):
            f.write(f"| {t:^10} | {amis[i]:^13.4f} | {times[i]:^13.4f} |\n")
        f.write(sep + "\n")
    
    plt.figure(figsize=(8, 5))
    plt.bar(list(map(str, TQ_LIST)), amis, color='skyblue', edgecolor='navy')
    plt.title("Optimized RED: tq Step vs AMI Score"); plt.xlabel("Quantum Step tq"); plt.ylabel("AMI Score"); plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(DIRS["plots"], "TQ_Sensitivity.png"))
    force_clean_all()

# --- PHASE 4: 可视化对比 ---
def phase4_visualize():
    print("\n" + " PHASE 4: High-Dimensional Visualization ".center(65, "#"))
    f_orig, t1 = run_single_algo_timed('OriginalRED', DIM, INPATH, os.path.join(DIRS["embeddings"], "final_"))
    f_opt, t2 = run_single_algo_timed('OptimizedRED', DIM, INPATH, os.path.join(DIRS["embeddings"], "final_"))
    
    print(" -> Generating t-SNE scatter plots...")
    tsne.HighVisualize(f_orig, GTPATH)
    tsne.HighVisualize(f_opt, GTPATH)
    
    res_path = os.path.join(DIRS["results"], "phase4_viz_efficiency.txt")
    with open(res_path, "w", encoding='utf-8') as f:
        f.write("【实验描述】：记录最终用于降维可视化的 Embedding 生成耗时。\n\n")
        f.write(f"Original RED 生成耗时:  {t1:.4f} s\n")
        f.write(f"Optimized RED 生成耗时: {t2:.4f} s\n")

    for f in ["final_orig.png", "final_opt.png"]:
        if os.path.exists(f): shutil.move(f, os.path.join(DIRS["plots"], f))
    force_clean_all()

if __name__ == "__main__":
    init_env()
    phase1_benchmark()
    phase2_robustness()
    phase3_sensitivity()
    phase4_visualize()
    print(f"\n[DONE] 所有实验报表已整齐对齐，并存入: {DIRS['results']}")
    print(f"[DONE] 所有分析图表已存入: {DIRS['plots']}")