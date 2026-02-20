import networkx as nx
import numpy as np
import random
import os

# =========================================================
# 升级版参数调节区：挑战算法极限
# =========================================================
CONFIG = {
    "n_stars": 8,
    "star_size_range": (5, 12),     # 星型大小随机化
    
    "n_cliques": 8,
    "clique_size_range": (4, 10),   # 团大小随机化 (测试非度数依赖)
    
    "n_cycles": 8,
    "cycle_size_range": (6, 15),    # 环大小随机化
    
    "n_chains": 8,                  # 新增：链状结构 (测试度数为2的歧义性)
    "chain_size_range": (6, 10),
    
    "n_bridges": 10,                # 显式增加桥接节点数量
    
    "noise_ratio": 0.05,            # 噪声边比例 (总边数的 5%)
    "connect_all": True             # 确保全局连通
}

# 角色标签定义 (Ground Truth)
ROLE_LABELS = {
    "STAR_HUB": 1,
    "STAR_LEAF": 2,
    "CLIQUE_MEMBER": 3,
    "CYCLE_MEMBER": 4,
    "BRIDGE_NODE": 5,
    "CHAIN_MEMBER": 6
}
# =========================================================

def generate_comprehensive_data():
    G = nx.Graph()
    labels = {}
    motif_entry_nodes = [] # 记录每个Motif的代表节点，用于后续连接

    def add_motif(motif_type):
        nonlocal G
        current_n = G.number_of_nodes()
        
        if motif_type == "clique":
            size = random.randint(*CONFIG["clique_size_range"])
            new_m = nx.complete_graph(size)
            role = ROLE_LABELS["CLIQUE_MEMBER"]
            m_labels = {n: role for n in range(size)}
            
        elif motif_type == "star":
            size = random.randint(*CONFIG["star_size_range"])
            new_m = nx.star_graph(size - 1)
            m_labels = {0: ROLE_LABELS["STAR_HUB"]}
            for i in range(1, size): m_labels[i] = ROLE_LABELS["STAR_LEAF"]
            
        elif motif_type == "cycle":
            size = random.randint(*CONFIG["cycle_size_range"])
            new_m = nx.cycle_graph(size)
            m_labels = {n: ROLE_LABELS["CYCLE_MEMBER"] for n in range(size)}
            
        elif motif_type == "chain":
            size = random.randint(*CONFIG["chain_size_range"])
            new_m = nx.path_graph(size)
            m_labels = {n: ROLE_LABELS["CHAIN_MEMBER"] for n in range(size)}
        
        # 映射到全局ID (从1开始)
        mapping = {n: n + current_n + 1 for n in new_m.nodes()}
        new_m = nx.relabel_nodes(new_m, mapping)
        
        G.add_edges_from(new_m.edges())
        for old_id, global_id in mapping.items():
            labels[global_id] = m_labels[old_id]
        
        motif_entry_nodes.append(list(mapping.values())[0])

    # 1. 生成各种类型的 Motifs
    for _ in range(CONFIG["n_stars"]): add_motif("star")
    for _ in range(CONFIG["n_cliques"]): add_motif("clique")
    for _ in range(CONFIG["n_cycles"]): add_motif("cycle")
    for _ in range(CONFIG["n_chains"]): add_motif("chain")

    # 2. 添加显式桥接节点 (Bridge Nodes)
    current_n = G.number_of_nodes()
    bridge_nodes = range(current_n + 1, current_n + CONFIG["n_bridges"] + 1)
    for b in bridge_nodes:
        labels[b] = ROLE_LABELS["BRIDGE_NODE"]
        # 每个桥接节点随机连接 2-4 个已有的 Motif 代表点
        targets = random.sample(motif_entry_nodes, random.randint(2, 4))
        for t in targets:
            G.add_edge(b, t)

    # 3. 强制全局连通 (Spanning Tree over motifs)
    if CONFIG["connect_all"]:
        all_entry = motif_entry_nodes + list(bridge_nodes)
        for i in range(len(all_entry) - 1):
            if not nx.has_path(G, all_entry[i], all_entry[i+1]):
                G.add_edge(all_entry[i], all_entry[i+1])

    # 4. 添加随机噪声边 (按比例)
    num_noise = int(G.number_of_edges() * CONFIG["noise_ratio"])
    nodes_list = list(G.nodes())
    added = 0
    while added < num_noise:
        u, v = random.sample(nodes_list, 2)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
            added += 1

    # 5. 导出 topo.txt
    all_nodes = sorted(list(G.nodes()))
    with open('topo.txt', 'w') as f:
        for u, v in G.edges():
            f.write(f"{u}\t{v}\n")

    # 6. 导出 gt.txt (包含用于 Robustness 测试的镜像节点)
    n_total = len(all_nodes)
    # 建立连续索引映射 (防止中间ID跳跃)
    id_map = {node: i+1 for i, node in enumerate(all_nodes)}
    
    # 重新写 topo.txt 使用连续ID (1 to N)
    with open('topo.txt', 'w') as f:
        for u, v in G.edges():
            f.write(f"{id_map[u]}\t{id_map[v]}\n")

    extended_labels = []
    # 原始节点
    for node in all_nodes:
        extended_labels.append(f"{id_map[node]} {labels[node]}")
    # 镜像节点 (ID偏移 n_total)
    for node in all_nodes:
        extended_labels.append(f"{id_map[node] + n_total} {labels[node]}")

    with open('gt.txt', 'w') as f:
        f.write('\n'.join(extended_labels))

    print(f"--- 挑战级测试数据生成完毕 ---")
    print(f"节点总数: {n_total} (含镜像 {n_total*2})")
    print(f"边总数: {G.number_of_edges()}")
    print(f"角色分布: {list(ROLE_LABELS.keys())}")
    print(f"噪声比例: {CONFIG['noise_ratio']*100}% ({num_noise}条边)")
    print(f"歧义性关键点: Chain vs Cycle (度数均为2)")

if __name__ == "__main__":
    generate_comprehensive_data()