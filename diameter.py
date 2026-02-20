import networkx as nx

# 从EL构建graph，计算diameter
def diameter(inpath):
    G = nx.Graph()
    file = open(inpath)
    for line in file:
        lines = line.split()
        head, tail = int(lines[0]), int(lines[1])
        G.add_edge(head, tail)
    file.close()
    N = len(G.nodes())
    

    #收集连通子图
    # 使用nx.connected_components替代nx.connected_component_subgraphs，以适配较新版本的NetworkX库
    connected=(G.subgraph(c) for c in nx.connected_components(G))
    ds=[]
    for i in connected:
        ds.append(nx.diameter(i))

    #d=连通子图各d的max
    #print('D for Subgraphs are',ds)
    #print(max(ds))
    return (max(ds))
    


#print(diameter())



# import networkx as nx
# import random

# def get_component_diameter(sub_G, samples=5):
#     """
#     启发式计算单个连通分量的直径
#     """
#     nodes = list(sub_G.nodes())
#     if len(nodes) <= samples:
#         return nx.diameter(sub_G)
    
#     max_d = 0
#     # 随机选几个种子节点进行采样
#     seeds = random.sample(nodes, samples)
    
#     for seed in seeds:
#         # 1. 从种子节点出发，找到距离它最远的节点 u
#         lengths = nx.single_source_shortest_path_length(sub_G, seed)
#         u = max(lengths, key=lengths.get)
        
#         # 2. 从 u 出发，找到距离 u 最远的距离（这通常就是直径的极佳近似）
#         u_lengths = nx.single_source_shortest_path_length(sub_G, u)
#         dist = max(u_lengths.values())
#         if dist > max_d:
#             max_d = dist
            
#     return max_d

# def diameter(inpath):
#     G = nx.Graph()
#     with open(inpath, 'r') as f:
#         for line in f:
#             parts = line.split()
#             if len(parts) < 2: continue
#             G.add_edge(int(parts[0]), int(parts[1]))
    
#     if len(G) == 0: return 0
    
#     # 找到所有连通分量
#     components = [G.subgraph(c) for c in nx.connected_components(G)]
    
#     # 计算每个分量的直径（采样法）并取最大值
#     ds = [get_component_diameter(comp) for comp in components]
    
#     return max(ds) if ds else 0