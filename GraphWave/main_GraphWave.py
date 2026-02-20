import pandas as pd
import networkx as nx
from texttable import Texttable
import argparse
from .spectral_machinery import WaveletMachine
from .rewrites import *

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    tab = Texttable() 
    tab.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    #print(tab.draw())

def read_graph(settings):
    """
    Reading the edge list from the path and returning the networkx graph object.
    :param path: Path to the edge list.
    :return graph: Graph from edge list.
    """
    #print(settings.edgelist_input)
    if settings.edgelist_input:
        #print('using edgelist input')
        graph = nx.read_edgelist(settings.input)
    else:
        edge_list = pd.read_csv(settings.input).values.tolist()
        graph = nx.from_edgelist(edge_list)
        # 使用nx.selfloop_edges函数而不是graph.selfloop_edges方法
        graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


# 只修改了输入输出路径与dimension，其余均是default
def parameter_parser(dim):

    """
    A method to parse up command line parameters.
    """

    parser = argparse.ArgumentParser(description = "Run WaveletMachine.")

    parser.add_argument("--mechanism",
                        nargs = "?",
                        default = "exact",
	                help = "Eigenvalue calculation method. Default is exact.")

    parser.add_argument("--input",
                        nargs = "?",
                        default = "./topoedges.csv",#"./data/food_edges.csv",
	                help = "Path to the graph edges. Default is food_edges.csv.")

    parser.add_argument("--output",
                        nargs = "?",
                        default = "./embedding.csv",#"./output/embedding.csv",
	                help = "Path to the structural embedding. Default is embedding.csv.")

    parser.add_argument("--heat-coefficient",
                        type = float,
                        default = 1000.0,
	                help = "Heat kernel exponent. Default is 1000.0.")

    parser.add_argument("--sample-number",
                        type = int,
                        default = dim,#50,
	                help = "Number of characteristic function sample points. Default is 50.")

    parser.add_argument("--approximation",
                        type = int,
                        default = 100,
	                help = "Number of Chebyshev approximation. Default is 100.")

    parser.add_argument("--step-size",
                        type = int,
                        default = 20,
	                help = "Number of steps. Default is 20.")

    parser.add_argument("--switch",
                        type = int,
                        default = 100,
	                help = "Number of dimensions. Default is 100.")

    parser.add_argument("--node-label-type",
                        type=str,
                        default= "int",
                        help = "Used for sorting index of output embedding. One of 'int', 'string', or 'float'. Default is 'int'")

    parser.add_argument("--edgelist-input",
                        action='store_true',
                        help="Use NetworkX's edgelist format for input instead of CSV. Default is False")

    return parser.parse_args()

def main_GraphWave(inpath,dim,outpath):
    # 转写输入
    writeCsvGW(inpath, outpath='topoedges.csv')

    settings = parameter_parser(dim)
    tab_printer(settings)
    G = read_graph(settings)
    N=len(nx.nodes(G))
    machine = WaveletMachine(G,settings)
    machine.create_embedding()
    machine.transform_and_save_embedding()

    # 转写输出
    rewrite(outpath,dim,N)


if __name__ == "__main__":
    #main_GraphWave(inpath='topo.txt', dim=2, outpath='matrix_GraphWave.txt')
    rewrite(outpath='matrix_GraphWave.txt', dim=2,N=30)
    a=readMatrix('matrix_GraphWave.txt',type=float)
    print(a)
