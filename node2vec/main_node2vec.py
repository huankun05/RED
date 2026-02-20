'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''
import argparse
import numpy as np
import networkx as nx
from . import node2vec
from gensim.models import Word2Vec

from . import rewrite_node2vec

def parse_args(paraconfig):
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default=paraconfig[0],
						help='Input graph path')
	parser.add_argument('--output', nargs='?', default='embedding_node2vec'+str(paraconfig[6])+'.txt',
	                    help='Embeddings path')
	parser.add_argument('--dimensions', type=int, default=paraconfig[1],
	                    help='Number of dimensions.')

	parser.add_argument('--walk-length', type=int, default=paraconfig[2],
	                    help='Length of walk per source. Default is 80.')
	parser.add_argument('--num-walks', type=int, default=paraconfig[3],
	                    help='Number of walks per source. Default is 10.')
	parser.add_argument('--window-size', type=int, default=paraconfig[4],
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--p', type=float, default=paraconfig[5],
	                    help='Return hyperparameter. Default is 1.')
	parser.add_argument('--q', type=float, default=paraconfig[6],
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--iter', default=paraconfig[7], type=int,  #default = 1
                      help='Number of epochs in SGD')


	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')
	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)
	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)
	return parser.parse_args()

def read_graph(args):
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(args,walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, epochs=args.iter)
	model.wv.save_word2vec_format(args.output)

	return

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	nx_G = read_graph(args)
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	learn_embeddings(args,walks)

def main_node2vec(paraconfig,outpathbasis='matrix_node2vec'):
	print('Node2vec starts')
	args = parse_args(paraconfig)
	#print('args.input:',args.input,'\nargs.output:',args.output,'\nargs.dimensions:',args.dimensions,'\nargs.walk_length:',args.walk_length,'\nargs.num_walks:',args.num_walks,'\nargs.window_size:',args.window_size,'\nargs.iter:',args.iter,'\nargs.workers:',args.workers,'\nargs.p:',args.p,'\nargs.q:',args.q,'\nargs.weighted:',args.weighted,'\nargs.unweightedt:',args.unweighted,'\nargs.directed:',args.directed,'\nargs.undirectedt:',args.undirected)
	main(args)
	rewrite_node2vec.rewrite_node2vec(args,outpathbasis)
	print('Node2vec ends\n')


#p,q in [0.25,0.5,1,2,4]
#paraconfig=[inputpath,dimensions,        walk-length,num-walk,w-size,  p,q,   epoch]
#paraconfig=['karate.edgelist',128,       80,10,10,  1,1,    1]  #default
if __name__ == "__main__":
	#paraconfig = ['topo.txt', 10, 80, 10, 10, 1, 1, 1]
	#main_node2vec(paraconfig)
	print('Do not use node2vec as main()')