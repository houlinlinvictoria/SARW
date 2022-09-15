# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:47:49 2020

@author: Lin
"""
from graphsaint.globals import *
import numpy as np
import scipy.sparse
import time
import math
import pdb
from math import ceil
import graphsaint.cython_sampler as cy
import random
import sys
import os


class GraphSampler:
    """
    This is the sampler super-class. Any GraphSAINT sampler is supposed to perform
    the following meta-steps:
     1. [optional] Preprocessing: e.g., for edge sampler, we need to calculate the
            sampling probability for each edge in the training graph. This is to be
            performed only once per phase (or, once throughout the whole training,
            since in most cases, training only consists of a single phase. see
            ../train_config/README.md for definition of a phase).
            ==> Need to override the `preproc()` in sub-class
     2. Parallel sampling: launch a batch of graph samplers in parallel and sample
            subgraphs independently. For efficiency, the actual sampling operation
            happen in cython. And the classes here is mainly just a wrapper.
            ==> Need to set self.cy_sampler to the appropriate cython sampler
              in `__init__()` of the sampler sub-class
     3. Post-processing: upon getting the sampled subgraphs, we need to prepare the
            appropriate information (e.g., subgraph adj with renamed indices) to
            enable the PyTorch trainer. Also, we need to do data conversion from C++
            to Python (or, mostly numpy). Post-processing is handled within the
            cython sampling file (./cython_sampler.pyx)

    Pseudo-code for the four proposed sampling algorithms (Node, Edge, RandomWalk,
    MultiDimRandomWalk) can be found in Appendix, Algo 2 of the GraphSAINT paper.

    Lastly, if you don't bother with writing samplers in cython, you can still code
    the sampler subclass in pure python. In this case, we have provided a function
    `_helper_extract_subgraph` for API consistency between python and cython. An
    example sampler in pure python is provided as `NodeSamplingVanillaPython` at the
    bottom of this file.
    """
    def __init__(self, adj_train, node_train, size_subgraph, args_preproc):
        """
        Inputs:
            adj_train       scipy sparse CSR matrix of the training graph
            node_train      1D np array storing the indices of the training nodes
            size_subgraph   int, the (estimated) number of nodes in the subgraph
            args_preproc    dict, addition arguments needed for pre-processing

        Outputs:
            None
        """
        self.adj_train = adj_train
        self.node_train = np.unique(node_train).astype(np.int32)
        # size in terms of number of vertices in subgraph
        self.size_subgraph = size_subgraph
        self.name_sampler = 'None'
        self.node_subgraph = None
        self.preproc(**args_preproc)

    def preproc(self, **kwargs):
        pass

    def par_sample(self, stage, **kwargs):
        return self.cy_sampler.par_sample()

    def _helper_extract_subgraph(self, node_ids):
        """
        ONLY used for serial Python sampler (NOT for the parallel cython sampler).
        Return adj of node-induced subgraph and other corresponding data struct.

        Inputs:
            node_ids        1D np array, each element is the ID in the original
                            training graph.
        Outputs:
            indptr          np array, indptr of the subg adj CSR
            indices         np array, indices of the subg adj CSR
            data            np array, data of the subg adj CSR. Since we have aggregator
                            normalization, we can simply set all data values to be 1
            subg_nodes      np array, i-th element stores the node ID of the original graph
                            for the i-th node in the subgraph. Used to index the full feats
                            and label matrices.
            subg_edge_index np array, i-th element stores the edge ID of the original graph
                            for the i-th edge in the subgraph. Used to index the full array
                            of aggregation normalization.
        """

        node_ids = np.unique(node_ids)
        node_ids.sort()
        orig2subg = {n: i for i, n in enumerate(node_ids)}
        n = node_ids.size
        indptr = np.zeros(node_ids.size + 1)
        indices = []
        subg_edge_index = []
        subg_nodes = node_ids   #去重&升序
        for nid in node_ids:
            idx_s, idx_e = self.adj_train.indptr[nid], self.adj_train.indptr[nid + 1]
            neighs = self.adj_train.indices[idx_s : idx_e]
            for i_n, n in enumerate(neighs):
                if n in orig2subg:
                    indices.append(orig2subg[n])
                    indptr[orig2subg[nid] + 1] += 1
                    subg_edge_index.append(idx_s + i_n)
        indptr = indptr.cumsum().astype(np.int64)
        indices = np.array(indices)
        subg_edge_index = np.array(subg_edge_index)
        data = np.ones(indices.size)
        assert indptr[-1] == indices.size == subg_edge_index.size
     
        '''
        sub_degree = len(data)
        node_num = len(indptr)-1
        average_degree = sub_degree/node_num
        tain_average_degree = 2.44414
        RE = abs(average_degree-tain_average_degree)/tain_average_degree
        
        #print("sub_degree",sub_degree)
        #print("subnode_num:",node_num)
        #print("average_degree",average_degree)
        #print("relative erro",RE)
        print((node_num,RE))
        #estimated_degree = sub_degree/node_num
        
        
        #print each subgraph node id and save
        #print("rw subgraph node id:", len(subg_nodes))
        
        with open('rw_subgraph_node_id.txt', 'a') as f:
            for i in range(len(subg_nodes)):
                f.write(str(subg_nodes[i])+',')
            f.write('\n')  
        #path = os.path.abspath('subgraph_node_id.txt')
        #print('path:', path)
        '''
        
        return indptr, indices, data, subg_nodes, subg_edge_index


# --------------------------------------------------------------------
# [BELOW] python wrapper for parallel samplers implemented with Cython
# --------------------------------------------------------------------

class rw_sampling(GraphSampler):
    """
    The sampler performs unbiased random walk, by following the steps:
     1. Randomly pick `size_root` number of root nodes from all training nodes;
     2. Perform length `size_depth` random walk from the roots. The current node
            expands the next hop by selecting one of the neighbors uniformly
            at random;
     3. Generate node-induced subgraph from the nodes touched by the random walk.
    """
    def __init__(self, adj_train, node_train, size_subgraph, size_root, size_depth):
        """
        Inputs:
            adj_train       see super-class
            node_train      see super-class
            size_subgraph   see super-class
            size_root       int, number of root nodes (i.e., number of walkers)
            size_depth      int, number of hops to take by each walker

        Outputs:
            None
        """
        self.size_root = size_root
        self.size_depth = size_depth
        size_subgraph = size_root * size_depth
        super().__init__(adj_train, node_train, size_subgraph, {})
#         print('adj_train:',adj_train)
#         print('adj_train.data:',adj_train.data)
#         print('adj_train.indptr:',adj_train.indptr)
#         print('adj_train.indices:',adj_train.indices)
        self.cy_sampler = cy.RW(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            NUM_PAR_SAMPLER,
            SAMPLES_PER_PROC,
            self.size_root,
            self.size_depth
        )

    def preproc(self, **kwargs):
        pass


class edge_sampling(GraphSampler):
    def __init__(self,adj_train,node_train,num_edges_subgraph):
        """
        The sampler picks edges from the training graph independently, following
        a pre-computed edge probability distribution. i.e.,
            p_{u,v} \\propto 1 / deg_u + 1 / deg_v
        Such prob. dist. is derived to minimize the variance of the minibatch
        estimator (see Thm 3.2 of the GraphSAINT paper).
        """
        self.num_edges_subgraph = num_edges_subgraph
        # num subgraph nodes may not be num_edges_subgraph * 2 in many cases,
        # but it is not too important to have an accurate estimation of subgraph
        # size. So it's probably just fine to use this number.
        self.size_subgraph = num_edges_subgraph * 2
        self.deg_train = np.array(adj_train.sum(1)).flatten()
        self.adj_train_norm = scipy.sparse.dia_matrix((1 / self.deg_train, 0), shape=adj_train.shape).dot(adj_train)
        super().__init__(adj_train, node_train, self.size_subgraph, {})
        self.cy_sampler = cy.Edge2(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            NUM_PAR_SAMPLER,
            SAMPLES_PER_PROC,
            self.edge_prob_tri.row,
            self.edge_prob_tri.col,
            self.edge_prob_tri.data.cumsum(),
            self.num_edges_subgraph,
        )

    def preproc(self,**kwargs):
        """
        Compute the edge probability distribution p_{u,v}.
        """
        self.edge_prob = scipy.sparse.csr_matrix(
            (
                np.zeros(self.adj_train.size),
                self.adj_train.indices,
                self.adj_train.indptr
            ),
            shape=self.adj_train.shape,
        )
        self.edge_prob.data[:] = self.adj_train_norm.data[:]
        _adj_trans = scipy.sparse.csr_matrix.tocsc(self.adj_train_norm)
        self.edge_prob.data += _adj_trans.data      # P_e \propto a_{u,v} + a_{v,u}
        self.edge_prob.data *= 2 * self.num_edges_subgraph / self.edge_prob.data.sum()
        # now edge_prob is a symmetric matrix, we only keep the
        # upper triangle part, since adj is assumed to be undirected.
        self.edge_prob_tri = scipy.sparse.triu(self.edge_prob).astype(np.float32)  # NOTE: in coo format


class mrw_sampling(GraphSampler):
    """
    A variant of the random walk sampler. The multi-dimensional random walk sampler
    is proposed in https://www.cs.purdue.edu/homes/ribeirob/pdf/ribeiro_imc2010.pdf

    Fast implementation of the sampler is proposed in https://arxiv.org/abs/1810.11899
    """
    def __init__(self, adj_train, node_train, size_subgraph, size_frontier, max_deg=10000):
        """
        Inputs:
            adj_train       see super-class
            node_train      see super-class
            size_subgraph   see super-class
            size_frontier   int, size of the frontier during sampling process. The
                            size of the frontier is fixed during sampling.
            max_deg         int, the sampler picks iteratively pick a node from the
                            frontier by probability proportional to node degree. If
                            we specify the `max_deg`, we are essentially bounding the
                            probability of picking any frontier node. This may help
                            with improving sampling quality for skewed graphs.

        Outputs:
            None
        """
        self.p_dist = None
        super().__init__(adj_train, node_train, size_subgraph, {})
        self.size_frontier = size_frontier
        self.deg_train = np.bincount(self.adj_train.nonzero()[0])
        self.name_sampler = 'MRW'
        self.max_deg = int(max_deg)
        self.cy_sampler = cy.MRW(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            NUM_PAR_SAMPLER,
            SAMPLES_PER_PROC,
            self.p_dist,
            self.max_deg,
            self.size_frontier,
            self.size_subgraph
        )

    def preproc(self,**kwargs):
        _adj_hop = self.adj_train
        self.p_dist = np.array(
            [
                _adj_hop.data[_adj_hop.indptr[v] : _adj_hop.indptr[v + 1]].sum()
                for v in range(_adj_hop.shape[0])
            ],
            dtype=np.int32,
        )


class node_sampling(GraphSampler):
    """
    Independently pick some nodes from the full training graph, based on
    pre-computed node probability distribution. The prob. dist. follows
    Sec 3.4 of the GraphSAINT paper. For detailed derivation, see FastGCN
    (https://arxiv.org/abs/1801.10247).
    """
    def __init__(self, adj_train, node_train, size_subgraph):
        """
        Inputs:
            adj_train       see super-class
            node_train      see super-class
            size_subgraph   see super-class

        Outputs:
            None
        """
        self.p_dist = np.zeros(len(node_train))
        super().__init__(adj_train, node_train, size_subgraph, {})
        self.cy_sampler = cy.Node(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            NUM_PAR_SAMPLER,
            SAMPLES_PER_PROC,
            self.p_dist,
            self.size_subgraph,
        )

    def preproc(self, **kwargs):
        """
        Node probability distribution is derived in https://arxiv.org/abs/1801.10247
        """
        _p_dist = np.array(
            [
                self.adj_train.data[
                    self.adj_train.indptr[v] : self.adj_train.indptr[v + 1]
                ].sum()
                for v in self.node_train
            ],
            dtype=np.int64,
        )
        self.p_dist = _p_dist.cumsum()
        if self.p_dist[-1] > 2**31 - 1:
            print('warning: total deg exceeds 2**31')
            self.p_dist = self.p_dist.astype(np.float64)
            self.p_dist /= self.p_dist[-1] / (2**31 - 1)
        self.p_dist = self.p_dist.astype(np.int32)


class full_batch_sampling(GraphSampler):
    """
    Strictly speaking, this is not a sampler. It simply returns the full adj
    matrix of the training graph. This can serve as a baseline to compare
    full-batch vs. minibatch performance.

    Therefore, the size_subgraph argument is not used here.
    """
    def __init__(self, adj_train, node_train, size_subgraph):
        super().__init__(adj_train, node_train, size_subgraph, {})
        self.cy_sampler = cy.FullBatch(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            NUM_PAR_SAMPLER,
            SAMPLES_PER_PROC,
        )


# --------------------------------------------
# [BELOW] Example sampler based on pure python
# --------------------------------------------

class NodeSamplingVanillaPython(GraphSampler):
    """
    This class is just to showcase how you can write the graph sampler in pure python.

    The simplest and most basic sampler: just pick nodes uniformly at random and return the
    node-induced subgraph.
    """
    def __init__(self, adj_train, node_train, size_subgraph):
        super().__init__(adj_train, node_train, size_subgraph, {})

    def par_sample(self, stage, **kwargs):        
        node_ids = np.random.choice(self.node_train, self.size_subgraph)
        ret = self._helper_extract_subgraph(node_ids)
        ret = list(ret)
        for i in range(len(ret)):
            ret[i] = [ret[i]]
        return ret

    def preproc(self):
        pass
    
class CNARW_SamplingVanillaPython(GraphSampler):
    """
    This class is to use cnarw to define sampler.
    """
    def __init__(self, adj_train, node_train, size_subgraph, size_root, size_depth):
        self.size_root = size_root
        self.size_depth = size_depth
        size_subgraph = size_root * size_depth
        super().__init__(adj_train, node_train, size_subgraph, {})
        
        
    '''
    在minibatch中每个sampler都将调用par_sample:用于将采样算法取出的node
    转化成可以输入到模型中的形式。
    '''
    def node_nbhd(self,v,indices,indptr):
        '''
        给定一个节点v，该函数返回其所有邻居节点以及节点v的deg
        '''
        #all_degree = indptr[-1]
        #print("all_degree:",all_degree)
        
        if indptr[v+1] - indptr[v]>0:
            deg_v = indptr[v+1] - indptr[v]
            root_nbhd = []
            u0 = indices[indptr[v]]
            root_nbhd.append(u0)
            r = 1
            while r<deg_v:
                u1 = indices[indptr[v+1]-deg_v+r]
                root_nbhd.append(u1)
                r = r + 1
            assert deg_v==len(root_nbhd)
            return deg_v,root_nbhd
        else:
            return None,None

        
    def par_sample(self, stage, **kwargs):
        iroot = 0
        root = np.random.choice(self.node_train, self.size_root) #获取root节点
        node_ids = []
        while iroot < self.size_root:
            #v 为当前root节点，将以该节点为起点进行cnarw
            idepth = 0
            v = root[iroot]
            node_ids.append(v)
            #root_nbhd 将储存root(iroot)的所有邻居节点
            deg_root,root_nbhd = self.node_nbhd(v, self.adj_train.indices,self.adj_train.indptr)
            if (deg_root != None):                
                while idepth < self.size_depth:  #沿着该root，走size_depth步
                    next_node = 0
                    p = random.random()
                    rand = random.randint(0,len(root_nbhd)-1)
                    next_node = root_nbhd[rand]
                    #print('next_node:',next_node)
                    if next_node == v: #判断是否为自环
                        app_v = np.random.choice(self.node_train, 1)
                        while app_v in root:
                            app_v = np.random.choice(self.node_train, 1)
                        root = np.concatenate((root,app_v))                                                     
                        break
                    deg_next_root,next_root_nbhd = self.node_nbhd(next_node, self.adj_train.indices,self.adj_train.indptr)         
                    if deg_next_root != None:
                        com_Node = len(set(root_nbhd) & set(next_root_nbhd))  #求公共节点个数
                        p_pick = 1 - com_Node/min(deg_root,deg_next_root)   #师弟写错了吧，是min 但师弟写的max
                        if p_pick > p: 
                            node_ids.append(next_node)                    
                            idepth = idepth + 1
                        """
                        if next_node == v: #判断是否为自环                    
                            app_v = np.random.choice(self.node_train, 1)
                            while app_v in root:
                                app_v = np.random.choice(self.node_train, 1)#是自环
                            break
                        else:
                            deg_next_root,next_root_nbhd = self.node_nbhd(next_node, self.adj_train.indices,self.adj_train.indptr)
                            if deg_next_root != None:
                                com_Node = len(set(root_nbhd) & set(next_root_nbhd))
                                p_pick = 1 - com_Node/min(deg_root,deg_next_root)
                                if p_pick > p:
                                    node_ids.append(next_node)
                                    idepth = idepth + 1
                         """


            iroot = iroot + 1
        """
        #print each subgraph node id and save
        print("node id:", len(node_ids))  #没去重没升序
        with open('node_id.txt', 'a') as f:
            for i in range(len(node_ids)):
                f.write(str(node_ids[i])+',')
            f.write('\n')
        #path = os.path.abspath('subgraph_node_id.txt')
        #print('path:', path)
       """             
        ret = self._helper_extract_subgraph(node_ids)
        ret = list(ret)
        for i in range(len(ret)):
            ret[i] = [ret[i]]
        return ret    
   
    
    def preproc(self, **kwargs):
        pass

    
##cnarw修改为收敛后取点的采样算法，法一
class CNARWconvergence_SamplingVanillaPython(GraphSampler):
    """
    This class is to use cnarw to define sampler.
    """
    def __init__(self, adj_train, node_train, size_subgraph, size_root, size_depth):
        self.size_root = size_root
        self.size_depth = size_depth
        size_subgraph = size_root * size_depth
        super().__init__(adj_train, node_train, size_subgraph, {})
        
        
    '''
    在minibatch中每个sampler都将调用par_sample:用于将采样算法取出的node
    转化成可以输入到模型中的形式。
    '''
    def node_nbhd(self,v,indices,indptr):
        '''
        给定一个节点v，该函数返回其所有邻居节点以及节点v的deg
        '''
        #all_degree = indptr[-1]
        #print("all_degree:",all_degree)
        
        if indptr[v+1] - indptr[v]>0:
            deg_v = indptr[v+1] - indptr[v]
            root_nbhd = []
            u0 = indices[indptr[v]]
            root_nbhd.append(u0)
            r = 1
            while r<deg_v:
                u1 = indices[indptr[v+1]-deg_v+r]
                root_nbhd.append(u1)
                r = r + 1
            assert deg_v==len(root_nbhd)
            return deg_v,root_nbhd
        else:
            return None,None        
    
    def par_sample(self, stage, **kwargs):
        iroot = 0
        root = np.random.choice(self.node_train, self.size_root) #获取root节点
        node_ids_long = []
        while iroot < self.size_root:
            #v 为当前root节点，将以该节点为起点进行cnarw
            idepth = 0
            v = root[iroot]
            node_ids_long.append(v)  #
            #root_nbhd 将储存root(iroot)的所有邻居节点
            deg_root,root_nbhd = self.node_nbhd(v, self.adj_train.indices,self.adj_train.indptr)
            if (deg_root != None):                
                while idepth < 500 + self.size_depth:  #沿着该root，走size_depth步
                    next_node = 0
                    p = random.random()
                    rand = random.randint(0,len(root_nbhd)-1)
                    next_node = root_nbhd[rand]
                    #print('next_node:',next_node)
                    if next_node == v: #判断是否为自环
                        app_v = np.random.choice(self.node_train, 1)
                        while app_v in root:
                            app_v = np.random.choice(self.node_train, 1)
                        root = np.concatenate((root,app_v))                                                     
                        break
                    deg_next_root,next_root_nbhd = self.node_nbhd(next_node, self.adj_train.indices,self.adj_train.indptr)         
                    if deg_next_root != None:
                        com_Node = len(set(root_nbhd) & set(next_root_nbhd))  #求公共节点个数
                        p_pick = 1 - com_Node/min(deg_root,deg_next_root)   #师弟写错了吧，是min 但师弟写的max
                        if p_pick > p: 
                            node_ids_long.append(next_node)                    
                            idepth = idepth + 1
                    node_ids_short = node_ids_long[-8:]
                    node_ids.extend(node_ids_short)
                    
            iroot = iroot + 1
        """
        #print each subgraph node id and save
        print("node id:", len(node_ids))  #没去重没升序
        with open('node_id.txt', 'a') as f:
            for i in range(len(node_ids)):
                f.write(str(node_ids[i])+',')
            f.write('\n')
        #path = os.path.abspath('subgraph_node_id.txt')
        #print('path:', path)
       """             
        ret = self._helper_extract_subgraph(node_ids)
        ret = list(ret)
        for i in range(len(ret)):
            ret[i] = [ret[i]]
        return ret    
      
    def preproc(self, **kwargs):
        pass


#cnarw修改为收敛后取点的采样算法，法二
class CNARWconvergence2_SamplingVanillaPython(GraphSampler):
    """
    This class is to use cnarw to define sampler.
    """
    def __init__(self, adj_train, node_train, size_subgraph, size_root, size_depth):
        self.size_root = size_root
        self.size_depth = size_depth
        size_subgraph = size_root * size_depth
        super().__init__(adj_train, node_train, size_subgraph, {})
        
        
    '''
    在minibatch中每个sampler都将调用par_sample:用于将采样算法取出的node
    转化成可以输入到模型中的形式。
    '''
    def node_nbhd(self,v,indices,indptr):
        '''
        给定一个节点v，该函数返回其所有邻居节点以及节点v的deg
        '''
        #all_degree = indptr[-1]
        #print("all_degree:",all_degree)
        
        if indptr[v+1] - indptr[v]>0:
            deg_v = indptr[v+1] - indptr[v]
            root_nbhd = []
            u0 = indices[indptr[v]]
            root_nbhd.append(u0)
            r = 1
            while r<deg_v:
                u1 = indices[indptr[v+1]-deg_v+r]
                root_nbhd.append(u1)
                r = r + 1
            assert deg_v==len(root_nbhd)
            return deg_v,root_nbhd
        else:
            return None,None

        
    def par_sample(self, stage, **kwargs):
        iroot = 0
        root = np.random.choice(self.node_train, self.size_root) #获取root节点
        node_ids = []
        while iroot < self.size_root:
            #v 为当前root节点，将以该节点为起点进行cnarw
            idepth = 0
            v = root[iroot]
            node_ids.append(v)
            #root_nbhd 将储存root(iroot)的所有邻居节点
            deg_root,root_nbhd = self.node_nbhd(v, self.adj_train.indices,self.adj_train.indptr)
            if (deg_root != None):                
                while idepth < 1000 + self.size_depth:    #沿着该root，走size_depth步
                    next_node = 0
                    p = random.random()
                    rand = random.randint(0,len(root_nbhd)-1)
                    next_node = root_nbhd[rand]
                    #print('next_node:',next_node)
                    if next_node == v: #判断是否为自环
                        app_v = np.random.choice(self.node_train, 1)
                        while app_v in root:
                            app_v = np.random.choice(self.node_train, 1)
                        root = np.concatenate((root,app_v))                                                     
                        break
                    deg_next_root,next_root_nbhd = self.node_nbhd(next_node, self.adj_train.indices,self.adj_train.indptr)    
                    if deg_next_root != None:
                        com_Node = len(set(root_nbhd) & set(next_root_nbhd))  #求公共节点个数
                        p_pick = 1 - com_Node/min(deg_root,deg_next_root)   #师弟写错了吧，是min 但师弟写的max
                        if p_pick > p: 
                            #node_ids.append(next_node)  
                            if idepth > 1000:
                                node_ids.append(next_node)
                            idepth = idepth + 1    
            iroot = iroot + 1
        """
        #print each subgraph node id and save
        print("node id:", len(node_ids))  #没去重没升序
        with open('node_id.txt', 'a') as f:
            for i in range(len(node_ids)):
                f.write(str(node_ids[i])+',')
            f.write('\n')
        #path = os.path.abspath('subgraph_node_id.txt')
        #print('path:', path)
       """             
        ret = self._helper_extract_subgraph(node_ids)
        ret = list(ret)
        for i in range(len(ret)):
            ret[i] = [ret[i]]
        return ret  
    
    def preproc(self, **kwargs):
        pass




# RW sampling by Python
class RW_SamplingVanillaPython(GraphSampler):
    def __init__(self, adj_train, node_train, size_subgraph, size_root, size_depth):
        self.size_root = size_root
        self.size_depth = size_depth
        size_subgraph = size_root * size_depth
        super().__init__(adj_train, node_train, size_subgraph, {})
        
        
    def node_nbhd(self,v,indices,indptr):
        '''
        给定一个节点v，该函数返回其所有邻居节点以及节点v的deg
        '''
        if indptr[v+1] - indptr[v]>0:
            deg_v = indptr[v+1] - indptr[v]
            root_nbhd = []
            u0 = indices[indptr[v]]
            root_nbhd.append(u0)
            r = 1
            while r<deg_v:
                u1 = indices[indptr[v+1]-deg_v+r]
                root_nbhd.append(u1)
                r = r + 1
            assert deg_v==len(root_nbhd)
            return deg_v,root_nbhd
        else:
            return None,None
        
    def par_sample(self, stage, **kwargs):
        iroot = 0
        root = np.random.choice(self.node_train, self.size_root) #获取root节点
        node_ids = []
        while iroot < self.size_root:
            #v 为当前root节点，将以该节点为起点进行rw
            idepth = 0
            v = root[iroot]
            node_ids.append(v)
            deg_root,root_nbhd = self.node_nbhd(v, self.adj_train.indices,self.adj_train.indptr)

            if (deg_root != None):                
                while idepth < self.size_depth:  #沿着该root，走size_depth步
                    next_node = 0
                    #p = random.random()
                    rand = random.randint(0,len(root_nbhd)-1)
                    next_node = root_nbhd[rand]
                    #print('next_node:',next_node)
                    if next_node == v: #判断是否为自环
                        app_v = np.random.choice(self.node_train, 1)
                        while app_v in root:
                            app_v = np.random.choice(self.node_train, 1)
                        root = np.concatenate((root,app_v))                                                     
                        break
                    #deg_next_root,next_root_nbhd = self.node_nbhd(next_node, self.adj_train.indices,self.adj_train.indptr)
                    else:
                        node_ids.append(next_node)                    
                        idepth = idepth + 1


            iroot = iroot + 1
        
        
        '''
        #print each subgraph node id and save
        print("rw node id:", len(node_ids))  #没去重没升序
        with open('rw_node_id.txt', 'a') as f:
            for i in range(len(node_ids)):
                f.write(str(node_ids[i])+',')
            f.write('\n')
        #path = os.path.abspath('subgraph_node_id.txt')
        #print('path:', path)
        '''
                     
        ret = self._helper_extract_subgraph(node_ids)
        ret = list(ret)
        for i in range(len(ret)):
            ret[i] = [ret[i]]
        return ret
        
            
    def preproc(self, **kwargs):
        pass   

    

