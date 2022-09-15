import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


graph = sp.load_npz("adj_train.npz")
feats = np.load("feats.npy")


indptr = graph.indptr
indices = graph.indices
data = graph.data


line = [indices[indptr[i]:indptr[i+1]] for i in range(len(indptr)-1)]

result = []
for i,point0 in enumerate(line): #循环所有边
    if i%1000==0:
        print(i)
    for point1 in point0:    #循环所有边
        sim = 0
        point1_children = line[point1]
        point0_children = point0
        z = len(point0)*len(line[point1])
        for p1 in point1_children:   #循环计算
            for p0 in point0_children:  #循环所有节点
                sim += cosine_similarity([list(feats[p1])],[list(feats[p0])])[0][0]
                # sim += 0.5
        
        result.append(sim/z)
        #print(result)

print(len(result))
np.save("sim.npy", np.array(result))


