
n_objects  = 5 # number of planets(nodes)
object_dim = 25 

n_relations  = n_objects * (n_objects - 1) # number of edges in fully connected graph
relation_dim = 1

effect_dim = 100 #effect's vector size

batch_size = 4092
USE_CUDA = True

SHAPE_2_IDX = {'Cone':0, 'Cube':1, 'Sphere':2, "Occluder":3}
IDX_2_SHAPE = {0 : "Cone", 1 : "Cube", 2 : "Sphere", 3: "Occluder"}
