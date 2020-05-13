#Sina Weibo Actors Following Relationship Matrix
1. actorId.txt  
This file contains profiles of movie actors who has an account in Sina Weibo. Following information are provided for each actor: the number of fans(10k), Sina Weibo verified name, Sina Weibo Account Id, Sina Weibo Nickname, actor self description. All actors have at least 10,000 of fans, it shows that all users in this dataset are famous to some point.  

2. weibo-actors-adjacent.npz  
This file is a compressed adjacent NumPy matrix of 8508 actors. The adjacent matrix `m` stores connection for each node pairs. For all node `u`, set `m[u,u]=0`, and for any node `u` follows node `v`, then set `m[u, v]=1`, other values in matrix `m` are set to `m[u, v]=3.4028235e+38` which is the maximum value of floating-point in 32bit.

3. matrix.npy
Use `unzip weibo-actors-adjacent.npz` and get `matrix.npy` file. This file is the NumPy matrix format which used as input of computations.