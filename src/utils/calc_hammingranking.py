import numpy as np

def calc_hammingDist(B1, B2):
	q = B2.shape[1]
	distH = 0.5 * (q - np.dot(B1, B2.transpose()))
	return distH

def calc_map(qB, rB, query_L, retrieval_L, type):
	# qB: {-1,+1}^{mxq}
	# rB: {-1,+1}^{nxq}
	# query_L: {0,1}^{mxl}
	# retrieval_L: {0,1}^{nxl}
	num_query = query_L.shape[0]

	# MAP
	map = 0
	for iter in range(num_query):
		gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
		tsum = np.sum(gnd)
		if tsum == 0:
			continue
		hamm = calc_hammingDist(qB[iter, :], rB)
		ind = np.argsort(hamm)
		gnd = gnd[ind]
		# count = np.linspace(1, tsum, tsum)

		tindex = np.asarray(np.where(gnd == 1)) + 1.0
		count = np.linspace(1, tindex.shape[-1], tindex.shape[-1])
		map = map + np.mean(count / (tindex))
	map = map / num_query

	# PR
	Gnd = (np.dot(query_L, retrieval_L.transpose()) > 0).astype(np.float32)
	Hamm = calc_hammingDist(qB, rB)
	Rank = np.argsort(Hamm)
	P,R = [],[]
	for k in range(1, num_query+1):
		p = np.zeros(num_query)
		r = np.zeros(num_query)
		for it in range(num_query):
			gnd = Gnd[it]
			gnd_all = np.sum(gnd)
			if gnd_all == 0:
				continue
			asc_id = Rank[it][:k]
			gnd = gnd[asc_id]
			gnd_r = np.sum(gnd)

			p[it] = gnd_r / k
			r[it] = gnd_r / gnd_all
		P.append(np.mean(p))
		R.append(np.mean(r))
	p_end = P[len(P)-1]
	r_end = R[len(R)-1]
	if r_end != 1:
		P = np.append(P,p_end)
		R = np.append(R,1)
	list = np.linspace(0,len(P)-1,20).astype(np.int)
	P = np.array(P)[list]
	R = np.array(R)[list]

	if type == "i2t":
		np.save("./I2T_P.npy",P)
		np.save("./I2T_R.npy",R)
	else:
		np.save("./T2I_P.npy",P)
		np.save("./T2I_R.npy",R)

	return map


if __name__=='__main__':
	qB = np.array([[ 1,-1, 1, 1],
								 [-1,-1,-1, 1],
								 [ 1, 1,-1, 1],
								 [ 1, 1, 1,-1]])
	rB = np.array([[ 1,-1, 1,-1],
								 [-1,-1, 1,-1],
								 [-1,-1, 1,-1],
								 [ 1, 1,-1,-1],
								 [-1, 1,-1,-1],
								 [ 1, 1,-1, 1]])
	query_L = np.array([[0, 1, 0, 0],
											[1, 1, 0, 0],
											[1, 0, 0, 1],
											[0, 1, 0, 1]])
	retrieval_L = np.array([[1, 0, 0, 1],
													[1, 1, 0, 0],
													[0, 1, 1, 0],
													[0, 0, 1, 0],
													[1, 0, 0, 0],
													[0, 0, 1, 0]])

	map = calc_map(qB, rB, query_L, retrieval_L, "i2t")
	print(map)
