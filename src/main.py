import os
import numpy as np
import tensorflow as tf
import scipy.io as scio
from net_structure_img import img_net_strucuture
from net_structure_txt import txt_net_strucuture
from utils.calc_hammingranking import calc_map
import _pickle as cPickle
from datetime import datetime, timedelta


# environmental setting: setting the following parameters based on your experimental environment.
GPU_ID = '0'  # GPU ID
per_process_gpu_memory_fraction = 0.9

# data parameters 数据源
DATA_DIR = '../wiki/'

# hyper-parameters 参数列表
MAX_ITER = 150   # epoch 最大迭代次数
gamma = 1
eta = 1
output_dim = 10  # bit 哈希码维度
batch_size = 2   # batch_size大小
IMAGE_DIM = 4096  # image_dim 图像原始维度
TEXT_DIM = 100  # text_dim 文本原始维度
HIDDEN_DIM = 2048  # hidden_dim 隐层维度
CLASS_DIM = 10  # c 类别数
LEARNING_RATE = 0.01  # lr 学习率
num_train = 1500 # 训练样本数量
num_semi = 500 # 无标签样本数量
num_test = 866 # 测试样本数量
unupdated_size = num_train+num_semi-batch_size*2

print("loading fea label data.....")
# wiki总数据
data = scio.loadmat(DATA_DIR+'wiki_deep_fea.mat')

# 训练集 有标签
train_list = data['trainIdx']
train_list = np.asarray(train_list,dtype='int32')
train_list = train_list[0]
train_list = [i-1 for i in train_list]     # list 集合

# 训练集 无标签
semi_list = data['valIdx']
semi_list = np.asarray(semi_list,dtype='int32')
semi_list = semi_list[0]
semi_list = [i-1 for i in semi_list]

# 总的训练集 有标签+无标签
all_train_list = np.concatenate((train_list,semi_list),axis=0)

# 测试集
test_list = data['testIdx']
test_list = np.asarray(test_list,dtype='int32')
test_list = test_list[0]
test_list = [i-1 for i in test_list]

# 标签信息
data1 = data['gnd'].reshape(-1)
all_label = np.zeros((num_train+num_semi+num_test, CLASS_DIM))
for i in range(len(data1)):
	all_label[i][data1[i] - 1] = 1

# 图像特征
img_fea = data['imgFea']    # (4096,2866)
img_fea = np.transpose(img_fea)  # (2866,4096)
img_fea = np.asarray(img_fea,dtype='float32')

# 文本特征
txt_fea = data['txtFea']    # (100,2866)
txt_fea = np.transpose(txt_fea)  # (2866,100)
txt_fea = np.asarray(txt_fea,dtype='float32')

print("ending fea label data.....")


print("start predict semi data label")
can_list = data['gnd']
pred_label = np.zeros((num_semi,CLASS_DIM))       # semi_data 的标签信息 (500,10)
cnt = 0
for i in semi_list:
    v1 = txt_fea[i]
    min = np.sqrt(np.sum(np.square(v1-txt_fea[train_list[0]])))
    index = 0
    for j in train_list:
        dis = np.sqrt(np.sum(np.square(v1-txt_fea[j])))
        if dis < min:
            min = dis
            index = can_list[j][0]-1
    pred_label[cnt][index]=1
    cnt+=1
print("end predict semi data label")

def train_img_net(image_input, cur_f_batch, var, ph, train_x, train_L, lr, train_step_x, Sim):
	F = var['F']      # (output_dim,num_train)
	batch_size = var['batch_size']
	batch_size = batch_size*2    # batch_size
	num_train = train_x.shape[0]

	# index = range(0, num_train - 1, 1)
	for iter in range(int(num_train / batch_size)):
		# random.permutation随机排列num_train个序列
		index = np.random.permutation(num_train)
		ind = index[0: batch_size]
		num = batch_size//2
		num1 = batch_size//2
		p = output_dim//2
		p1 = output_dim//2
		epi = np.random.uniform(-1e-9, 1e-9)  # 一个(-1e-9,1e-9)范围内的随机数
		epi1 = 0

		# CDPAE distance-preserving
		T = var['G']     # (output_dim,num_train)
		T = np.transpose(T)  # (num_train,output_dim)
		T = T[:num,:] + epi    # (batch_size,output_dim)
		T_n = T
		T_n[:,0:p] = epi1
		I = var['F']
		I = np.transpose(I)
		I = I[:num,:] + epi
		I_n = I
		I_n[:,0:p1] = epi1

		T1 = var['G']
		T1 = np.transpose(T1)
		T1 = T1[num1:batch_size,:] + epi
		T1_n = T1
		T1_n[:,0:p] = epi1
		I1 = var['F']
		I1 = np.transpose(I1)
		I1 = I1[num1:batch_size,:] + epi
		I1_n = I1
		I1_n[:,0:p1] = epi1


		# ind = index[iter * batch_size: (iter + 1) * batch_size]
		# setdiff1d可以求解出存在于第一个集合但是并不存在于第二个集合中的元素。返回值是一个数组集合
		unupdated_ind = np.setdiff1d(range(num_train), ind)
		# train_L为训练集标签  astype转换数组的数据类型
		sample_L = train_L[ind,:]
		image = train_x[ind,:].astype(np.float32)
		S = calc_neighbor(sample_L, train_L)
		# eval()将字符串str当成有效的表达式来求值并返回计算结果  feed_dict的作用是给使用placeholder创建出来的tensor赋值
		cur_f = cur_f_batch.eval(feed_dict={image_input: image})
		F[:, ind] = cur_f

		train_step_x.run(feed_dict={ph['S_x']: S, ph['G']: var['G'], ph['b_batch']: var['B'][:, ind],
																ph['F_']: F[:, unupdated_ind], ph['lr']: lr, image_input: image,
									ph['Im']:I, ph['Im_n']:I_n, ph['Im1']:I1, ph['Im1_n']:I1_n,
									ph['Tx']:T, ph['Tx_n']:T_n, ph['Tx1']:T1, ph['Tx1_n']:T1_n
									})

	return F


def train_txt_net(text_input, cur_g_batch, var, ph, train_y, train_L, lr, train_step_y, Sim):
	G = var['G']
	batch_size = var['batch_size']
	batch_size = batch_size * 2  # batch_size
	# num_train为 txt 训练文本的数量
	num_train = train_y.shape[0]

	for iter in range(int(num_train / batch_size)):
		index = np.random.permutation(num_train)
		ind = index[0: batch_size]
		num = batch_size//2
		num1 = batch_size//2
		p = output_dim//2
		p1 = output_dim//2
		epi = np.random.uniform(-1e-9, 1e-9)  # 一个(-1e-9,1e-9)范围内的随机数
		epi1 = 0

		# CDPAE distance-preserving
		T = var['G']     # (output_dim,batch_size)
		T = np.transpose(T)
		T = T[:num,:] + epi
		T_n = T
		T_n[:,0:p] = epi1
		I = var['F']
		I = np.transpose(I)
		I = I[:num,:] + epi
		I_n = I
		I_n[:,0:p1] = epi1

		T1 = var['G']
		T1 = np.transpose(T1)
		T1 = T1[num1:batch_size,:] + epi
		T1_n = T1
		T1_n[:,0:p] = epi1
		I1 = var['F']
		I1 = np.transpose(I1)
		I1 = I1[num1:batch_size,:] + epi
		I1_n = I1
		I1_n[:,0:p1] = epi1


		unupdated_ind = np.setdiff1d(range(num_train), ind)
		sample_L = train_L[ind, :]
		text = train_y[ind, :].astype(np.float32)
		text = text.reshape([text.shape[0], 1, text.shape[1], 1])

		S = calc_neighbor(train_L, sample_L)
		cur_g = cur_g_batch.eval(feed_dict={text_input: text})
		G[:,ind] = cur_g

		train_step_y.run(feed_dict={ph['S_y']: S, ph['F']: var['F'], ph['b_batch']: var['B'][:, ind],
																ph['G_']: G[:, unupdated_ind], ph['lr']: lr, text_input: text,
									ph['Im']: I, ph['Im_n']: I_n, ph['Im1']: I1, ph['Im1_n']: I1_n,
									ph['Tx']: T, ph['Tx_n']: T_n, ph['Tx1']: T1, ph['Tx1_n']: T1_n
									})
	return G


# 计算两个标签的相似性
def calc_neighbor(label_1, label_2):
	# dot()函数是矩阵乘 *则表示逐个元素相乘
	Sim = (np.dot(label_1, label_2.transpose()) > 0).astype(int)
	return Sim


def calc_loss(B, F, G, Sim, gamma, eta):
	# matmul矩阵相乘
	# theta代表θij
	theta = np.matmul(np.transpose(F), G) / 2
	# term1项
	term1 = np.sum(np.log(1+np.exp(theta)) - Sim * theta)
	# term2项
	term2 = np.sum(np.power((B-F), 2) + np.power(B-G,2))
	# term3项
	term3 = np.sum(np.power(np.matmul(F, np.ones((F.shape[1],1))),2)) + np.sum(np.power(np.matmul(G, np.ones((F.shape[1],1))),2))
	# J损失函数 gamma代表γ eta代表参数
	loss = term1 + gamma * term2 + eta * term3
	return loss


def generate_image_code(image_input, cur_f_batch, X, bit):
	batch_size = 128
	num_data = X.shape[0]
	# np.linspace函数可以num_data个等差数据 第一个元素为0 最后一个元素为num_data-1
	index = np.linspace(0, num_data - 1, num_data).astype(int)
	# 初始化哈希码B
	B = np.zeros([num_data, bit], dtype=np.float32)
	for iter in range(num_data // batch_size + 1):
		min_val = (iter+1)*batch_size
		if min_val>num_data:
			min_val= num_data
		ind = index[iter * batch_size : min_val]
		image = X[ind,:].astype(np.float32)
		cur_f = cur_f_batch.eval(feed_dict={image_input: image})
		B[ind, :] = cur_f.transpose()
	B = np.sign(B)
	return B


def generate_text_code(text_input, cur_g_batch, Y, bit):
	batch_size = 128
	num_data = Y.shape[0]
	index = np.linspace(0, num_data - 1, num_data).astype(int)
	B = np.zeros([num_data, bit], dtype=np.float32)
	for iter in range(num_data // batch_size + 1):
		min_val = (iter + 1) * batch_size
		if min_val > num_data:
			min_val = num_data
		ind = index[iter * batch_size : min_val]
		text = Y[ind, :].astype(np.float32)

		text = text.reshape([text.shape[0], 1, text.shape[1], 1])

		cur_g = cur_g_batch.eval(feed_dict={text_input: text})
		B[ind, :] = cur_g.transpose()
	B = np.sign(B)
	return B

def cos_dis(x,y):
	dis = 1.0 -tf.reduce_sum(tf.multiply(x,y),1)/\
		  tf.sqrt(tf.reduce_sum(tf.square(x),1))/\
		  tf.sqrt(tf.reduce_sum(tf.square(y),1))
	return dis

def cos_loss(x,y):
	loss =tf.reduce_mean(cos_dis(x,y))
	return loss

def main():
	gpuconfig = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction))
	os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

	with tf.Graph().as_default(), tf.Session(config=gpuconfig) as sess:
		# config = tf.ConfigProto(allow_soft_placement=True)
		# config.gpu_options.allow_growth = True
		# sess = tf.Session(config=config)

		# construct image network
		image_input = tf.placeholder(tf.float32, [None,IMAGE_DIM])
		net = img_net_strucuture(image_input, IMAGE_DIM, HIDDEN_DIM,output_dim)
		cur_f_batch = tf.transpose(net['fc8'])     # (output_dim,2000)

		# construct text network
		text_input =tf.placeholder(tf.float32,(None,) + (1, TEXT_DIM, 1))
		#   [None,TEXT_DIM]
		cur_g_batch = txt_net_strucuture(text_input, TEXT_DIM, output_dim)
		cur_g_batch = tf.transpose(cur_g_batch)    # (output_dim,2000)

		# training DCMH algorithm
		train_L = all_label[train_list]   # label (1500,10)
		train_x = img_fea[all_train_list]  # img (2000,4096)
		train_y = txt_fea[all_train_list]   # txt (2000,100)

		query_L = all_label[test_list]
		query_x = img_fea[test_list]
		query_y = txt_fea[test_list]

		retrieval_L = all_label[all_train_list]  # (2000,10)
		retrieval_x = img_fea[all_train_list]   #(2000,4096)
		retrieval_y = txt_fea[all_train_list]    #(2000,100)

		# train+semi 的标签信息 train_L (2000,10)
		train_L = np.concatenate((train_L, pred_label), axis=0)

		# 构建相似性矩阵 (2000,2000) 和总的训练样本数量一致
		Sim = calc_neighbor(train_L, train_L)

		var = {}
		lr = np.linspace(np.power(10, -1.5), np.power(10, -6.), MAX_ITER)
		var['lr'] = lr
		var['batch_size'] = batch_size
		var['F'] = np.random.randn(output_dim, num_train+num_semi)
		var['G'] = np.random.randn(output_dim, num_train+num_semi)
		var['B'] = np.sign(var['F']+var['G'])
		var['unupdated_size'] = unupdated_size

		ph = {}
		ph['lr'] = tf.placeholder('float32', (), name='lr')
		ph['S_x'] = tf.placeholder('float32', [batch_size*2, num_train+num_semi], name='pS_x')
		ph['S_y'] = tf.placeholder('float32', [num_train+num_semi, batch_size*2], name='pS_y')
		ph['F'] = tf.placeholder('float32', [output_dim, num_train+num_semi], name='pF')
		ph['G'] = tf.placeholder('float32', [output_dim, num_train+num_semi], name='pG')
		ph['F_'] = tf.placeholder('float32', [output_dim, unupdated_size], name='unupdated_F')
		ph['G_'] = tf.placeholder('float32', [output_dim, unupdated_size], name='unupdated_G')
		ph['b_batch'] = tf.placeholder('float32', [output_dim, batch_size*2], name='b_batch')
		ph['ones_'] = tf.constant(np.ones([unupdated_size, 1], 'float32'))
		ph['ones_batch'] = tf.constant(np.ones([batch_size*2, 1], 'float32'))

		# comprehensive distance-preserving
		ph['Im'] = tf.placeholder('float32',[batch_size,output_dim],name='Image')     # batch_size的前一半,图像维度
		ph['Im_n'] = tf.placeholder('float32',[batch_size,output_dim],name='Image_n')   # batch_size的前一半,图像维度的一半
		ph['Im1'] = tf.placeholder('float32',[batch_size,output_dim],name='Image1')   # batch_size的后一半,图像维度
		ph['Im1_n'] = tf.placeholder('float32', [batch_size, output_dim], name='Image1_n')   # batch_size的后一半,图像维度的一半

		ph['Tx'] = tf.placeholder('float32',[batch_size,output_dim],name='Text')
		ph['Tx_n'] = tf.placeholder('float32',[batch_size,output_dim],name='Text_n')    # batch_size的前一半,文本维度的一半
		ph['Tx1'] = tf.placeholder('float32',[batch_size,output_dim],name='Text1')
		ph['Tx1_n'] = tf.placeholder('float32', [batch_size, output_dim], name='Text1_n')   # batch_size的后一半，文本维度的一半


		theta_x = 1.0 / 2 * tf.matmul(tf.transpose(cur_f_batch), ph['G'])   # (2000,64) (64,2000) => (2000,2000)
		theta_y = 1.0 / 2 * tf.matmul(tf.transpose(ph['F']), cur_g_batch)

		# DCMH image loss
		logloss_x = -tf.reduce_sum(tf.multiply(ph['S_x'], theta_x) - (tf.log(1.0+tf.exp(-tf.abs(theta_x))) + tf.maximum(0.0,theta_x)))
		quantization_x = tf.reduce_sum(tf.pow((ph['b_batch'] - cur_f_batch), 2))
		balance_x = tf.reduce_sum(tf.pow(tf.matmul(cur_f_batch, ph['ones_batch']) + tf.matmul(ph['F_'], ph['ones_']), 2))
		loss1_x = tf.div(logloss_x + gamma * quantization_x + eta * balance_x, float((num_train+num_semi) * batch_size))
		# CDPAE image loss
		D_x =tf.reshape(tf.sqrt(tf.multiply(cos_dis(ph['Im'],ph['Im1']),
										   cos_dis(ph['Tx'],ph['Tx1']))+0.00001),[batch_size,1])
		loss2_x = cos_loss(ph['Im_n'],ph['Tx_n'])+cos_loss(ph['Im1_n'],ph['Tx1_n'])  # L_pair
		I1_x = tf.reshape(cos_dis(ph['Im_n'],ph['Im1_n']),[batch_size,1])
		T1_x = tf.reshape(cos_dis(ph['Tx_n'],ph['Tx1_n']),[batch_size,1])
		I2_x = tf.reshape(cos_dis(ph['Im_n'],ph['Tx1_n']),[batch_size,1])
		T2_x = tf.reshape(cos_dis(ph['Tx_n'],ph['Im1_n']),[batch_size,1])
		lo2 = tf.abs(I2_x - D_x) + tf.abs(T2_x - D_x)
		lo3 = tf.abs(I1_x - D_x) +  tf.abs(T1_x - D_x)
		loss3_x = tf.reduce_mean(lo2)     # L_heter
		loss4_x = tf.reduce_mean(lo3)     # L_homo
		loss_x = loss1_x + loss2_x + (loss3_x + loss4_x)*0.3


		# DCMH text loss
		logloss_y = -tf.reduce_sum(tf.multiply(ph['S_y'], theta_y) - (tf.log(1.0+tf.exp(-tf.abs(theta_y))) + tf.maximum(0.0,theta_y)))
		quantization_y = tf.reduce_sum(tf.pow((ph['b_batch'] - cur_g_batch), 2))
		balance_y = tf.reduce_sum(tf.pow(tf.matmul(cur_g_batch, ph['ones_batch']) + tf.matmul(ph['G_'], ph['ones_']), 2))
		loss1_y = tf.div(logloss_y + gamma * quantization_y + eta * balance_y, float((num_train+num_semi) * batch_size))
		# CDPAE text loss
		D_y = tf.reshape(tf.sqrt(tf.multiply(cos_dis(ph['Im'], ph['Im1']),
											 cos_dis(ph['Tx'], ph['Tx1'])) + 0.00001), [batch_size, 1])
		loss2_y = cos_loss(ph['Im_n'], ph['Tx_n']) + cos_loss(ph['Im1_n'], ph['Tx1_n'])  # L_pair
		I1_y = tf.reshape(cos_dis(ph['Im_n'], ph['Im1_n']), [batch_size, 1])
		T1_y = tf.reshape(cos_dis(ph['Tx_n'], ph['Tx1_n']), [batch_size, 1])
		I2_y = tf.reshape(cos_dis(ph['Im_n'], ph['Tx1_n']), [batch_size, 1])
		T2_y = tf.reshape(cos_dis(ph['Tx_n'], ph['Im1_n']), [batch_size, 1])
		lo2 = tf.abs(I2_y - D_y) + tf.abs(T2_y - D_y)
		lo3 = tf.abs(I1_y - D_y) + tf.abs(T1_y - D_y)
		loss3_y = tf.reduce_mean(lo2)  # L_heter
		loss4_y = tf.reduce_mean(lo3)  # L_homo
		loss_y = loss1_y + loss2_y + (loss3_y + loss4_y)*0.3


		# 优化器
		optimizer = tf.train.GradientDescentOptimizer(ph['lr'])

		gradient_x = optimizer.compute_gradients(loss_x)
		gradient_y = optimizer.compute_gradients(loss_y)
		train_step_x = optimizer.apply_gradients(gradient_x)
		train_step_y = optimizer.apply_gradients(gradient_y)

		sess.run(tf.global_variables_initializer())

		loss_ = calc_loss(var['B'], var['F'], var['G'], Sim, gamma, eta)
		print('epoch: %3d, loss: %3.3f' % (0, loss_))

		result = {}
		result['loss'] = []
		result['imapi2t'] = []
		result['imapt2i'] = []

		print('...training procedure starts')

		for epoch in range(MAX_ITER):
			lr = var['lr'][epoch]
			# update F
			var['F'] = train_img_net(image_input, cur_f_batch, var, ph,  train_x, train_L, lr, train_step_x, Sim)

			# update G
			var['G'] = train_txt_net(text_input, cur_g_batch, var, ph, train_y, train_L, lr, train_step_y, Sim)

			# update B
			var['B'] = np.sign(gamma * (var['F'] + var['G']))

			# calculate loss
			loss_ = calc_loss(var['B'], var['F'], var['G'], Sim, gamma, eta)

			print('...epoch: %3d, loss: %3.3f, comment: update B' % (epoch + 1, loss_))
			result['loss'].append(loss_)

		print('...training procedure finish')

		# 测试集
		qBX = generate_image_code(image_input, cur_f_batch, query_x, output_dim)
		qBY = generate_text_code(text_input, cur_g_batch, query_y, output_dim)
		# 检索数据库
		rBX = generate_image_code(image_input, cur_f_batch, query_x, output_dim)
		rBY = generate_text_code(text_input, cur_g_batch, query_y, output_dim)

		# MAP PR
		mapi2t = calc_map(qBX, rBY, query_L, query_L, "i2t")
		mapt2i = calc_map(qBY, rBX, query_L, query_L, "t2i")


		print('test map: map(i->t): %3.3f, map(t->i): %3.3f' % (mapi2t, mapt2i))



if __name__ == '__main__':
	main()



