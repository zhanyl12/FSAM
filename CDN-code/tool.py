import pickle
import dpkt
import random
import numpy as np
from tqdm import tqdm, trange

cdn = ['origin', 'ali', 'tencent', 'baidu', 'cloudflare', 'cloudfront','qiniu', 'fastly', 'self']
websites_kind = ['origin', 'ali', 'tencent', 'baidu', 'cloudflare', 'cloudfront','qiniu', 'fastly', 'self']
websites = [0, 1, 2, 3, 4, 5, 6, 7, 8]


ip_features = {'hl':1,'tos':1,'len':2,'df':1,'mf':1,'ttl':1,'p':1}
tcp_features = {'off':1,'flags':1,'win':2}
udp_features = {'ulen':2}
max_byte_len = 50
n = 8

def mask(p):
	p.src = b'\x00\x00\x00\x00'
	p.dst = b'\x00\x00\x00\x00'
	p.sum = 0
	p.id = 0
	p.offset = 0

	if isinstance(p.data, dpkt.tcp.TCP):
		p.data.sport = 0
		p.data.dport = 0
		p.data.seq = 0
		p.data.ack = 0
		p.data.sum = 0

	elif isinstance(p.data, dpkt.udp.UDP):
		p.data.sport = 0
		p.data.dport = 0
		p.data.sum = 0

	return p

def pkt2feature(data, k):
	flow_dict = {'train':{}, 'test':{}}
	print(k)
	# train->protocol->flowid->[pkts]
	for p in websites:
		flow_dict['train'][p] = []
		flow_dict['test'][p] = []
		all_pkts = []
		p_keys = list(data[p].keys())

		for flow in p_keys:
			pkts = data[p][flow]
			#动这里
			#print ('The pkts is: ')
			#print (pkts[0])
			packet_number = int(len(pkts))
			pkts_after = []
			feature_vector = []
			for each in pkts:
				#print(each)
				pkt = mask(each)
				raw_byte = pkt.pack()
				pkts_after.append(raw_byte)

			#还有一个可以修改的地方 是是否需要丢掉字节数较少的数据包
			if len(pkts_after)<=15:  #delete no data flow
				continue
			#method 1 first n
			'''
			for i in range(n):
				feature_vector.append(pkts_after[i])
			'''
			#method 2 random n
			'''
			counting_number = 0
			last_number = -1
			while counting_number<n:
				choose_number = random.randint(0, packet_number-1)
				if choose_number != last_number:
					last_number = choose_number
					feature_vector.append(pkts_after[choose_number])
					counting_number = counting_number + 1
				else:
					continue
			'''
			#method 3 random lianxu n
			'''
			choose_number = random.randint(0,packet_number-1-n)
			for i in range(n):
				feature_vector.append(pkts_after[choose_number+i])
			'''
			#method 4 mid n
			
			for i in range(n):
				feature_vector.append(pkts_after[int(packet_number/2)+i])
			

			all_pkts.append(feature_vector)
			#all_pkts.extend(pkts)
		random.Random(1024).shuffle(all_pkts)
		print (p,len(all_pkts))
		#for each in all_pkts:
		#	print(len(each))
		for idx in range(len(all_pkts)):
			byte = []
			pos = []
			for i in range(n):
				for x in range(min(len(all_pkts[idx][i]),max_byte_len)):
					byte.append(int(all_pkts[idx][i][x]))
					pos.append(i*max_byte_len + x)

				byte.extend([0]*((i+1)*max_byte_len-len(byte)))
				pos.extend([0]*((i+1)*max_byte_len-len(pos)))
			#print (len(byte),len(pos))
			if idx in range(k*int(len(all_pkts)*0.1), (k+1)*int(len(all_pkts)*0.1)):
				flow_dict['test'][p].append((byte, pos))
			else:
				flow_dict['train'][p].append((byte, pos))

		'''

		for idx in range(len(all_pkts)):
			pkt = mask(all_pkts[idx])
			raw_byte = pkt.pack()

			byte = []
			pos = []
			for x in range(min(len(raw_byte),max_byte_len)):
				byte.append(int(raw_byte[x]))
				#pos.append(x)

			byte.extend([0]*(max_byte_len-len(byte)))
			#pos.extend([0]*(max_byte_len-len(pos)))
			# if len(byte) != max_byte_len or len(pos) != max_byte_len:
			# 	print(len(byte), len(pos))
			# 	input()
			if idx in range(k*int(len(all_pkts)*0.1), (k+1)*int(len(all_pkts)*0.1)):
				flow_dict['test'][p].append((byte, pos))
			else:
				flow_dict['train'][p].append((byte, pos))
		'''
	return flow_dict

def load_epoch_data(flow_dict, train='train'):
	flow_dict = flow_dict[train]
	x, y, label = [], [], []

	for p in websites:
		pkts = flow_dict[p]
		for byte, pos in pkts:
			x.append(byte)
			y.append(pos)
			label.append(p)

	return np.array(x), np.array(y), np.array(label)[:, np.newaxis]


if __name__ == '__main__':
	# f = open('flows.pkl','rb')
	# data = pickle.load(f)
	# f.close()

	# print(data.keys())

	# dns = data['dns']
	# # print(list(dns.keys())[:10])

	# # wide dataset contains payload
	# print('================\n',
	# 	len(dns['203.206.160.197.202.89.157.51.17.53.51648'][0]))

	# print('================')
	# flow_dict = pkt2feature(data)
	# x, y, label = train_epoch_data(flow_dict)
	# print(x.shape)
	# print(y.shape)
	# print(label[0])
	hehe=[8]
	with open('pro_flows.pkl','rb') as f:
		data = pickle.load(f)
	print('Finish loading...')
	for each in hehe:
		n = int(each)
		print ('n is : ',n)
		for i in trange(10, mininterval=2, \
			desc='  - (Building fold dataset)   ', leave=False):
			print(n,i)
			flow_dict = pkt2feature(data, i)
			with open('mid'+'_pro_flows_%d_noip_fold.pkl'%i, 'wb') as f:
				pickle.dump(flow_dict, f)