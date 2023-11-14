import numpy as np
import dpkt
import random
import pickle
import os

cdn = ['origin', 'ali', 'tencent', 'baidu', 'cloudflare', 'cloudfront','qiniu', 'fastly', 'self']
websites_kind = ['blog', 'picture', 'video', 'bbs', 'social']
websites = [0, 1, 2, 3,4]

data_path = 'C://research//all_data_after'

def get_flows():
	#flows = [{} for _ in range(len(cdn))]
	flows = [{} for _ in range(len(websites))]

	filenames=os.listdir(data_path)
	for files in filenames:
		if 'pcapng' in files:
			print(files)
			pcap = dpkt.pcapng.Reader(open(data_path+'//'+files, 'rb'))
			website_class = int(files.strip().split('_')[0])
			cdn_class = files.strip().split('_')[3]
			temp_list = []
			zhanzhan = 0
			for _, buff in pcap:
				if zhanzhan >= 1000:
					break
				eth = dpkt.ethernet.Ethernet(buff)
				if isinstance(eth.data, dpkt.ip.IP) and (isinstance(eth.data.data, dpkt.udp.UDP)or isinstance(eth.data.data, dpkt.tcp.TCP)):
					# tcp or udp packet
					ip = eth.data
					temp_list.append(ip)
					zhanzhan = zhanzhan+1
			flows[website_class][files.split('.')[0]] = temp_list

	return flows

if __name__ == '__main__':
	flows = get_flows()
	for name in websites_kind:
		index = websites_kind.index(name)
		print('============================')
		print('Generate flows for %s'%name)
		print('Total flows: ', len(flows[index]))
		cnt = 0
		for k, v in flows[index].items():
			cnt += len(v)
		print('Total pkts: ', cnt)

	with open('pro_flows.pkl', 'wb') as f:
		pickle.dump(flows, f)