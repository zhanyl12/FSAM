from scapy.all import *
import os

filenames=os.listdir(r'C://research//all_data_after')
for file in filenames:
	if 'pcapng' in file:
		print(file)
		csv_table=[]
		with open('C://research//all_data_after//'+file.replace('.pcapng','.csv'),encoding='utf-8') as csv_file:
			for row in csv_file:
				csv_table.append(row)
		#print(csv_table)
		csv_file.close()
		packets = rdpcap('C://research//all_data_after//'+file)
		i = 0
		print('C://research//all_data_after//'+file)
		f = open('C://research//data_feature//'+file.replace('.pcapng','.txt'),'w',encoding='utf-8')
		print('C://research//data_feature//'+file.replace('.pcapng','.txt'))
		for packet in packets:
			i = i + 1
			if 'Raw' in packet.payload:
				#if 'UDP' in packet.payload:
				if 'TCP' in packet.payload:
					#print(packet.payload)
					#print(len(packet['Raw'].load))
					f.write('TCP'+','+str(csv_table[i].strip().split(',')[5].strip())+','+str(len(packet['Raw'].load))+'\n')
				elif 'UDP' in packet.payload:
					#print(packet.payload)
					f.write('UDP'+','+str(csv_table[i].strip().split(',')[5].strip())+','+str(len(packet['Raw'].load))+'\n')
				else:
					f.write('None'+','+str(csv_table[i].strip().split(',')[5].strip())+','+str(len(packet['Raw'].load))+'\n')
					#print(len(packet['Raw'].load))
			else:
				#packet.show()
				if 'TCP' in packet.payload:
					#print(packet.payload)
					f.write('TCP'+','+str(csv_table[i].strip().split(',')[5].strip())+','+'0'+'\n')
				elif 'UDP' in packet.payload:
					f.write('UDP'+','+str(csv_table[i].strip().split(',')[5].strip())+','+'0'+'\n')
					#print(len(packet['Raw'].load))
				else:
					f.write('None'+','+str(csv_table[i].strip().split(',')[5].strip())+','+'0'+'\n')
					#print(len(packet['Raw'].load))
		f.close()

'''
print(packets[2]['UDP'])
print(packets[0].payload)
print(packets[2].payload)
print(packets[5].payload)
print(len(packets[5]['Raw'].load))
'''
#print(packets[2]['Raw'].load)



'''if packet.haslayer(IP):
	print (packet[IP].dst)
	'''
#tshark.exe -r D:\博士学习\process_data\4_zhanzhan_cloudflare_windows7_35.pcapng -T fields -e frame.time_relative -e ip.src -e ip.dst -e ip.proto -e frame.len -e _ws.col.Info -E header=y -E separator=, > D:\博士学习\process_data\out123.csv