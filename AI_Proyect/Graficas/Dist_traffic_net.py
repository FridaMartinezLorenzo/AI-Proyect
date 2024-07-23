import pandas as pd
import matplotlib.pyplot as plt


# Upload the dataset
EHMS= '../WUSTL-EHMS/wustl-ehms-2020.csv'
df = pd.read_csv(EHMS)


# Attributes related to the comunication protocol TCP/IP
tcp_ip_columns = ['SrcAddr', 'DstAddr', 'Sport', 'Dport', 'SrcBytes', 'DstBytes', 'SrcLoad', 'DstLoad',
                  'SrcGap', 'DstGap', 'SIntPkt', 'DIntPkt', 'SIntPktAct', 'DIntPktAct', 'SrcJitter', 'DstJitter',
                  'sMaxPktSz', 'dMaxPktSz', 'sMinPktSz', 'dMinPktSz', 'Dur', 'Trans', 'TotPkts', 'TotBytes',
                  'Load', 'Loss', 'pLoss', 'pSrcLoss', 'pDstLoss', 'Rate', 'SrcMac', 'DstMac', 'Packet_num']


total_columns = len(df.columns)
print(total_columns)
columns_no_iot = total_columns - len(tcp_ip_columns)
columns_iot = len(tcp_ip_columns)


# Grafph how many attributes are part of the IoT attributes
plt.figure(figsize=(8, 6))
labels = ['Non-TCP/IP attributes', 'TCP/IP attributes']
counts = [columns_no_iot, columns_iot]
colors = ['pink', 'magenta']


plt.bar(labels, counts, color=colors)
plt.title('Distribution of the attibutes of the traffic-comunication protocols')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()
