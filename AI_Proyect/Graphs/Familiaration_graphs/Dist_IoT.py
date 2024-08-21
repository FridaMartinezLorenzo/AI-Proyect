import pandas as pd
import matplotlib.pyplot as plt


EHMS= '../WUSTL-EHMS/wustl-ehms-2020.csv'
df = pd.read_csv(EHMS)


iot_columns = ['SrcAddr', 'DstAddr', 'Temp', 'SpO2', 'Pulse_Rate', 'SYS', 'DIA', 'Heart_rate', 'Resp_Rate', 'ST']
total_columns = len(df.columns)
columns_no_iot = total_columns - len(iot_columns)
columns_iot = len(iot_columns)


# Graph how many attributes are related to IoT  
plt.figure(figsize=(8, 6))
labels = ['Non-IoT attributes', 'IoT attributes']
counts = [columns_no_iot, columns_iot]
colors = ['pink', 'magenta']


plt.bar(labels, counts, color=colors)
plt.title('Distribution of the IoT devices-app')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()
