import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# โหลดข้อมูล
df = pd.read_csv('datapart2.csv', sep=',')
x = df[['weight', 'height']]

# รัน KMeans ครั้งเดียว
model = KMeans(n_clusters=2, random_state=20)
result = model.fit(x)

# แสดงผลการจัดกลุ่ม
print("การจัดกลุ่มสมาชิก:")
for i in range(len(df)):
    group = result.labels_[i]
    weight = df.iloc[i]['weight']
    height = df.iloc[i]['height']
    print(f"สมาชิกคนที่ {i+1}: น้ำหนัก {weight}, ส่วนสูง {height} -> กลุ่ม {group}")

# พล็อตกราฟ
plt.figure(figsize=(8, 6))
plt.scatter(x['weight'], x['height'], c=result.labels_, cmap='viridis', s=60)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
            marker='X', s=200, edgecolor='k', label='Centroids', c='red')
plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('KMeans Clustering Results')
plt.legend()
plt.colorbar(label='Cluster')
plt.show()