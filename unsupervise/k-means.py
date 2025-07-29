import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('../data/data1.csv')

# เลือกจุดศูนย์กลางจาก id=8 และ id=11
center1 = df[df['id'] == 8].iloc[0]
center2 = df[df['id'] == 11].iloc[0]


features = ['freq', 'amount']

# ฟังก์ชันคำนวณระยะห่างแบบยุคลิด
def euclidean(row, center):
    
    return np.sqrt(np.sum((row[features] - center[features]) ** 2))

# ฟังก์ชันหลักสำหรับคำนวณและแสดงผล
def assign_cluster(df, center1, center2):
    # คำนวณระยะห่างจากแต่ละจุดไปยัง center1 และ center2
    df['dist_center1'] = df.apply(lambda row: euclidean(row, center1), axis=1)
    df['dist_center2'] = df.apply(lambda row: euclidean(row, center2), axis=1)

    df['dist_center1'] = df['dist_center1'].round(2) # ปัดเศษให้เหลือ 2 ตำแหน่งทศนิยม
    df['dist_center2'] = df['dist_center2'].round(2)

    # ส่วนนี้แสดงผลระยะห่างของแต่ละ id ไปยังจุดศูนย์กลางทั้งสอง
    print(df[['id', 'dist_center1', 'dist_center2']])

    # ส่วนนี้แสดงผลว่าแต่ละ id ใกล้จุดศูนย์กลางที่ 1 หรือ 2 มากกว่า
    for idx, row in df.iterrows():
        if row['dist_center1'] < row['dist_center2']:
            print(f"id={row['id']} ตำแหน่งที่ 1")
        else:
            print(f"id={row['id']} ตำแหน่งที่ 2")

# ฟังก์ชันสำหรับหาค่าเฉลี่ยของแต่ละกลุ่ม
def calculate_group_means(df):
    df['group'] = df.apply(lambda row: 1 if row['dist_center1'] < row['dist_center2'] else 2, axis=1) #สร้างคอลัมน์กลุ่มตามระยะห่าง
    
    # คำนวณค่าเฉลี่ยของแต่ละกลุ่ม เฉพาะคอลัมน์ features
    mean_group1 = df[df['group'] == 1][features].mean()
    mean_group2 = df[df['group'] == 2][features].mean()
    
    # แสดงผลค่าเฉลี่ยของแต่ละกลุ่ม
    print("\nค่าเฉลี่ยของแต่ละกลุ่ม")
    print("กลุ่มที่ 1 :")
    print(mean_group1)
    print("กลุ่มที่ 2 :")
    print(mean_group2)

# ฟังก์ชันสำหรับคำนวณ WCSS (within-cluster sum of squares)
def calculate_wcss(df, center1, center2):
    # สร้างคอลัมน์ group เพื่อแบ่งกลุ่ม
    df['group'] = df.apply(lambda row: 1 if row['dist_center1'] < row['dist_center2'] else 2, axis=1)
    # คำนวณผลรวมของระยะห่างกำลังสองภายในแต่ละกลุ่ม
    wcss1 = ((df[df['group'] == 1][features] - center1[features]) ** 2).sum().sum()
    wcss2 = ((df[df['group'] == 2][features] - center2[features]) ** 2).sum().sum()
    wcss_total = wcss1 + wcss2
    # แสดงผล WCSS ของแต่ละกลุ่มและรวม
    print("\nWCSS ของแต่ละกลุ่ม")
    print(f"กลุ่มที่ 1 : {wcss1:.2f}")
    print(f"กลุ่มที่ 2 : {wcss2:.2f}")
    print(f"WCSS รวม : {wcss_total:.2f}")
    return wcss_total

# ฟังก์ชันสำหรับแสดงกราฟ scatter plot
def plot_clusters(df, center1, center2):
    # สร้างคอลัมน์ group เพื่อแบ่งกลุ่ม (1 = ใกล้ center1, 2 = ใกล้ center2)
    df['group'] = df.apply(lambda row: 1 if row['dist_center1'] < row['dist_center2'] else 2, axis=1)
    
    # วาดจุดข้อมูลแต่ละกลุ่ม
    plt.scatter(df[df['group'] == 1]['freq'], df[df['group'] == 1]['amount'], color='blue', label='กลุ่มที่ 1')
    plt.scatter(df[df['group'] == 2]['freq'], df[df['group'] == 2]['amount'], color='orange', label='กลุ่มที่ 2')
    
    # วาดจุดศูนย์กลาง
    plt.scatter(center1['freq'], center1['amount'], color='red', marker='X', s=200, label='Center 1 (id=8)')
    plt.scatter(center2['freq'], center2['amount'], color='green', marker='X', s=200, label='Center 2 (id=11)')
    
    # ใส่ชื่อแกนและ legend
    plt.xlabel('freq')
    plt.ylabel('amount')
    plt.title('Scatter Plot แสดงกลุ่มและจุดศูนย์กลาง')
    plt.legend()
    plt.show()

# เรียกใช้ฟังก์ชันหลัก
assign_cluster(df, center1, center2)
calculate_group_means(df)
calculate_wcss(df, center1, center2)
plot_clusters(df, center1, center2)


