import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

cols = ['x', 'y', 'label']
data = pd.read_csv('data1.txt', names=cols, header=None)


def distance(x1, y1, c_x, c_y):
    d = ( (x1-c_x)**2 ) + ( (y1-c_y)**2 )
    return d

def assign(x1, y1, c_dist, f_d, kpts):
    i = 0
    center_points = []
    for each in c_dist:
        j = 0
        temp = [] 
        for m in kpts:
            if c_dist[i][j] == f_d[i]:
                temp.append(x1[i])
                temp.append(y1[i])
                temp.append(f_d[i])
                temp.append(j+1)
                center_points.append(temp)
            j = j + 1
        i = i + 1
    return center_points

def average(pts, k):
    x_sum = 0
    y_sum = 0
    count = 0
    dis_sum = 0
    temp = []
    i = 0
    for each in pts:
        if pts[i][3] == k:
            x_sum = x_sum + pts[i][0]
            y_sum = y_sum + pts[i][1]
            dis_sum = dis_sum + pts[i][2]
            count = count + 1
            i = i + 1
        else:
            i = i + 1

    x_ave = float(x_sum) / float(count)
    y_ave = float(y_sum) / float(count)
    dis_ave = float(dis_sum) / float(count)
    temp.append(x_ave)
    temp.append(y_ave)
    temp.append(dis_ave)
    temp.append(k)
    return temp

def cluster (X, Y, k_points):
    print("In Cluster Function")

    # took center points apart by x, y, and label
    centerpts_labels = []
    center_x = []
    center_y = []
    i = 0
    for cen in k_points:
        center_x.append(k_points[i][0])
        center_y.append(k_points[i][1])
        centerpts_labels.append(k_points[i][2])
        i = i + 1

    #calculate distance for each coordinate and their min
    final_distances = []
    coor_dist = []
    j = 0
    for x in X:
        d_list = [] 
        m = 0
        for c in centerpts_labels:
            d_list.append(distance(X[j], Y[j], center_x[m], center_y[m]))
            m = m + 1
        j = j + 1
        coor_dist.append(d_list)
        d_min = min(d_list)
        final_distances.append(d_min)

    # assign a center point label (1,2,etc.) and sort by label
    newCoor = []
    newCoor = assign(X, Y, coor_dist, final_distances, k_points)
    newCoor.sort(key = lambda i: i[3])
    
    # find average based on how many center points/labels there are. 
    # for example, if there are 2 center points, it will find the average of the x, y, and distance of label 1 and label 2 respectively
    kpts = []
    j = 0
    for c in k_points:
        kpts.append(j+1)
        j = j + 1
    average_list = []
    for each in kpts:
        average_list.append(average(newCoor, each))

    pre_k_points = k_points
    print("Previous k points: ", pre_k_points)

    # change center points x and y
    i = 0
    cur_k_points = k_points
    for each in pre_k_points:
        cur_k_points[i][0] = average_list[i][0]
        cur_k_points[i][1] = average_list[i][1]
        i = i + 1
    print ("New Center Points (k_points): ", cur_k_points)

    # set x and y again
    X = []
    Y = []
    i = 0
    for each in newCoor:
        X.append(newCoor[i][0])
        Y.append(newCoor[i][1])
        i = i + 1
    return X, Y, newCoor, k_points

def k_means(x_min, y_min, x_max, y_max, num):
    center_point_list = []
    temp = []
    for i in range(num):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        temp.append(x)
        temp.append(y)
        temp.append(i+1)
        center_point_list.append(temp)
        temp = []
    return center_point_list



coordinates = data.iloc[:,:-1].values
labels = data.iloc[:,-1].values
x = []
y = []
idx = 0
for coor in coordinates:
    x.append(coordinates[idx][0])
    y.append(coordinates[idx][1])
    idx = idx + 1

# get min and max of x and y
xmin = min(x)
ymin = min(y)
xmax = max(x)
ymax = max(y)

# create center points based on min and max
n = 2
kpts = k_means(xmin, ymin, xmax, ymax, n)

print ("Beginning Center Points:", kpts)
print("")

# call function
i = 0
while i < 6:
    newx, newy, new_Coor, kpts = cluster(x, y, kpts)
    i = i + 1
    print("")


# seperate x and y from centroids
i = 0
cen_x = []
cen_y = []
for each in kpts:
    cen_x.append(kpts[i][0])
    cen_y.append(kpts[i][1])
    i = i + 1

# percentage
#i = 0
#corr = 0
#for c in labels:
#    if labels[i] == new_Coor[i][3]:
#        corr = corr + 1
#print("Percentage: ", corr/idx)

#list1 = [x for x in mylist if x in goodvals]
#list2  = [x for x in mylist if x not in goodvals]

i = 0
for e in new_Coor:
    if new_Coor[i][3] == 1:
        plt.scatter(new_Coor[i][0], new_Coor[i][1], color="red")
    elif new_Coor[i][3] == 2:
        plt.scatter(new_Coor[i][0], new_Coor[i][1], color="green")
    else:
        plt.scatter(new_Coor[i][0], new_Coor[i][1], color= "blue")
    i = i + 1
# plot
#plt.scatter(newx, newy)
plt.scatter(cen_x, cen_y, color="black")
plt.show()
