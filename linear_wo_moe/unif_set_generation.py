import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture



def point_god(num,cov, theta,shift, x, y):
    thetaa = np.random.uniform(0.0, 2.0*np.pi, num)
    zero_to_one = np.random.uniform(0.0, 1.0, num)

    tmp_1 = circle_radius * np.sqrt(zero_to_one) * np.cos(thetaa)
    tmp_2 = circle_radius * np.sqrt(zero_to_one) * np.sin(thetaa)
    tmp = np.row_stack((tmp_1,tmp_2))
    tmp = np.matmul(cov,np.array(tmp))
    x_tmp, y_tmp = np.einsum("ab,bc->ac", [[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]], tmp)
    x = np.append(x,x_tmp+shift[0])
    y = np.append(y,y_tmp+shift[1])
    return x, y


circle_radius = 1
x, y = [[],[]]


#cluster 1
num = 600
shift = [-2,5]
cov = [[7,0],[0,1]]
theta = 1/15 *np.pi
x, y = point_god(num,cov, theta,shift, x, y)

#cluster 2
num = 600
shift = [9,2]
cov = [[10,0],[0,1]]
theta = 0 *np.pi
x, y = point_god(num,cov, theta,shift, x, y)

#cluster 3
num = 600
shift = [17,5]
cov = [[2,0],[0,1]]
theta = 1/5 *np.pi
x, y = point_god(num,cov, theta,shift, x, y)

x = x.reshape((-1, 1))
y = y.reshape((-1, 1))

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_x.fit(x)
scaler_y.fit(y)
x = scaler_x.transform(x)
y = scaler_y.transform(y)


# #kmeans
# kmeans = KMeans(n_clusters=3).fit(x)
# membership = kmeans.predict(x)
#GMM
gm = GaussianMixture(n_components=3).fit(x)
membership = gm.predict(x)

plt.scatter(x, y, c=membership)
plt.axis('equal')

residual = np.array([])
for i in Counter(membership).keys():
    idx = membership==i
    model = LinearRegression()
    model.fit(x[idx], y[idx])
    y_pred = model.predict(x[idx])
    plt.plot(x[idx], y_pred,color='k') 
    residual = np.append(residual, np.abs(y_pred - y[idx]))
    with open('regression'+str(i)+'.pickle', 'wb') as f:
        pickle.dump([model.coef_[0][0],model.intercept_[0]], f, pickle.HIGHEST_PROTOCOL)
plt.title(np.mean(residual))
plt.savefig('my_plot.png')
plt.close()


with open('2d_syn_data.pickle', 'wb') as f:
    pickle.dump({'x':x,'y':y}, f, pickle.HIGHEST_PROTOCOL)



# pd.DataFrame(x).to_csv("2d_syn_x.csv", index=False)
# pd.DataFrame(y).to_csv("2d_syn_y.csv", index=False)


