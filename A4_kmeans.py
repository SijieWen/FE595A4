from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

## PROBLEM 2
# elbow heuristic
iris = datasets.load_iris()
x = iris.data
y = iris.target

sse = []
for k in range(1,10):
    kmeans = KMeans(n_clusters=k).fit(x)
    sse.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1,10), sse, 'bx-')
plt.title('Elbow method')
plt.xlabel('k')
plt.ylabel("SSE")
plt.show()   
# From the plot, we can recognize that k=3 is the optimal number of clusters.