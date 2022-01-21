import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

torch.manual_seed(0);

test1 = np.load('test1.npy')
test2 = np.load('test2.npy')
test3 = np.load('test3.npy')
test4 = np.load('test4.npy')
test5 = np.load('test5.npy')
test6 = np.load('test6.npy')
test7 = np.load('test7.npy')
test8 = np.load('test8.npy')
test9 = np.load('test9.npy')
test10 = np.load('test10.npy')

df1 =test1
x1, y1 = np.split(df1, [10] ,axis=1)
X_reduced1 = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(x1)
print(X_reduced1.shape)
plt.scatter(X_reduced1[:, 0], X_reduced1[:, 1] ,s=40, c='red')

df2 =test2
x2, y2 = np.split(df2, [10] ,axis=1)
X_reduced2 = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(x2)
print(X_reduced2.shape)
plt.scatter(X_reduced2[:, 0], X_reduced2[:, 1] ,s=5, c='blue')

df3 =test3
x3, y3 = np.split(df3, [10] ,axis=1)
X_reduced3 = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(x3)
print(X_reduced3.shape)
plt.scatter(X_reduced3[:, 0], X_reduced3[:, 1] ,s=5, c='green')

df4 =test4
x4, y4 = np.split(df4, [10] ,axis=1)
X_reduced4 = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(x4)
print(X_reduced4.shape)
plt.scatter(X_reduced4[:, 0], X_reduced4[:, 1] ,s=5, c='magenta')

df5 =test5
x5, y5 = np.split(df5, [10] ,axis=1)
X_reduced5 = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(x5)
print(X_reduced5.shape)
plt.scatter(X_reduced5[:, 0], X_reduced5[:, 1] ,s=5, c='lime')

df6 =test6
x6, y6 = np.split(df6, [10] ,axis=1)
X_reduced6 = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(x6)
print(X_reduced6.shape)
plt.scatter(X_reduced6[:, 0], X_reduced6[:, 1] ,s=5, c='pink')

df7 =test7
x7, y7 = np.split(df7, [10] ,axis=1)
X_reduced7 = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(x7)
print(X_reduced7.shape)
plt.scatter(X_reduced7[:, 0], X_reduced7[:, 1] ,s=5, c='navy')

df8 =test8
x8, y8 = np.split(df8, [10] ,axis=1)
X_reduced8 = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(x8)
print(X_reduced8.shape)
plt.scatter(X_reduced8[:, 0], X_reduced8[:, 1] ,s=5, c='orange')

df9 =test9
x9, y9 = np.split(df9, [10] ,axis=1)
X_reduced9 = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(x9)
print(X_reduced9.shape)
plt.scatter(X_reduced9[:, 0], X_reduced9[:, 1] ,s=5, c='darkgoldenrod')

df10 =test10
x10, y10 = np.split(df10, [10] ,axis=1)
X_reduced10 = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(x10)
print(X_reduced10.shape)
plt.scatter(X_reduced10[:, 0], X_reduced10[:, 1] ,s=5, c='grey')

plt.title("t-SNE", fontsize=30)
plt.savefig('T-SNE.png')
plt.show()