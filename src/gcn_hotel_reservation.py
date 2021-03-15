import os
import sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader
from torch_geometric.nn import TopKPooling
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import MSELoss, CrossEntropyLoss
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def meanNorm(X):
	dfx = []
	col = X.columns.tolist()
	
	for ix,r in X.iterrows():
		r = pd.Series(r)
		val = r.values.tolist()
		s = sum(val)

		rw = [i/s for i in val]

		dfx.append(rw)
	X = pd.DataFrame(dfx, columns = col)
	return X

def visualize(h, color):
	z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())

	plt.figure(figsize=(10,10))
	plt.xticks([])
	plt.yticks([])

	plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
	plt.show()


class GCN(torch.nn.Module):
	def __init__(self, hidden_channels):
		super(GCN, self).__init__()
		torch.manual_seed(12345)
		self.conv1 = GCNConv(num_features, hidden_channels)
		self.conv2 = GCNConv(hidden_channels, num_classes)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		x = self.conv1(x, edge_index)
		x = x.relu()
		x = F.dropout(x, p=0.1, training=self.training)
		x = self.conv2(x, edge_index)
#         return F.softmax(x)
		return x


if __name__ == "__main__":
	model = ""
	num_features = 1
	num_classes = 6
	hc = 16
	n_epochs = 20
	batch_size = 128


	n0 = pd.read_csv("../data/tracing-data/hotel-reservation/0_frontend.csv")
	n0['label'] = 0
	n1 = pd.read_csv("../data/tracing-data/hotel-reservation/1_search.csv")
	n1['label'] = 1
	n2 = pd.read_csv("../data/tracing-data/hotel-reservation/2_geo.csv")
	n2['label'] = 2
	n3 = pd.read_csv("../data/tracing-data/hotel-reservation/3_rate.csv")
	n3['label'] = 3
	n4 = pd.read_csv("../data/tracing-data/hotel-reservation/4_profile.csv")
	n4['label'] = 4
	n5 = pd.read_csv("../data/tracing-data/hotel-reservation/5_locale.csv")
	n5['label'] = 5

	dat = pd.concat([n0, n1, n2, n3, n4, n5])
	print(dat.shape)
	# dat = dat.iloc[0:10000, :]

	y = dat['label']
	X = dat[["0_frontend", "1_search", "2_geo", "3_rate", "4_profile", "5_locale"]]

	X = meanNorm(X)
	print(X.shape)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	train_dat = pd.DataFrame(X_train, columns = ["0_frontend", "1_search", "2_geo", "3_rate", "4_profile", "5_locale"])
	train_dat['label'] = y_train
	test_dat = pd.DataFrame(X_test, columns = ["0_frontend", "1_search", "2_geo", "3_rate", "4_profile", "5_locale"])
	test_dat['label'] = y_test

	graphs_train = list()
	count = 0
	for row in train_dat.iterrows():
		r = pd.Series(row)
		r = r[1]
		x = torch.tensor([[r['0_frontend']],
						  [r['1_search']],
						  [r['2_geo']],
						  [r['3_rate']],
						  [r['4_profile']],
						  [r['5_locale']]] ,dtype = torch.float)
		edge_index = torch.tensor([[0, 0, 1, 1, 4],
								   [1, 4, 2, 3, 5]], dtype=torch.long)
		if r['label'] == 0:
			y = torch.tensor([1,0,0,0,0,0], dtype = torch.float)
		elif r['label'] == 1:
			y = torch.tensor([0,1,0,0,0,0], dtype = torch.float)
		elif r['label'] == 2:
			y = torch.tensor([0,0,1,0,0,0], dtype = torch.float)
		elif r['label'] == 3:
			y = torch.tensor([0,0,0,1,0,0], dtype = torch.float)
		elif r['label'] == 4:
			y = torch.tensor([0,0,0,0,1,0], dtype = torch.float)
		elif r['label'] == 5:
			y = torch.tensor([0,0,0,0,0,1], dtype = torch.float)
		else:
			print("unknown label encountered")
			break
		graphs_train.append(Data(x = x, edge_index = edge_index,y = y))
		count = count + 1
	
	
	graphs_test = list()
	for row in test_dat.iterrows():
		r = pd.Series(row)
		r = r[1]
		x = torch.tensor([[r['0_frontend']],
						  [r['1_search']],
						  [r['2_geo']],
						  [r['3_rate']],
						  [r['4_profile']],
						  [r['5_locale']]] ,dtype = torch.float)
		edge_index = torch.tensor([[0, 0, 1, 1, 4],
								   [1, 4, 2, 3, 5]], dtype=torch.long)
		if r['label'] == 0:
			y = torch.tensor([1,0,0,0,0,0], dtype = torch.float)
		elif r['label'] == 1:
			y = torch.tensor([0,1,0,0,0,0], dtype = torch.float)
		elif r['label'] == 2:
			y = torch.tensor([0,0,1,0,0,0], dtype = torch.float)
		elif r['label'] == 3:
			y = torch.tensor([0,0,0,1,0,0], dtype = torch.float)
		elif r['label'] == 4:
			y = torch.tensor([0,0,0,0,1,0], dtype = torch.float)
		elif r['label'] == 5:
			y = torch.tensor([0,0,0,0,0,1], dtype = torch.float)
		else:
			print("unknown label encountered")
			break
		graphs_test.append(Data(x = x, edge_index = edge_index,y = y))
		count = count + 1


	print(f'Number of graphs: {len(graphs_train)}')

	data = graphs_train[0]  # Get the first graph object.

	print()
	print(data)
	print('===========================================================================================================')

	# Gather some statistics about the graph.
	print(f'Number of nodes: {data.num_nodes}')
	print(f'Number of edges: {data.num_edges}')
	print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')

		# Initialize scaling and dataloader

	scaler = StandardScaler()
	loader = DataLoader(graphs_train, batch_size = batch_size)

	# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device = torch.device('cpu')
	for d in range(1):
		data = next(iter(loader))
		data = data.to(device)
		print(data)
	model = GCN(hidden_channels = hc).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
	loss_func = CrossEntropyLoss()   

	print("total batches:", len(loader))
	print("cpu or gpu:", device)


	count = 0
	for data in loader:
		data = data.to(device)
		count = count + 1
		print(count)
		for epochs in range(n_epochs):
			optimizer.zero_grad()                                                   
			out = model(data) 
			loss = loss_func(out, data.y.long())
			loss.backward()                                                         
			optimizer.step()
			print(f'Epoch: {epochs:03d}, Loss: {loss:.4f}')


	num_pred = 0
	count_pred = 0
	for tg in graphs_test:
	    out = model(tg)
	    pred = out.argmax(dim=1)

	    if torch.all(torch.eq(data.y, pred)):
	        num_pred += 1
	    count_pred +=1

	print("Accuracy: ", num_pred/count_pred)