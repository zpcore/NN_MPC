import numpy as np

class data_construct:
	data=[]
	length=0
	count=0
	num_example=0;
	def __init__(self, length, data):
		self.data=data
		self.length=length
		self.num_example=len(data)
		self.count=0

	def clr_count(self):
		self.count=0

	def next_batch(self,batch_x,batch_y,batch_size):
		temp_data=self.data[self.count*batch_size
			:self.count*batch_size+batch_size+self.length-2+1]
		#print temp_data
		for i in range(batch_size):
			for j in range(self.length):
				batch_x+=(temp_data[i+j][0:2,0].tolist())
			batch_y.append(temp_data[i+self.length-1][2,0].tolist())
		self.count+=1#count+=length
		batch_x=np.reshape(batch_x,(batch_size,2*self.length))
		batch_y=np.reshape(batch_y,(batch_size,1))
		#print batch_y
		#in case list go out of range
		if self.count*batch_size+batch_size+self.length-2>=len(self.data):
			self.count=0
		return batch_x, batch_y

	def plot_data(self,x,y,sim_len):
		for i in range(sim_len):
			x+=self.data[self.length+i-1][0:2,0].tolist()
			y.append(self.data[i+self.length-1][2,0].tolist())
		x=np.reshape(x,(sim_len,2))
		y=np.reshape(y,(sim_len,1))
		return x,y

#class property
	@property
	def data(self):
		return self.data

	@property
	def count(self):
		return self.count

	@property
	def num_example(self):
		return self.num_example