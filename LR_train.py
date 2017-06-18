
import numpy

def LoadData(file_path):
	input_data = numpy.loadtxt(file_path)

	train_x = input_data[-20:, 0:2]*10
	train_y = input_data[-20:, 2:3]
	#print(train_x)

	test_x = input_data[-20:, 0:2]*10
	test_y = input_data[-20:, 2]

	return train_x, train_y, test_x, test_y

def ComputeModel(train_x, train_y, opt):
	numSample, numFeature = train_x.shape
	alpha = opt['alpha']
	maxIter = opt['maxIter']
	weights = numpy.ones((numFeature), dtype = float)
	
	for k in range(maxIter):
		for i in range(numSample):
			output = sigmod(numpy.dot(train_x[i, :], weights))
			error = train_y[i] - output
			weights = weights + alpha * train_x[i, :] * error
			#print(train_x[i, :].transpose())
			#print(numpy.dot(train_x.transpose(), error))

	return weights

def sigmod(z):
	#print(z)
	return 1.0 / (1 + numpy.exp(-z)) 

def testModel(test_x, test_y, weights):
	numSample = test_x.shape[0]
	for k in range(numSample):
		output = sigmod(numpy.dot(test_x[k, :], weights))
		#if output < 0.5:
			#output = 0
		#else:
			#output = 1
		print(test_x[k], "     ", output, "     ", test_y[k])
		#print(test_y[k])

if __name__=="__main__":
	train_file = "./data1.txt"
	train_x, train_y, test_x, test_y = LoadData(train_file)
	print(test_y)
	opts = {'alpha': 0.01, 'maxIter': 50}
	weights = ComputeModel(train_x, train_y, opts)
	print weights
	testModel(test_x, test_y,weights)
	#print train_y