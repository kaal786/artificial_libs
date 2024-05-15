
def mse(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return mean_error


def mae(actual,predicted):
	abs_error=0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		abs_error+=abs(prediction_error)
	mean_error=abs_error/float(len(actual))
	return mean_error
