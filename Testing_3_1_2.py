import sys
import random
from random import randint
import cross_validation_splitter
def read_data(fname):
	temp_data=[]
	try:
		with open(fname) as docs:
			for line in docs:
				line=map(float,line.split())
				temp_data.append(line)

	except Exception,e:
		raise e
		print "File Not Found, program will exit"
		exit()

	return temp_data

def populate_weights(max_features):
	weights=[randint(-1,1) for i in range(max_features)]
	return weights

def perceptron(data,labels,weights,epoch,rate,C):
	j=0
	x=0
	t=0
	gamma_t=rate
	train_error=0
	temp=0
	while j<epoch:
		for i in range(len(data)):
			t=i+1
			gamma_t=float(rate)/(1+((rate*t)/C))
			dot_product=map(lambda x,y:x*y,weights,data[i])
			derived_label=reduce(lambda x,y:x+y,dot_product)
			actual_label=labels[i]
			if derived_label*actual_label<=1:
				train_error+=1
				update=map(lambda x:x*C*gamma_t*actual_label,data[i])
				weights=map(lambda x,y: x+y,map(lambda x:x*(1-gamma_t),weights),update)
			else:
				temp+=1
				gamma_t=1-gamma_t
				weights=map(lambda x:x*gamma_t,weights)
		j+=1

	return weights

def calculate_accuracy(weights,data,labels,hits,miss):
	TP=FP=FN=0
	for i in range(len(data)):
		dot_product=map(lambda x,y:x*y,weights,data[i])
		derived_label=reduce(lambda x,y:x+y,dot_product)
		actual_label=labels[i]
		if derived_label>0 and actual_label<0:
			FP+=1
		if derived_label<0 and actual_label>0:
			FN+=1
		if derived_label>0 and actual_label>0:
			TP+=1
		if derived_label*actual_label>0:
			hits+=1
		else:
			miss+=1
	return float(hits)/(hits+miss),TP,FP,FN

"""def cross_validation(k):
	for i in range(k):
		file_name="data/handwriting/training_0%d.data """

def main():
	k=5
	cross_validation_splitter.split(k)
	epoch_list=[3,5,20]
	C=[0.5, 0.25, .125, 0.7, 0.9, 1]
	learning_rate=[0.01, 0.50, 0.70, 0.90]
	accuracies_final=[]
	cross_validation_splitter.cross_validate(k,epoch_list, C, learning_rate)
	C=0.5
	learning_rate=[0.70]
	epoch_list=[3]
	#learning_rate=[]


	file_name="data/madelon/madelon_train.data"
	temp_data =read_data(file_name)
	file_name="data/madelon/madelon_train.labels"
	labels=read_data(file_name)

	combined_data=[]
	for i in range(len(temp_data)):
		combined_data.append(labels[i]+temp_data[i])
	
	try:
		file_name=sys.argv[2]
	except Exception:
		file_name="data/madelon/madelon_test.data"

	temp_test_data =read_data(file_name)
	file_name="data/madelon/madelon_test.labels"
	test_labels =read_data(file_name)
	combined_test_data=[]
	for i in range(len(temp_test_data)):
		combined_test_data.append(test_labels[i]+temp_test_data[i])

	test_labels=[i[0] for i in combined_test_data]
	test_data=[[1]+i[1:] for i in combined_test_data]

	weight_initialize=populate_weights(len(combined_data[0])-1)
	initialized_bias=randint(-1,1)
	
	for epoch in epoch_list:
		random.shuffle(combined_data)
		labels=[i[0] for i in combined_data]
		data=[[1]+i[1:] for i in combined_data]
		
		for rate in learning_rate:
			weights=[initialized_bias]+weight_initialize
			final_weight_vector=perceptron(data,labels,weights,epoch,rate,C)
		
			if len(test_data[0])>len(data[0]):
				difference=len(test_data[0])-len(data[0])
				final_weight_vector=final_weight_vector+[0]*difference
		
			accuracy,TP,FP,FN=calculate_accuracy(final_weight_vector,test_data,test_labels,0,0)
			try:
				p=float(TP)/(TP+FP)
				r=float(TP)/(TP+FN)
				F1=float(2*p*r)/(p+r)
				accuracies_final.append([accuracy,rate,epoch,p,r,F1])
				print "Accuracy found:",accuracy*100,"for rate:",rate,"C",C,"with epoch:",epoch,"p:",p,"r:",r,"F1:",F1
			except Exception:
				pass
			weights=[]
	try:
		result=max(accuracies_final)

		print "\n"
		print result[0],"max accuracy for learning rate",result[1],"C",C,"epoch",result[2],"with p:",result[3],"r:",result[4],"F1:",result[5]
	except Exception:
		pass
main()
