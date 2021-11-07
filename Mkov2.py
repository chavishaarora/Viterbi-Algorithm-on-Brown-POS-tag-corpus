import nltk
import re
import pandas as pd
from IPython.display import display
from sklearn.model_selection import KFold
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np
from nltk.stem import 	WordNetLemmatizer
from stemming.porter2 import stem
from nltk.stem import PorterStemmer
from sklearn.model_selection import KFold
import spacy
import inflect 
p = inflect.engine()
nlp=spacy.load('en_core_web_sm')

filex = open("Brown_train.txt").read()
#split to sentences
sentences=filex.splitlines()

def getting_variables_trainingset(sentences,tag_word_list,word_tag_list,wordslist):
	tags=[]
	tag_map=dict()
	uniquetagset=set()
	for i in sentences:
		word_tag=i.split()
		for k in word_tag:
			word=''
			tag=''
			if(k.find('_')!= -1):
				word,tag=k.split('_')
				tag=tag[:2] #taking only the first two characters
				tags.append(tag)
				uniquetagset.add(tag)
				tag_word_list.append([tag,word])
				word_tag_list.append([word,tag])
				wordslist.append(word)
	tags=[]
	for tag,word in tag_word_list:
		tags.append(tag)

	tags_words_dict={}
	for i in tag_word_list:
		tags_words_dict.setdefault(i[0],[]).append(i[1])
	words_tags_dict={}
	for i in tag_word_list:
		words_tags_dict.setdefault(i[1],[]).append(i[0])

	tagset_list=list(uniquetagset)
	for i in range(len(tagset_list)):
		tag_map[tagset_list[i]]=i
	return  tags_words_dict,words_tags_dict,tags,tag_map
    #Calculating Emisiion probability of new being adjective is:

def emission_prob(word,tag,words_tags_dict,Distincttags,tags_words_dict):
	count=0
	totallen=len(tags_words_dict.get(tag))
	listoftags=words_tags_dict.get(word)
	totallen=len(listoftags)
	count=0
	for i in listoftags:
		if(i==tag):
			count=count+1
	return((count+1)/(totallen +len(Distincttags)))
#print(count/totallen)

#Calculating Transition probabilities

def transmission_prob(tag1,tag2,tags):
	c1=0
	c2=0
	for k in range(len(tags)-1):
		if (tags[k]==tag1):
			c1=c1+1
			if(tags[k+1]==tag2):
				c2=c2+1
	return(c2/c1)

def getdistincttagspairs(Distincttags):
	Distincttags_pairs=[]
	for i in Distincttags:
		for j in Distincttags:
			k=i+"_"+j
			Distincttags_pairs.append(k)
	return Distincttags_pairs

def gettag_mapforpairs(Distincttags_pairs):
	tag_map_pairs=dict()
	for i in range(len(Distincttags_pairs)):
		tag_map_pairs[Distincttags_pairs[i]]=i
	return tag_map_pairs


def Mkov2(Distincttags_pairs,Distincttags,tags,tag_map,tag_map_pairs):
	tags_matrix=np.zeros((len(Distincttags_pairs),len(Distincttags)),dtype='float32')
	for i in range(len(tags)-2):
		tag1=tags[i]
		tag2=tags[i+1]
		tag_pair=tag1+"_"+tag2
		index=tag_map_pairs.get(tag_pair)
		tags_matrix[index,tag_map.get(tags[i+2])]+=1
	for i in range (len(Distincttags_pairs)):
		sum=0
		for k in range(2):
			for j in range(len(Distincttags)):
				if (k==0):
					sum=sum+tags_matrix[i,j]
				else:
					if(sum==0):
						tags_matrix[i,j]=0
					else:
						tags_matrix[i,j]=tags_matrix[i,j]/sum
	trans_df=pd.DataFrame(tags_matrix, columns=Distincttags, index=Distincttags_pairs)
	return trans_df


def getemdf(Distincttags,tags,words,DistinctWords,words_tags_dict,tags_words_dict): #list of all tags in training set, list of all words in training set
	emissionmatrix=np.zeros((len(Distincttags),len(DistinctWords)),dtype='float32')
	for i,tag in enumerate(list(Distincttags)):
		for j,key in enumerate(list(DistinctWords)):
			emissionmatrix[i,j]=emission_prob(key,tag,words_tags_dict,Distincttags,tags_words_dict)
	emission_df=pd.DataFrame(emissionmatrix, columns = list(DistinctWords), index=list(Distincttags))
	return emission_df

def viterbi(sentence,Distincttags,words_tags_dict,tags,tags_df,emission_df):
	state=[]
	for key in range(len(sentence)):
		p = [] 
		count=0
		tempvariable=0
		for tag in Distincttags:
			if (key==0 or key ==1):
				transition_p=1
			else:
				transition_p=tags_df.loc[state[-2]+"_"+state[-1],tag]
				count=count+1
			# compute emission and state probabilities
			emission_p=emission_df.loc[tag,sentence[key]]
			state_probability=emission_p*transition_p
			p.append(state_probability)
			tempvariable+=1
			#print(tempvariable)
		pmax=max(p)
		state_max = Distincttags[p.index(pmax)]
		#print(state_max)
		state.append(state_max)
		#print(state)
	return(list(state))
	#return (list(state))
def findindex(tag,Distingtags):
	count=-1
	for tags in Distingtags:
		count=count+1
		if(tag==tags):
			break
	return count

#print(viterbi(sentencetesting))
Avg_F1score=0
Avg_Precision=0
Avg_Recall=0
data=np.array(sentences)
kfold = KFold(3, True, 1)
j=0
for train, test in kfold.split(data):
	#to find the indivdual accuracy for each training and testing data 
	totalwords=0
	sentence_train=data[train]
	testing=data[test]
	tag_word_list_train=[]
	tags_train=[]
	word_tag_list_train=[]
	tags_words_dict_train={}
	words_tags_dict_train={}
	wordslist_train=[]
	tag_map_train=dict()
	tags_words_dict_train,words_tags_dict_train,tags_train,tag_map_train=getting_variables_trainingset(sentences,tag_word_list_train,word_tag_list_train,wordslist_train)
		#print(len(tags))
	correctcountoftags=0
	totaltagsintestingdata=len(tags_train)
	Distincttag_train=set(tags_train)
	Distincttags_train=[]
	for i in Distincttag_train:
		Distincttags_train.append(i)
		#print(len(tag_word_list))
	DistinctWord_train=set(wordslist_train)
	DistinctWords_train=[]
	for g in DistinctWord_train:
		DistinctWords_train.append(g)
		#getting data frame
	emiss_df=getemdf(Distincttags_train,tags_train,wordslist_train,DistinctWords_train,words_tags_dict_train,tags_words_dict_train)
	Distincttags_pairs=getdistincttagspairs(Distincttags_train)
	tag_map_pairs_train=gettag_mapforpairs(Distincttags_pairs)
	#print("Attemting new way")
	#tags_df_train=gettrasmissiondf(Distincttags_train,tags_train,tag_map_train)
	tags_df_train=Mkov2(Distincttags_pairs,Distincttags_train,tags_train,tag_map_train,tag_map_pairs_train)
	#display(tags_df_train)

	accuracycount=0
	lensentences=0
	finalthing=[]
	tempcountignorelater=0
	Confusion_matrix=np.zeros((len(Distincttags_train)+1,len(Distincttags_train)+1),dtype='float32')
	#Confusion_matrix=
	for sent in testing:
		finalthing=[]
		tempcountignorelater=tempcountignorelater+1
		finalsentence=[] #for each sentence IN TESTING DATA, finalsentence splits the setence with word_tag pairs to just words
		finaltags=[] #for each sentence in Testing DATA,finaltags represents the list of all tags in order for a sentences
		word_tag=sent.split() 
		for x in word_tag:
			word=''
			tag=''
			if(x.find('_')!= -1):
				word,tag=x.split('_')
				lensentences=lensentences+1
				tag=tag[:2] #taking only the first two characters
				finalsentence.append(word)
				finaltags.append(tag)
		finalthing.append(viterbi(finalsentence,Distincttags_train,words_tags_dict_train,tags_train,tags_df_train,emiss_df))
		totalwords=totalwords+len(finaltags)
		if(len(finaltags)==len(finalthing[0])):
			for d in range(len(finaltags)):
				if(finaltags[d]==finalthing[0][d]):
					index=findindex(finaltags[d],Distincttags_train)
					Confusion_matrix[index][index]=Confusion_matrix[index][index]+1
					accuracycount=accuracycount+1
				else:
					#columns represent true tags
					#rows represent predicted tags
					column=findindex(finaltags[d],Distincttags_train)
					row=findindex(finalthing[0][d],Distincttags_train)
					Confusion_matrix[row][column]+=1

		#taking the sum of all columns in the last column
		for i in range(len(Distincttags_train)):
			columnwise_sum=0
			for j in range(len(Distincttags_train)):
				columnwise_sum=columnwise_sum+Confusion_matrix[i][j]
			Confusion_matrix[i][len(Distincttags_train)]=columnwise_sum

		#taking the sum of all rows in the last row
		for i in range(len(Distincttags_train)):
			rowwise_sum=0
			for j in range(len(Distincttags_train)):
				rowwise_sum=rowwise_sum+Confusion_matrix[j][i]
			Confusion_matrix[len(Distincttags_train)][i]=rowwise_sum
	Tags_labels_Df=[]
	for g,key in enumerate(Distincttags_train):
		Tags_labels_Df.append(key)
	Tags_labels_Df.append("sum")
	#print(Confusion_matrix)
	Confusion_df=pd.DataFrame(Confusion_matrix,columns = list(Tags_labels_Df), index=list(Tags_labels_Df))
	print("Confusion Matrix :")
	display(Confusion_df)
	print("Accuracy :" ,(accuracycount/totalwords)*100)

	#Calculating Precision,Recall,F1-score for each individual tag
	#Starting with finding 4 variable for each tag, TP,TN,FP,FN
	#Creating a dictionary with unique tags as the keys
	print("Precision,Recall and F1-Score for each tag in the respective order are :")
	Dict_Unique_Tags={}
	for tag in Distincttags_train:
		Dict_Unique_Tags.setdefault(tag,[])
	#Scanning through the Confusion matrix and getting the true positives for each tag which is found at [i,i] for tag i
	TotalF1score=0
	Totalprecision=0
	Totalrecall=0
	for j in range(len(Confusion_matrix)-1):
		tag=Distincttags_train[j] #the tag found
			#True positive for this tag is found at
		Precision=0
		Recall=0
		F1score=0 
		TP=Confusion_matrix[j][j]
		FN=Confusion_matrix[len(Confusion_matrix)-1][j]-TP
		FP=Confusion_matrix[j][len(Confusion_matrix)-1]-TP
		if ((TP+FP)==0):
			Precision=0
		else:
			Precision=TP/(TP+FP)
		if ((TP+FN)==0):
			Recall=0
		else:
			Recall=TP/(TP+FN)
		#Precision=TP/TP+FP
		#Recall=TP/TP+FN
		if(Precision+Recall==0):
			F1score=0
		else:
			F1score=(2*Precision*Recall)/(Precision+Recall)
		List=Dict_Unique_Tags.get(tag)
		List.append(Precision)
		List.append(Recall)
		List.append(F1score)
		TotalF1score=TotalF1score+F1score
		Totalprecision=Totalprecision+Precision
		Totalrecall=Totalrecall+Recall
		Dict_Unique_Tags.setdefault(tag,List)
	#break
	print(Dict_Unique_Tags)
	Avg_Precision=Avg_Precision+(Totalprecision/len(Dict_Unique_Tags))
	Avg_Recall=Avg_Recall+(Totalrecall/len(Dict_Unique_Tags))
	Avg_F1score=Avg_F1score+(TotalF1score/len(Dict_Unique_Tags))
	#for key in Dict_Unique_Tags:
	print("Average F1-Score of this fold:", TotalF1score/len(Dict_Unique_Tags))
	print("Average Precision of this fold :",Totalprecision/len(Dict_Unique_Tags))
	print("Average Recall : of this fold",Totalrecall/len(Dict_Unique_Tags))

print("Average Precision of the three folds : ", Avg_Precision/3)
print("Average Recall of the three folds : ", Avg_Recall/3)
print("Average F1score of the three folds : ", Avg_F1score/3)
