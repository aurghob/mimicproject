#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import pandas as pd
import gensim.models as g
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential,Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import RMSprop
from keras.metrics import categorical_accuracy, mean_squared_error, mean_absolute_error, logcosh
from keras_contrib.utils import save_load_utils
#import jellyfish as jf
import stringdist
import ast
import ast
import numpy as np
import time
import pickle
import sys
import re
path = "C:\\Masters Studies\\NLP\\Project\\12-22 batches results"

numOfTrainRecords = 50
numOfTestRecords = 10

numDoc = 0
num_docs = 600000
max_len = 30
vector_dim = 100


allClinicalNotes_header_text=pd.read_csv(path+'Train_Test_SectionHeader_text.csv')
allClinicalNotes_header_text=allClinicalNotes_header_text.drop(['Unnamed: 0'], axis=1)
newColumns = list(allClinicalNotes_header_text.columns)
newColumns[0]= "FileName"
allClinicalNotes_header_text.columns= newColumns

trainSetDF_FULL=pd.read_csv(path+'trainSetDF.csv')
#trainSetDF_FULL.columns = ['0','Index','Order']
trainSetDF_FULL = trainSetDF_FULL.drop(['Unnamed: 0'], axis=1)
train_Dict=trainSetDF_FULL.head(numOfTrainRecords).set_index('FileName').T.to_dict('list')

trainSetDF_FULL_TEXT = allClinicalNotes_header_text[allClinicalNotes_header_text['FileName'].isin(train_Dict.keys())]
train_TEXT_Dict=trainSetDF_FULL_TEXT.set_index('FileName').T.to_dict('dict')
#train_TEXT_Dict.get('DischargeSummary4709920033')
del trainSetDF_FULL
del trainSetDF_FULL_TEXT

testSetDF=pd.read_csv(path+'testSetDF.csv')
#testSetDF.columns = ['0','Index','Order']
testSetDF = testSetDF.drop(['Unnamed: 0'], axis=1)
test_Dict=testSetDF.head(numOfTestRecords).set_index('FileName').T.to_dict('list')
#test_Dict=testSetDF.set_index('FileName').T.to_dict('list')
#
testSetDF_FULL_TEXT = allClinicalNotes_header_text[allClinicalNotes_header_text['FileName'].isin(test_Dict.keys())]
test_TEXT_Dict=testSetDF_FULL_TEXT.head(numOfTestRecords).set_index('FileName').T.to_dict('dict')

del testSetDF
del testSetDF_FULL_TEXT

del allClinicalNotes_header_text

print("Loading Doc2Vec...")
model_doc2vec= g.Doc2Vec.load(path+"model_dbow0_pretrainedpubWikiPMC_trained_vd100_full-36.bin")
print("Doc2Vec, Loaded")

sectionHeaders = ['history_present_illness','activity','discharge_condition','past_medical_history','chief_complaint','follow_up','discharge_instructions','allergies_and_adverse_reactions','admission_date','hospital_course','findings','review_of_systems','family_history','laboratory_and_radiology_data','diagnoses','physical_examination','assessment_and_plan','social_history','medications','history_source','assessment','discharge medications','Unknown']

t1=time.time()

def splitAndRemoveEmptys(text):
    splitText = re.split('; |, |\*|\n|-|:|\s',text)
    splitText[:] = [x for x in splitText if x != '']
    return splitText

def getEmbedding(embeddingFlag,splitText,fileName_SectionHeader_Tag):
    if embeddingFlag == "Infer":
        return model_doc2vec.infer_vector(splitText, steps=20, alpha=0.025)
    elif embeddingFlag == "NotInfer":
        return list(model_doc2vec.docvecs[fileName_SectionHeader_Tag])
        

embedding_matrix = np.zeros((num_docs, vector_dim))
def sectionHeaderSimilarities(oneNote_TEXT_Dict,fileName,sectionHeader_Tag,embeddingFlag):
    global embedding_matrix
    global numDoc
    fileName_SectionHeader_Tag = fileName+'_'+sectionHeader_Tag
    numDoc=numDoc+1
#    SectionSim=[model_doc2vec.docvecs.similarity(sectionHeader,fileName_SectionHeader_Tag) for sectionHeader in sectionHeaders]
    sectionText=oneNote_TEXT_Dict.get(sectionHeader_Tag)
    sectionTextSpacesRemoved = re.sub(' +',' ',sectionText.lower())
    if sectionHeader_Tag != 'Unknown':
        splitText = splitAndRemoveEmptys(sectionTextSpacesRemoved)
        vec = getEmbedding(embeddingFlag,splitText,fileName_SectionHeader_Tag)
        embedding_matrix[numDoc]=vec
    else:
        unknText = sectionTextSpacesRemoved.split('**Unknown**')
        if (unknText[-1]==None or unknText[-1]==""):
            del unknText[-1]
        unknownSectionCounter = 0
        for unknownText in unknText:
            splitText = splitAndRemoveEmptys(unknownText)
            vec = getEmbedding(embeddingFlag,splitText,fileName_SectionHeader_Tag+str(unknownSectionCounter+1))
            embedding_matrix[numDoc]=vec
            numDoc=numDoc+1
            unknownSectionCounter=unknownSectionCounter+1
#    embedding_matrix[numDoc]=list(model_doc2vec.docvecs[fileName_SectionHeader_Tag])
    return numDoc

def createListOfClinicalNotes_Doc2vec_And_SectionSeq(trainOrTestDict,trainOrTest_TEXT_Dict,embeddingFlag):
    counter = 0
    listOfClinicalNotes_Doc2vec = []
    listOfClinicalNotes_SectionSeq = []
    
    for fileName, seqOfSecHeader in trainOrTestDict.items():
        counter = counter+1
        seqOfSecHeader=ast.literal_eval(seqOfSecHeader[0])
        if len(seqOfSecHeader)>1:
#            print(fileName)
            listOfClinicalNotes_Doc2vec.append([sectionHeaderSimilarities(trainOrTest_TEXT_Dict.get(fileName),fileName,sectionHeader,embeddingFlag) for sectionHeader in seqOfSecHeader])#np.array([list(model_doc2vec.docvecs[fileName+'_'+sectionHeader])+sectionHeaderSimilarities(fileName+'_'+sectionHeader) for sectionHeader in seqOfSecHeader])
            listOfClinicalNotes_SectionSeq.append(np.array([sectionHeaders.index(sectionHeader)+1 for sectionHeader in seqOfSecHeader]))
        if counter%1000 == 0:
            print (str(counter)+ ' files processed')
    return listOfClinicalNotes_Doc2vec,listOfClinicalNotes_SectionSeq


def getXY(someDict, someTextDict,embeddingFlag):
    listOfClinicalNotes_Doc2vec,listOfClinicalNotes_SectionSeq = createListOfClinicalNotes_Doc2vec_And_SectionSeq(someDict,someTextDict,embeddingFlag)
    X = pad_sequences(maxlen=max_len, dtype='int32',sequences=listOfClinicalNotes_Doc2vec, padding="post", value=0)
    y1 = pad_sequences(maxlen=max_len, sequences=listOfClinicalNotes_SectionSeq, padding="post", value=0)
    y = [to_categorical(i, num_classes=max_len+1) for i in y1]
    return X,y
    
def getTrainTestXY(train_Dict, test_Dict, getTrainOrTestOrBoth,embeddingFlagTraining):
    if getTrainOrTestOrBoth == "Train":
        return getXY(train_Dict,train_TEXT_Dict,embeddingFlagTraining)
    elif getTrainOrTestOrBoth == "Test":
        return getXY(test_Dict,test_TEXT_Dict,embeddingFlagTraining)
    elif getTrainOrTestOrBoth == "Both":
        X_Train,y_Train = getXY(train_Dict,train_TEXT_Dict,embeddingFlagTraining)
        X_Test,y_Test = getXY(test_Dict,test_TEXT_Dict,embeddingFlagTraining)
        return X_Train,y_Train,X_Test,y_Test
    

def getF1score(inp):
    eps = 0.000000001
    return (2*(float(inp[0]*inp[1])))/(float(inp[0]+inp[1]+eps))

def mapHeadersListToString(headersToCharMap, listToMap):
    return "".join(list(map(lambda x: headersToCharMap.get(x), listToMap)))

def getSequenceDistances(trueSequence, predSequence, headersToCharMap):
    trueMapped = mapHeadersListToString(headersToCharMap, trueSequence)
    predMapped = mapHeadersListToString(headersToCharMap, predSequence)
    editDistance = stringdist.levenshtein(trueMapped, predMapped)
#    dlDistance = jf.damerau_levenshtein_distance(trueMapped,predMapped)
    return editDistance, 0
    
def evaluateModel(model,X_Test,y_Test):
    counter = 0
    numCorrectSeqPred=0
    numCorrectPredPerSeq = [0]*len(sectionHeaders)
    numAppearancesPerSeq = [0]*len(sectionHeaders)
    numTotalPredPerSeq = [0]*len(sectionHeaders)
    editDistance = 0
    editDistanceWithSwap = 0
    
    numToChar = {}
    for x in range(26):
        numToChar[x] = str(chr(x+97))
    
    headersToCharMap = {value:numToChar.get(index) for index,value in enumerate(sectionHeaders)}
    
    for i in range(len(X_Test)):
        counter = counter +1
        p = model.predict(np.array([X_Test[i]]))
        p = np.argmax(p, axis=-1)
        true = np.argmax(y_Test[i], -1)
        predictedSeq=[sectionHeaders[pred-1] for pred in p[0] if pred!=0]
        trueSeq=[sectionHeaders[t-1] for t in true if t!=0]
        
        editDistances = getSequenceDistances(trueSeq, predictedSeq, headersToCharMap)
        editDistance += editDistances[0]
        editDistanceWithSwap += editDistances[1]
        
        pairwise = zip (trueSeq, predictedSeq)
        matched_sections = [pair[0] for idx, pair in enumerate(pairwise) if pair[0] == pair[1]]
        unMatched_sections = [pair[1] for idx, pair in enumerate(pairwise) if pair[0] != pair[1]]
        for corrSec in matched_sections:
            numCorrectPredPerSeq[sectionHeaders.index(corrSec)]=numCorrectPredPerSeq[sectionHeaders.index(corrSec)]+1
            numTotalPredPerSeq[sectionHeaders.index(corrSec)]=numTotalPredPerSeq[sectionHeaders.index(corrSec)]+1
        for inCorrSec in unMatched_sections:
           numTotalPredPerSeq[sectionHeaders.index(inCorrSec)]=numTotalPredPerSeq[sectionHeaders.index(inCorrSec)]+1
        for section in trueSeq:
            numAppearancesPerSeq[sectionHeaders.index(section)]=numAppearancesPerSeq[sectionHeaders.index(section)]+1
        if str(predictedSeq) == str(trueSeq):
            numCorrectSeqPred=numCorrectSeqPred+1
        if counter%10000 == 0:
            print (str(counter)+ ' files processed')
    
    editDistance = editDistance/float(len(X_Test))
    editDistanceWithSwap = editDistanceWithSwap/float(len(X_Test))
    
    eps = 0.000000001
    perSections_Prec = [float(pair[1])/float(pair[0]+eps) for idx, pair in enumerate(zip(numTotalPredPerSeq, numCorrectPredPerSeq))] 
    perSections_Rec = [float(pair[1])/float(pair[0]+eps) for idx, pair in enumerate(zip(numAppearancesPerSeq, numCorrectPredPerSeq))]
    perSections_F1Score = list(map(getF1score,zip(perSections_Prec,perSections_Rec)))
    return pd.DataFrame(list(zip(sectionHeaders,perSections_Prec,perSections_Rec,numAppearancesPerSeq,numCorrectPredPerSeq,perSections_F1Score)),columns = ['sectionHeaders','perSections_Prec','perSections_Rec','numAppearancesPerSeq','numCorrectPredPerSeq','F1-Score-PerSection']),float(numCorrectSeqPred)/float(len(X_Test)), editDistance, editDistanceWithSwap

def printTrueSeqVsPredictionSeq(model,X_Test,y_Test):
    for i in range(len(X_Test)):
        p = model.predict(np.array([X_Test[i]]))
        p = np.argmax(p, axis=-1)
        true = np.argmax(y_Test[i], -1)
        
        
        print("{:50}{}".format("True", "Pred"))
        print(80 * "=")
        for t, pred in zip(true, p[0]):
            if t!=0:
                print("{:50} {}".format(sectionHeaders[t-1], sectionHeaders[pred-1]))
        print(80 * "=")


def createModelArchitecture(max_len,num_LSTM_Units,vector_dim,num_docs,embedding_matrix,dropout):
    doc_input = Input(shape=(max_len,),dtype='float32', name='doc_input')
    #
    ##np.savetxt('embedding_matrix.csv', embedding_matrix, delimiter=',')
    #
    print("Creating Embedding Layer...")
    #embedding layer intialized with the matrix created earlier
    embedded_doc_input = Embedding(output_dim=vector_dim, input_dim=num_docs,weights=[embedding_matrix], trainable=False,mask_zero=True)(doc_input)
    #
    print("Embedding Layer Created, Creating BiDirectional Layer...")
    model=(Bidirectional(LSTM(units=num_LSTM_Units, return_sequences=True,recurrent_dropout=dropout)))(embedded_doc_input) # variational biLSTM
    model=(Bidirectional(LSTM(units=num_LSTM_Units, return_sequences=True,recurrent_dropout=dropout)))(model) # variational biLSTM
    #
    print("BiDirectional Layer Created, Creating TimeDistributed Layer...")
    model=(TimeDistributed(Dense(vector_dim, activation="relu")))(model)  # a dense layer as suggested by neuralNer
    #
    print("TimeDistributed Layer Created, Creating CRF Layer...")
    crf = CRF(max_len+1)  # CRF layer
    out = crf(model)
    print("CRF Layer Created, Training...")
    model = Model(doc_input, out)
    return model,crf

def trainKerasModel(max_len,num_LSTM_Units,learning_rate,vector_dim,num_docs,embedding_matrix,embeddingLayerFlag,embeddingFlag,dropout,batch_size,epochs,X_Train,y_Train,experiment):
    model,crf=createModelArchitecture(max_len,num_LSTM_Units,vector_dim,num_docs,embedding_matrix,dropout)
    print("Model Architecture Created")
    callbacks=[EarlyStopping(patience=2, monitor='val_loss')]
    
    model.compile(loss=crf.loss_function, optimizer=RMSprop(lr=learning_rate), metrics=[crf.accuracy])
    model.summary()
    
    history = model.fit(X_Train, np.array(y_Train), batch_size=batch_size, epochs=epochs,
                        validation_split=0.15, verbose=1,callbacks=callbacks)
    
    print("Model Training Done, Saving...")
    modelName = 'KerasModel'+'_lstm'+str(num_LSTM_Units)+'_lr'+str(learning_rate)+'_dropOut'+str(dropout)+'_bSize'+str(batch_size)+'_epochs'+str(epochs)+'_'+embeddingLayerFlag+'_'+embeddingFlag+'_'+str(experiment)+'exp.h5'
    save_load_utils.save_all_weights(model,path+modelName,include_optimizer=False)
    print("Model Saved")
    
def testKerasModel(max_len,num_LSTM_Units,learning_rate,vector_dim,num_docs,embedding_matrix,embeddingLayerFlag,embeddingFlag,dropout,batch_size,epochs,X_Test,y_Test,experiment):
    model1,crf1=createModelArchitecture(max_len,num_LSTM_Units,vector_dim,num_docs,embedding_matrix,dropout)
    model1.compile(loss=crf1.loss_function, optimizer="rmsprop", metrics=[crf1.accuracy])
    print("Model Architecture Created")
    print("Loading Model")
    modelName = 'KerasModel'+'_lstm'+str(num_LSTM_Units)+'_lr'+str(learning_rate)+'_dropOut'+str(dropout)+'_bSize'+str(batch_size)+'_epochs'+str(epochs)+'_'+embeddingLayerFlag+'_'+embeddingFlag+'_'+str(experiment)+'exp.h5'
    save_load_utils.load_all_weights(model1,path+modelName,include_optimizer=False)
    
    print("Model Loaded, Testing...")
    summaryResult,percCorrectSeqPred,editDistance, editDistanceWithSwap = evaluateModel(model1,X_Test,y_Test)
    
    print("Percentage of Correctly Predicted Sequences :: ", percCorrectSeqPred)
    print("Average Edit Distance :: ", editDistance)
    print("Average Edit Distance with swaps allowed :: ", editDistanceWithSwap)
    print(summaryResult)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


num_LSTM_Units = int(sys.argv[1])
learning_rate = float(sys.argv[2])
dropout = float(sys.argv[3])
batch_size = int(sys.argv[4])
epochs = int(sys.argv[5])
embeddingLayerFlag = sys.argv[6]#"AllEmbeddings" #TrainEmbeddings
experiment = int(sys.argv[7]) # 1,2,3

print("Creating the Training Embedding Matrix and Training Data")

if experiment == 1:
    if embeddingLayerFlag == "AllEmbeddings":
        embeddingFlag = "Infer"
        X_Train,y_Train,X_Test,y_Test = getTrainTestXY(train_Dict,test_Dict,"Both",embeddingFlag)
        trainKerasModel(max_len,num_LSTM_Units,learning_rate,vector_dim,num_docs,embedding_matrix,embeddingLayerFlag,embeddingFlag,dropout,batch_size,epochs,X_Train,y_Train,experiment)
        testKerasModel(max_len,num_LSTM_Units,learning_rate,vector_dim,num_docs,embedding_matrix,embeddingLayerFlag,embeddingFlag,dropout,batch_size,epochs,X_Test,y_Test,experiment)
    elif embeddingLayerFlag == "TrainEmbeddings":
        embeddingFlag = "Infer"
        X_Train,y_Train = getTrainTestXY(train_Dict,test_Dict,"Train",embeddingFlag)
        trainKerasModel(max_len,num_LSTM_Units,learning_rate,vector_dim,num_docs,embedding_matrix,embeddingLayerFlag,embeddingFlag,dropout,batch_size,epochs,X_Train,y_Train,experiment)
        X_Test,y_Test = getTrainTestXY(train_Dict,test_Dict,"Test",embeddingFlag)
        testKerasModel(max_len,num_LSTM_Units,learning_rate,vector_dim,num_docs,embedding_matrix,embeddingLayerFlag,embeddingFlag,dropout,batch_size,epochs,X_Test,y_Test,experiment)
elif experiment == 2:
    if embeddingLayerFlag == "AllEmbeddings":
        embeddingFlag = "Infer"
        X_Train,y_Train = getTrainTestXY(train_Dict,test_Dict,"Train",embeddingFlag)
        X_Test,y_Test = getTrainTestXY(train_Dict,test_Dict,"Test","NotInfer")
        trainKerasModel(max_len,num_LSTM_Units,learning_rate,vector_dim,num_docs,embedding_matrix,embeddingLayerFlag,embeddingFlag,dropout,batch_size,epochs,X_Train,y_Train,experiment)
        testKerasModel(max_len,num_LSTM_Units,learning_rate,vector_dim,num_docs,embedding_matrix,embeddingLayerFlag,embeddingFlag,dropout,batch_size,epochs,X_Test,y_Test,experiment)
    elif embeddingLayerFlag == "TrainEmbeddings":
        embeddingFlag = "Infer"
        X_Train,y_Train = getTrainTestXY(train_Dict,test_Dict,"Train",embeddingFlag)
        trainKerasModel(max_len,num_LSTM_Units,learning_rate,vector_dim,num_docs,embedding_matrix,embeddingLayerFlag,embeddingFlag,dropout,batch_size,epochs,X_Train,y_Train,experiment)
        X_Test,y_Test = getTrainTestXY(train_Dict,test_Dict,"Test","NotInfer")
        testKerasModel(max_len,num_LSTM_Units,learning_rate,vector_dim,num_docs,embedding_matrix,embeddingLayerFlag,embeddingFlag,dropout,batch_size,epochs,X_Test,y_Test,experiment)
elif experiment == 3:
    if embeddingLayerFlag == "AllEmbeddings":
        embeddingFlag = "NotInfer"
        X_Train,y_Train = getTrainTestXY(train_Dict,test_Dict,"Train",embeddingFlag)
        X_Test,y_Test = getTrainTestXY(train_Dict,test_Dict,"Test","NotInfer")
        trainKerasModel(max_len,num_LSTM_Units,learning_rate,vector_dim,num_docs,embedding_matrix,embeddingLayerFlag,embeddingFlag,dropout,batch_size,epochs,X_Train,y_Train,experiment)
        testKerasModel(max_len,num_LSTM_Units,learning_rate,vector_dim,num_docs,embedding_matrix,embeddingLayerFlag,embeddingFlag,dropout,batch_size,epochs,X_Test,y_Test,experiment)
    elif embeddingLayerFlag == "TrainEmbeddings":
        embeddingFlag = "NotInfer"
        X_Train,y_Train = getTrainTestXY(train_Dict,test_Dict,"Train",embeddingFlag)
        trainKerasModel(max_len,num_LSTM_Units,learning_rate,vector_dim,num_docs,embedding_matrix,embeddingLayerFlag,embeddingFlag,dropout,batch_size,epochs,X_Train,y_Train,experiment)
        X_Test,y_Test = getTrainTestXY(train_Dict,test_Dict,"Test","NotInfer")
        testKerasModel(max_len,num_LSTM_Units,learning_rate,vector_dim,num_docs,embedding_matrix,embeddingLayerFlag,embeddingFlag,dropout,batch_size,epochs,X_Test,y_Test,experiment)


#listOfSectionHeaders_Dict={'activity': ['activity','Activity Status'],
# 
# 'admission_date': ['admission_date', 'admission date'],
# 
# 'allergies_and_adverse_reactions': ['Known Adverse and Allergic Drug Reactions ( if none, enter NKA )',
#'Allergic Disorder History',
#  'ALLERGIES / ADVERSE REACTIONS',
#  'allergy adverse reaction',
#  'allergies and adverse reactions',
#  'allergy and adverse reaction',
#  'allergy adverse reaction',
#  'adverse reactions',
#  'adverse reaction',
#  'history of allergies',
#  'history of allergy',
#  'history allergy',
#  'allergies',
#  'allergy'
#  ],
#
# 'assessment': ['clinical impression',
#  'initial impression',
#  'impression',
#  'interpretation'],
#
# 'assessment_and_plan': [
#  'impression / recommendations',
#  'impression recommendation',
#  'impression and recommendation',
#  'impression and recommendations',
#  'impression and plans',
#  'impression and plan',
#  'impression / plan',
#  'impression plan',
#  'plan and discussion',
#  'plan discussion',
#  'assessment_and_plan',
#  'assessment and recommendations',
#  'assessment and recommendation',
#  'assessment recommendation',
#  'assessment and plan',
#  'assessment plan',
#  'assessment / plan',
#  'assessment plan',
#  'Assessment',
#  'clinical comments',
#  'clinical comment',
#  'a & p',
#  'a / p',
#  'a p'],
#
#
#
# 'chief_complaint': ['identifying data/chief complaint',
#  'identify data chief complaint',
#  'identification / chief complaint',
#  'identification chief complaint',
#  'identification and chief complaint',
#  'reason for admission and consultation',
#  'reason admission consultation',
#  'reason for admission / chief complaint',
#  'reason for admission/chief complaint',
#  'reason for admission chief complaint',
#  'reason admission chief complaint'
#  'reason for hospitalization',
#  'reason hospitalization',
#  'reason for admission',
#  'reason admission',
#  'reason for visit',
#  'reason visit',
#  'chief concern',
#  'id / cc',
#  'id cc',
#  'here for a chief complaint of',
#  'here a chief complaint',
#  'chief complaint'],
#
#
# 'diagnoses': [
#  'discharge diagnoses',
#  'FINAL DIAGNOSES',
#  'FINAL DIAGNOSIS',
#  'Diagnosis',
#  'diagnoses'],
#
# 'discharge_condition': ['DISCHARGE CONDITION'],
#
# 'discharge_instructions': ['discharge instructions',
#  'discharge instruction'],
#
# 'family_history': ['family history'],
#
# 'findings': [
#  'findings at surgery',
#  'finding at surgery',
#  'finding surgery',
#  'diagnostic findings',
#  'diagnostic finding',
#  'indications / findings',
#  'indication finding',
#  'diagnostic impression',
#  'findings',
#  'finding'],
#
# 'follow_up': [
#  'followup care plan',
#  'return to clinic',
#  'return clinic',
#  'rtc',
#  'followup instructions',
#  'followup instruction',
#  'follow-up appointments ; arrangements for care',
#  'follow up appointment arrangement for care',
#  'follow up appointment arrangement care',
#  'FOLLOW-UP PLANS',
#  'follow up appointment',
#  'followup appointments',
#  'followup appointment',
#  'followup',
#  'follow - up',
#  'follow-up',
#  'follow up'],
#
# 'history_present_illness': [
#  'History of Present Illness / Subjective Complaint',
#  'hpi / interval history',
#  'hpi interval history',
#  'patient hpi',
#  'summary of present illness',
#  'summary present illness',
#  'HPI / Subjective Complaint',
#  'EVENTS/EXTENDED HISTORY OF PRESENT ILLNESS',
#  'history / physical examination',
#  'history physical examination',
#  'in clinical history',
#  'clinical history / indications',
#  'clinical history indication',
#  'clinical history',
#  'issues briefly as following',
#  'issue briefly as following',
#  'current medical problems',
#  'current medical problem',
##  'indications',
##  'indication',
#  'patient history',
#  'history of chronic illness',
#  'history chronic illness',
#  'clinical presentation',
#  'history of present illness',
#  'history present illness',
#  'issues briefly as follows',
#  'issue briefly as follow',
#  'clinical indication',
#  'history present illness',
#  'history of the present illness',
#  'interval history',
#  'history present illness',
#  'present illness',
#  'hpi'],
#
# 'history_source': [
#  'historian',
#  'history obtained from',
#  'history obtain from',
#  'history obtain',
#  'HX obtained from',
#  'hx obtain from',
#  'hx obtain',
#  'history source',
#  'SOURCES OF INFORMATION',
#  'source of information',
#  'source information',
#  'informant',
#  'source'],
#
# 'hospital_course': ['course in the emergency department',
#  'course emergency department',
#  'BRIEF SUMMARY OF HOSPITAL COURSE',
#  'Hospital Course by system',
#  'SUMMARY OF HOSPITAL COURSE',
#  'history / hospital course',
#  'history hospital course',
#  'brief hospital course',
#  'hospital course'],
#                     
# 'laboratory_and_radiology_data': [
#  'AVAILABLE PERTINENT LABORATORY & X - RAY FINDINGS',
#  'Available Pertinent Laboratory & X-Ray Findings',
#  'available pertinent laboratory x ray findings',
#  'available pertinent laboratory x ray finding',
#  'studies performed',
#  'study perform',
#  'diagnostic procedure',
#  'laboratory and study data',
#  'laboratory study data',
#  'laboratory data / diagnostic studies',
#  'laboratory data diagnostic study',
#  'laboratory data',
#  'diagnostics',
#  'study data',
#  'laboratory and radiographic studies',
#  'laboratory and radiographic study',
#  'laboratory radiographic study',
#  'clinical data',
#  'imaging procedure',
#  'ancillary studies',
#  'ancillary study',
#  'Lab and radiological results',
#  'lab and radiological result',
#  'lab radiological result',
#  'available lab and x-ray results',
#  'available lab and x ray results',
#  'available lab and x ray result',
#  'available lab x ray result',
#  'lab and imaging',
#  'lab imaging',
#  'laboratory and radiology data',
#  'laboratory data / radiology',
#  'laboratory data radiology',
#  'comparison imaging',
#  'other results',
#  'other result',
#  'LABORATORY STUDIES',
#  'Laboratory data',
#  'Other labs',
#  'laboratory and radiology findings',
#  'laboratory and radiology finding',
#  'laboratory radiology finding',
#  'pertinent studies',
#  'pertinent study',
#  'PERTINENT LABORATORY VALUES ON PRESENTATION',
#  'PERTINENT RADIOLOGY / IMAGING',
#  'lab and radiological results',
#  'lab and radiological result',
#  'lab radiological result',
#  'laboratory and x - ray data',
#  'laboratory and x-ray data',
#  'laboratory and x ray data',
#  'laboratory x ray data',
#  'laboratory and radiology data',
#  'laboratory radiology data',
#  'diagnostic tests and procedures',
#  'diagnostic test and procedure',
#  'diagnostic test procedure',
#  'special studies',
#  'special study',
#  'preprocedure studies',
#  'preprocedure study',
#  'diagnostic studies',
#  'diagnostic study',
#  'Laboratory or imaging',
#  'laboratory imaging',
#  'comparison_studies',
#  'comparison studies',
#  'comparison study'
#  ],
#
#
#
#
# 'medications': [
#  'MEDICATIONS / HERBS / SUPPLEMENTS',
#  'medication herb supplement',
#  'Medications administered',
#  'medication administer',
#  'MEDICATIONS ON ADMISSION',
#  'CURRENT MEDICATIONS',
#  'Last dose of Antibiotics',
#  'Infusions',
#  'Other ICU Medications',
#  'Other Medications',
#  'premedications',
#  'premedication',
#  'premorbid medications',
#  'premorbid medication',
#  'medications at vanderbilt',
#  'medication at vanderbilt',
#  'medication vanderbilt',
#  'Medications Known to be Prescribed for or Used by the Patient',
#  'Medications Known to be Prescribed for or Used by the Patient ( with dose , route , and frequency )',
#  'most recent medication',
#  'discharge medications',
#  'discharge medication',
#  'medications',
#  'medication'],
#
#
#
# 'past_medical_history': ['past medical history and review of systems',
#  'past medical history and review of system',
#  'past medical history review system',
#  'past medical problems',
#  'past medical problem',
#  'history of past illness',
#  'history past illness',
#  'past medical history',
#  'Past Medical / Surgical History',
#  'Other Past Medical History',
#  'PAST MEDICAL HISTORY',
#  'previous medical history',
#  'hematology / oncology history',
#  'hematology oncology history',
#  'history of general health',
#  'history general health',
#  'past medical history / past surgical history',
#  'past medical history past surgical history',
#  'medical problems',
#  'medical problem',
#  'significant past medical history',
#  'history of major illnesses and injuries',
#  'history of major illness and injury',
#  'history major illness injury',
#  'past med history',
#  'past hospitalization history',
#  'past medical and surgical history',
#  'past medical surgical history',
#  'brief medical history',
#  'Past Medical History / Problem List',
#  'Past Medical History/Problem List',
#  'past medical history problem list',
#  'past medical issues',
#  'past medical issue',
#  'past medical history / surgical history',
#  'past medical history surgical history',
#  'past infectious history',
#  'past medical history/family history',
#  'past medical history family history',
#  'Known Significant Medical Diagnoses and Conditions',
#  'past medical history',
#  'medical history',
#  'past history',
#  'illnesses',
#  'illness',
#  'pmhx',
#  'pmh'],
#
#
# 'physical_examination': ['physical examination as compared to admission',
#  'physical examination as compare to admission',
#  'physical examination as compare admission',
#  'external examination',
#  'physical exam compared admission',
#  'PHYSICAL EXAM AT TIME OF ADMISSION',
#  'physical exam as compared to admission',
#  'physical exam as compare to admission',
#  'physical exam as compare admission',
#  "My key findings of this patient ' s physical exam are",
#  'my key finding of this patient physical exam be',
#  'my key finding this patient physical exam be',
#  'admission physical exam',
#  "I examined the patient and confirmed the House Staff ' s Admission Physical Exam",
#  'i examine the patient and confirm the house staff admission physical exam',
#  'i examine patient confirm house staff admission physical exam',
#  'examination on discharge',
#  'physicial examination',
#  'examination on discharge compared to admission',
#  'examination on discharge compare to admission',
#  'examination discharge compare admission',
#  'examination discharge',
#  'physical examination by organ systems',
#  'physical examination by organ system',
#  'physical examination organ system',
#  'physical findings',
#  'physical finding',
#  'physical exam compare admission',
#  'physical examination',
#  'PE on admission',
#  'PE on discharge',
#  'admission exam',
#  'admit exam',
#  'exam on admission',
#  'admission examination',
#  'physical exam',
#  'admission PE',
#  'examination',
#  'exam',
#  'pe'],
#
#
# 'review_of_systems': ['systems review',
#  'system review',
#  'history of symptoms & diseases',
#  'history of symptom disease',
#  'history symptom disease',
#  'review of symptoms and diseases',
#  'review of symptom and disease',
#  'review symptom disease',
#  'cardiovascular review of systems',
#  'cardiovascular review of system',
#  'cardiovascular review system',
#  'cardiac review of systems',
#  'cardiac review of system',
#  'cardiac review system',
#  'social history / family history/review of systems',
#  'social history family history review of system',
#  'social history family history review system',
#  'review of systems',
#  'review of system',
#  'review system',
#  'ros'],
#
#
# 'social_history': [
#  'Social / Occupational History',
#  'SOCIAL HISTORY']
# }
#
#listOfSectionHeaders=list(listOfSectionHeaders_Dict.keys())
#
#def findWholeWord(w):
#    return re.compile(r'(^\s*\b({0})\b)|([\r\n]+\s*\b({0})\b)'.format(w), flags=re.IGNORECASE).search
#
#def workOnSimilarKeywords(keywords,note):
#    exists = False
#    return any([exists or findWholeWord(keyword.lower())(note.lower())  for keyword in keywords])
#
#def getsectionSpans(keywords,note,key):
#    for keyword in keywords:
#        match = findWholeWord(keyword.lower())(note.lower())
#        if match:
#            return (match.span()[0],match.span()[1],key)
#            
#    return any([findWholeWord(keyword.lower())(note.lower())  for keyword in keywords])
#
#def cleanClinicalNote(text):
#    timeRegex = "\\b(1[012]|0[1-9]):([0-5][0-9])(\\s)?([Aa]|[pP])[mM]"
#    p1 = re.compile(timeRegex)
#    m1=p1.subn("TIMETOKEN",text) 
#    
#    squareBracketStarRegex = "\[\*\*(.*?)\*\*\]"
#    p2 = re.compile(squareBracketStarRegex)
#    m2=p2.subn("",m1[0]) 
#    
#    numberRegex="[-+]?[0-9]*\.?[0-9]+"
#    p3 = re.compile(numberRegex)
#    m3=p3.subn("NUM",m2[0]) 
#
#    return m3[0]
#
#
#counter = 0
#def getSectionText(note): 
#    global counter
#    counter = counter+1
#    sectionSpans = [getsectionSpans(listOfSectionHeaders_Dict.get(key),note,key) for key in listOfSectionHeaders]
#    sectionSpansSorted = sorted([tup for tup in sectionSpans if tup], key=lambda element: element[0] )
#    sectionSequence = [tup[2] for tup in sectionSpansSorted]
#    map_sectionHeader_span=dict([ (tup[2], tup[:2]) for tup in sectionSpansSorted ]) 
#    sectionText = []
#    sectionHeadersToRemove=[]
#    for sectionHeader in listOfSectionHeaders:
#        currSectionHeaderIndices = map_sectionHeader_span.get(sectionHeader)
#        if not currSectionHeaderIndices:
#            sectionText.append("")
#            continue
#        idx = sectionSequence.index(sectionHeader)+1
#        if idx<len(sectionSequence):
#            nextSectionHeaderIndices = map_sectionHeader_span.get(sectionSequence[sectionSequence.index(sectionHeader)+1])
#            if nextSectionHeaderIndices[0] - currSectionHeaderIndices[1] <4:
#                sectionText.append("")
#                sectionHeadersToRemove.append(sectionHeader)
#                continue
#            else:
#                txt = note[currSectionHeaderIndices[1]+1:nextSectionHeaderIndices[0]]
#                txt = cleanClinicalNote(txt)
#                sectionText.append(txt)
#        else:
#            txt = note[currSectionHeaderIndices[1]+1:]
#            if len(txt) <4:
#                sectionText.append("")
#                sectionHeadersToRemove.append(sectionHeader)
#                continue
#            else:
#                txt = cleanClinicalNote(txt)
#                sectionText.append(txt)
#    for sectionHeaderToRemove in sectionHeadersToRemove:
#        sectionSequence.remove(sectionHeaderToRemove)
#    sectionText.append(sectionSequence)
#    if counter % 1000 == 0:
#        print(str(counter)+" notes processed")
#    return sectionText
#
#
#import re
#
#fileName='801.txt'
#with open(path+fileName, 'r') as myfile:
#  data = myfile.read()
##
#dataPara = getSectionText(data)
#seqInThisNote = dataPara[21]
#q = [x for x in dataPara[:21] if str(x) != '']
#
#
#model_doc2vec.random.seed(0)
#doc2VecOfNewTest = [model_doc2vec.infer_vector(section_text_tokens.split(" "), steps=20, alpha=0.025) for section_text_tokens in q]
#doc2VecOfNewTestWithSimFeatures = []
#for doc2VecOfNewTestOneSec in doc2VecOfNewTest:
##    doc2VecOfNewTestWithSimFeatures.append(list(doc2VecOfNewTestOneSec)+[model_doc2vec.docvecs.similarity(sectionHeader,str(model_doc2vec.docvecs.most_similar([doc2VecOfNewTestOneSec])[0][0])) for sectionHeader in sectionHeaders])
#    doc2VecOfNewTestWithSimFeatures.append(list(doc2VecOfNewTestOneSec))
#    
#
##firstNonZeroIndex = (np.where(~embedding_matrix.any(axis=1))[0])[0]
#listOfDoc2VecIndices = []
#for oneSecDoc2VecOfNewTestWithSimFeatures in doc2VecOfNewTestWithSimFeatures:
#    numDoc = numDoc+1  
#    embedding_matrix[numDoc]=oneSecDoc2VecOfNewTestWithSimFeatures
#    listOfDoc2VecIndices.append(numDoc)             
#          
#listOfSeqIndices=np.array([sectionHeaders.index(sectionHeader)+1 for sectionHeader in seqInThisNote])
#       
#X_Test1 = pad_sequences(maxlen=max_len, dtype='int32',sequences=[listOfDoc2VecIndices], padding="post", value=0)
#y1_Test1 = pad_sequences(maxlen=max_len, sequences=[listOfSeqIndices], padding="post", value=0)
#y_Test1 = [to_categorical(i, num_classes=max_len+1) for i in y1_Test1]#21 section headers + 1 token for no input  
#
#
#printTrueSeqVsPredictionSeq(model1,X_Test1,y_Test1)
#
##hist = pd.DataFrame(history.history)
##
##import matplotlib.pyplot as plt
##plt.style.use("ggplot")
##plt.figure(figsize=(12,12))
##plt.plot(hist["viterbi_acc"])
##plt.plot(hist["val_viterbi_acc"])
##plt.show()
