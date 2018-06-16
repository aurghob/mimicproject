#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import operator
import json

import os
import ast
import time
from collections import Counter

path = "C:\\Masters Studies\\NLP\\Project\\testClampBatch\\"
path_sectionMap = "C:\\Masters Studies\\NLP\\Project\\"
path_json = "C:\\Masters Studies\\NLP\\Project\\"
myFunCount = 0

listOfSectionHeaders_Dict={'activity': ['activity'],
 
 'admission_date': ['admission_date', 'admission date'],
 
 'allergies_and_adverse_reactions': ['Known Adverse and Allergic Drug Reactions ( if none, enter NKA )',
'Allergic Disorder History',
  'ALLERGIES / ADVERSE REACTIONS',
  'allergy adverse reaction',
  'allergies and adverse reactions',
  'allergy and adverse reaction',
  'allergy adverse reaction',
  'adverse reactions',
  'adverse reaction',
  'history of allergies',
  'history of allergy',
  'history allergy',
  'allergies',
  'allergy'
  ],

 'assessment': ['clinical impression',
  'initial impression',
  'impression',
  'interpretation'],

 'assessment_and_plan': [
  'impression / recommendations',
  'impression recommendation',
  'impression and recommendation',
  'impression and recommendations',
  'impression and plans',
  'impression and plan',
  'impression / plan',
  'impression plan',
  'plan and discussion',
  'plan discussion',
  'assessment_and_plan',
  'assessment and recommendations',
  'assessment and recommendation',
  'assessment recommendation',
  'assessment and plan',
  'assessment plan',
  'assessment / plan',
  'assessment plan',
  'Assessment',
  'clinical comments',
  'clinical comment',
  'a & p',
  'a / p',
  'a p'],



 'chief_complaint': ['identifying data/chief complaint',
  'identify data chief complaint',
  'identification / chief complaint',
  'identification chief complaint',
  'identification and chief complaint',
  'reason for admission and consultation',
  'reason admission consultation',
  'reason for admission / chief complaint',
  'reason for admission/chief complaint',
  'reason for admission chief complaint',
  'reason admission chief complaint'
  'reason for hospitalization',
  'reason hospitalization',
  'reason for admission',
  'reason admission',
  'reason for visit',
  'reason visit',
  'chief concern',
  'id / cc',
  'id cc',
  'here for a chief complaint of',
  'here a chief complaint',
  'chief complaint'],


 'diagnoses': [
         'PRINCIPAL DISCHARGE DIAGNOSIS',
  'discharge diagnoses',
  'ADMIT DIAGNOSIS',
  'OTHER DIAGNOSIS',
  'FINAL DIAGNOSES',
  'FINAL DIAGNOSIS',
  'Diagnosis',
  'diagnoses'],

 'discharge_condition': ['DISCHARGE CONDITION'],

 'discharge_instructions': ['discharge instructions',
  'discharge instruction'],

 'family_history': ['family history'],

 'findings': [
  'findings at surgery',
  'finding at surgery',
  'finding surgery',
  'diagnostic findings',
  'diagnostic finding',
  'indications / findings',
  'indication finding',
  'diagnostic impression',
  'findings',
  'finding'],

 'follow_up': [
  'followup care plan',
  'return to clinic',
  'return clinic',
  'rtc',
  'followup instructions',
  'followup instruction',
  'follow-up appointments ; arrangements for care',
  'follow up appointment arrangement for care',
  'follow up appointment arrangement care',
  'FOLLOW-UP PLANS',
  'follow up appointment',
  'followup appointments',
  'followup appointment',
  'followup',
  'follow - up',
  'follow-up',
  'follow up'],

 'history_present_illness': [
  'History of Present Illness / Subjective Complaint',
  'hpi / interval history',
  'hpi interval history',
  'patient hpi',
  'summary of present illness',
  'summary present illness',
  'HPI / Subjective Complaint',
  'EVENTS/EXTENDED HISTORY OF PRESENT ILLNESS',
  'history / physical examination',
  'history physical examination',
  'in clinical history',
  'clinical history / indications',
  'clinical history indication',
  'clinical history',
  'issues briefly as following',
  'issue briefly as following',
  'current medical problems',
  'current medical problem',
#  'indications',
#  'indication',
  'patient history',
  'history of chronic illness',
  'history chronic illness',
  'clinical presentation',
  'history of present illness',
  'history present illness',
  'issues briefly as follows',
  'issue briefly as follow',
  'clinical indication',
  'history present illness',
  'history of the present illness',
  'interval history',
  'history present illness',
  'present illness',
  'hpi'],

 'history_source': [
  'historian',
  'history obtained from',
  'history obtain from',
  'history obtain',
  'HX obtained from',
  'hx obtain from',
  'hx obtain',
  'history source',
  'SOURCES OF INFORMATION',
  'source of information',
  'source information',
  'informant'],
#  'source'],

 'hospital_course': ['course in the emergency department',
  'course emergency department',
  'BRIEF SUMMARY OF HOSPITAL COURSE',
  'BRIEF RESUME OF HOSPITAL COURSE',
  'Hospital Course by system',
  'SUMMARY OF HOSPITAL COURSE',
  'history / hospital course',
  'history hospital course',
  'brief hospital course',
  'hospital course'],
                     
 'laboratory_and_radiology_data': [
  'AVAILABLE PERTINENT LABORATORY & X - RAY FINDINGS',
  'Available Pertinent Laboratory & X-Ray Findings',
  'available pertinent laboratory x ray findings',
  'available pertinent laboratory x ray finding',
  'studies performed',
  'study perform',
  'diagnostic procedure',
  'laboratory and study data',
  'laboratory study data',
  'laboratory data / diagnostic studies',
  'laboratory data diagnostic study',
  'laboratory data',
  'diagnostics',
  'study data',
  'laboratory and radiographic studies',
  'laboratory and radiographic study',
  'laboratory radiographic study',
  'clinical data',
  'imaging procedure',
  'ancillary studies',
  'ancillary study',
  'Lab and radiological results',
  'lab and radiological result',
  'lab radiological result',
  'available lab and x-ray results',
  'available lab and x ray results',
  'available lab and x ray result',
  'available lab x ray result',
  'lab and imaging',
  'lab imaging',
  'laboratory and radiology data',
  'laboratory data / radiology',
  'laboratory data radiology',
  'comparison imaging',
  'other results',
  'other result',
  'LABORATORY STUDIES',
  'Laboratory data',
  'Other labs',
  'laboratory and radiology findings',
  'laboratory and radiology finding',
  'laboratory radiology finding',
  'Pertinent Results',
  'pertinent studies',
  'pertinent study',
  'PERTINENT LABORATORY VALUES ON PRESENTATION',
  'PERTINENT RADIOLOGY / IMAGING',
  'lab and radiological results',
  'lab and radiological result',
  'Labs / Radiology',
  'lab radiological result',
  'laboratory and x - ray data',
  'laboratory and x-ray data',
  'laboratory and x ray data',
  'laboratory x ray data',
  'laboratory and radiology data',
  'laboratory radiology data',
  'diagnostic tests and procedures',
  'diagnostic test and procedure',
  'diagnostic test procedure',
  'special studies',
  'special study',
  'preprocedure studies',
  'preprocedure study',
  'diagnostic studies',
  'diagnostic study',
  'Laboratory or imaging',
  'laboratory imaging',
  'comparison_studies',
  'comparison studies',
  'comparison study'
  ],




 'medications': [
  'ADMISSION MEDICATIONS',
  'MEDICATIONS / HERBS / SUPPLEMENTS',
  'medication herb supplement',
  'Medications administered',
  'medication administer',
  'MEDICATIONS ON ADMISSION',
  'CURRENT MEDICATIONS',
  'Last dose of Antibiotics',
  'Infusions',
  'Other ICU Medications',
  'Other Medications',
  'premedications',
  'premedication',
  'premorbid medications',
  'premorbid medication',
  'medications at vanderbilt',
  'medication at vanderbilt',
  'medication vanderbilt',
  'Medications Known to be Prescribed for or Used by the Patient',
  'Medications Known to be Prescribed for or Used by the Patient ( with dose , route , and frequency )',
  'most recent medication',
  'medications',
  'medication'],

'discharge medications':['discharge medications',
  'discharge medication'],

 'past_medical_history': ['past medical history and review of systems',
  'past medical history and review of system',
  'past medical history review system',
  'past medical problems',
  'past medical problem',
  'history of past illness',
  'history past illness',
  'past medical history',
  'Past Medical / Surgical History',
  'Other Past Medical History',
  'PAST MEDICAL HISTORY',
  'previous medical history',
  'hematology / oncology history',
  'hematology oncology history',
  'history of general health',
  'history general health',
  'past medical history / past surgical history',
  'past medical history past surgical history',
  'medical problems',
  'medical problem',
  'significant past medical history',
  'history of major illnesses and injuries',
  'history of major illness and injury',
  'history major illness injury',
  'past med history',
  'past hospitalization history',
  'past medical and surgical history',
  'past medical surgical history',
  'brief medical history',
  'Past Medical History / Problem List',
  'Past Medical History/Problem List',
  'past medical history problem list',
  'past medical issues',
  'past medical issue',
  'past medical history / surgical history',
  'past medical history surgical history',
  'past infectious history',
  'past medical history/family history',
  'past medical history family history',
  'Known Significant Medical Diagnoses and Conditions',
  'past medical history',
  'medical history',
  'past history',
  'illnesses',
  'illness',
  'pmhx',
  'pmh'],


 'physical_examination': ['physical examination as compared to admission',
  'physical examination as compare to admission',
  'physical examination as compare admission',
  'external examination',
  'physical exam compared admission',
  'PHYSICAL EXAM AT TIME OF ADMISSION',
  'physical exam as compared to admission',
  'physical exam as compare to admission',
  'physical exam as compare admission',
  "My key findings of this patient ' s physical exam are",
  'my key finding of this patient physical exam be',
  'my key finding this patient physical exam be',
  'admission physical exam',
  "I examined the patient and confirmed the House Staff ' s Admission Physical Exam",
  'i examine the patient and confirm the house staff admission physical exam',
  'i examine patient confirm house staff admission physical exam',
  'examination on discharge',
  'physicial examination',
  'examination on discharge compared to admission',
  'examination on discharge compare to admission',
  'examination discharge compare admission',
  'examination discharge',
  'physical examination by organ systems',
  'physical examination by organ system',
  'physical examination organ system',
  'physical findings',
  'physical finding',
  'physical exam compare admission',
  'physical examination',
  'PE on admission',
  'PE on discharge',
  'admission exam',
  'admit exam',
  'exam on admission',
  'admission examination',
  'physical exam',
  'admission PE',
  'examination',
  'exam',
  'pe'],


 'review_of_systems': ['systems review',
  'system review',
  'history of symptoms & diseases',
  'history of symptom disease',
  'history symptom disease',
  'review of symptoms and diseases',
  'review of symptom and disease',
  'review symptom disease',
  'cardiovascular review of systems',
  'cardiovascular review of system',
  'cardiovascular review system',
  'cardiac review of systems',
  'cardiac review of system',
  'cardiac review system',
  'social history / family history/review of systems',
  'social history family history review of system',
  'social history family history review system',
  'review of systems',
  'review of system',
  'review system',
  'ros'],


 'social_history': [
  'Social / Occupational History',
  'SOCIAL HISTORY']
 }

def findWholeWord(w):
    return re.compile(r'(^\s*\b({0})\b)|([\r\n]+\s*\b({0})\b)'.format(w), flags=re.IGNORECASE).search

def getparaSpans(keywords,para,index,key):
    global paras
    for keyword in keywords:
        match = findWholeWord(keyword.lower())(para.lower())
        if match:
            return (match.span()[0],match.span()[1],key)

def cleanClinicalNote(text):
    timeRegex = "\\b(1[012]|0[1-9]):([0-5][0-9])(\\s)?([Aa]|[pP])[mM]"
    p1 = re.compile(timeRegex)
    m1=p1.subn("TIMETOKEN",text) 
    
    squareBracketStarRegex = "\[\*\*(.*?)\*\*\]"
    p2 = re.compile(squareBracketStarRegex)
    m2=p2.subn("",m1[0]) 
    
    numberRegex="[-+]?[0-9]*\.?[0-9]+"
    p3 = re.compile(numberRegex)
    m3=p3.subn("NUM",m2[0]) 

    return m3[0]

def paragrapher(text):
    d = re.split(r"\r\n\r\n", text)
    d1 = re.split(r"\n\n", text)
    if len(d1)>len(d):
        return d1
    else:
        return d

def getSpans(paras):
    someResult = []
    index = 0
    for para in paras:
        if (len(paras)>1):
            for key in listOfSectionHeaders:
                for keyword in listOfSectionHeaders_Dict.get(key):
                    gotSomething = findWholeWord(keyword.lower())(para.lower())
    #            gotSomething = getparaSpans(listOfSectionHeaders_Dict.get(key),para,index,key)
                    if gotSomething:
                        someResult.append((gotSomething.span()[0],gotSomething.span()[1],key,index))
            if not gotSomething:
                for unknownkeyword in unknownValues:
                    match_unknownkeyword = findWholeWord(unknownkeyword.lower())(para.lower())
                    if match_unknownkeyword:
                        someResult.append((match_unknownkeyword.span()[0],match_unknownkeyword.span()[1],'Unknown',index))
    #                    break
        else:
            for key in listOfSectionHeaders:
                for keyword in listOfSectionHeaders_Dict.get(key):
                    gotSomething = findWholeWord(keyword.lower())(para.lower())
    #            gotSomething = getparaSpans(listOfSectionHeaders_Dict.get(key),para,index,key)
                    if gotSomething:
                        someResult.append((gotSomething.span()[0],gotSomething.span()[1],key,index))
            for unknownkeyword in unknownValues:
                match_unknownkeyword = findWholeWord(unknownkeyword.lower())(para.lower())
                if match_unknownkeyword:
                    someResult.append((match_unknownkeyword.span()[0],match_unknownkeyword.span()[1],'Unknown',index))
        index=index+1
    return someResult

def unique_by_key(elements, key=None):
    if key is None:
        # no key: the whole element must be unique
        key = lambda e: e
    return {key(el): el for el in elements}.values()

def preprocessAndGetSectionText(data):  
    paras = paragrapher(cleanClinicalNote(data))   
    allSpans=getSpans(paras)
    spansSorted = sorted(list(set(allSpans)), key=operator.itemgetter(3, 0,1))    

    for tup in spansSorted:
        if paras[tup[3]][tup[0]-3:tup[0]+1] == '\r\n.\r':
           spansSorted.remove(tup)
    
    unique_by_second_element = unique_by_key(spansSorted, key=operator.itemgetter(0,3))
    spansSortedUnique = sorted(unique_by_second_element, key=operator.itemgetter(3, 0,1)) 
    spansSortedUniqueCopy =  sorted(unique_by_second_element, key=operator.itemgetter(3, 0,1))

    sectionText={key: '' for key in listOfSectionHeaders+['Unknown']}
    
    for i in range(len(spansSortedUnique)):
        currentSpan = spansSortedUnique[i]
        currentSection = currentSpan[2]
        paraListIndex = currentSpan[3]
        currentSpanSI = currentSpan[0]
        currentSpanEI = currentSpan[1]
        if i<len(spansSortedUnique)-1 and spansSortedUnique[i+1][3] == paraListIndex:
            nextSpan = spansSortedUnique[i+1]
            nextSection = nextSpan[2]
            nextSpanSI = nextSpan[0]
        else:
            nextSpanSI = len(paras[paraListIndex])
            nextSection = None
        txt = paras[paraListIndex][currentSpanEI+1:nextSpanSI]
        if len(txt) >4:
            if currentSection == 'Unknown':
                if len(sectionText[currentSection]) == 0 and nextSection and nextSection == 'Unknown':
                    sectionText[currentSection] = txt
                    spansSortedUniqueCopy.remove(currentSpan)
                    continue
                else:
                    if nextSection and nextSection == 'Unknown':
                        sectionText[currentSection]=sectionText[currentSection]+' '+txt
                        spansSortedUniqueCopy.remove(currentSpan)
                    else:
                        sectionText[currentSection]=sectionText[currentSection]+txt+'**Unknown**'
            else:
                if len(sectionText[currentSection]) == 0 and nextSection and nextSection == currentSection:
                    sectionText[currentSection]=txt
                    spansSortedUniqueCopy.remove(currentSpan)
                elif nextSection and nextSection == currentSection:
                    sectionText[currentSection]=sectionText[currentSection]+' '+txt
                    spansSortedUniqueCopy.remove(currentSpan)
                else:
                    sectionText[currentSection]=sectionText[currentSection]+' '+txt
        else:
            spansSortedUniqueCopy.remove(currentSpan)
        if i<len(spansSortedUnique)-1 and spansSortedUnique[i+1][3]-paraListIndex >1:
            sectionText[currentSection]=sectionText[currentSection]+' '+" ".join(paras[paraListIndex+1:spansSortedUnique[i+1][3]])
    
    sectionSequence = [tup[2] for tup in spansSortedUniqueCopy]
    paraIdx = 0
    for i in range(len(spansSortedUniqueCopy)):
        currSpan=spansSortedUniqueCopy[i]
        currSpanParaIdx = currSpan[3]
        if paraIdx!=currSpanParaIdx:
            if i==0:
                txt="_".join(paras[i:currSpanParaIdx])
                sectionText['Unknown']=txt+'**Unknown**'+sectionText['Unknown']
                if currSpan[2]!='Unknown':
                    sectionSequence=['Unknown']+sectionSequence
            elif currSpan[0]!=0:
                prevSectionHeader = spansSortedUniqueCopy[i-1][2]
                sectionText[prevSectionHeader]=sectionText[prevSectionHeader]+' '+paras[currSpanParaIdx][0:currSpan[0]]
            paraIdx = currSpanParaIdx
            
    sectionText['Order of Section Header Appearence']=sectionSequence
    return sectionText


listOfSectionHeaders=list(listOfSectionHeaders_Dict.keys())

clamp_Map = pd.read_table(path_sectionMap+'section_map.txt', sep = "\t", header = None)

listOfSectionHeaders_Dict_CLAMP = clamp_Map.set_index(1).iloc[0:].stack().groupby(level=0).apply(list).to_dict()

for k in listOfSectionHeaders:
    listOfSectionHeaders_Dict_CLAMP.pop(k, None)
    
removeKeys = ['anesthesia','carbon_copy','closing','complications',
'condition',
'counts',
'data_base',
'description',
'estimated_blood_loss',
'identifying_information',
'instructions',
'objective_data',
'orders',
'problem_list',
'providers',
'references',
'reliability',
'report'
'report_status',
'technique'
]

listOfSectionHeaders=list(listOfSectionHeaders_Dict.keys())

for k in removeKeys:
    listOfSectionHeaders_Dict_CLAMP.pop(k, None)
    
unknownValues = []
for key, value in listOfSectionHeaders_Dict_CLAMP.items():
    unknownValues=unknownValues+value
unknownValues.append('comments')  
unknownValues.append('Date of Birth')  
unknownValues.append('TO DO/PLAN')
unknownValues.remove('recommendations')
unknownValues.remove('Recommendations')
unknownValues.remove('recommendation')
unknownValues=list(set(unknownValues))  

unknownValues.sort(key = lambda s: len(s))  
unknownValues.reverse()

#fileName='104351.txt'
#fileName='DischargeSummary4826672.txt'
#fileName='DischargeSummary9457973.txt'
#path=path+"batch32\\"
#with open(path+fileName, 'r') as myfile:
#  data = myfile.read()

#tmp=preprocessAndGetSectionText(data)

def getBatch(rootString):
    x = re.search(r'batch\d+', rootString)
    if("i2b2" in rootString):
        return "i2b2"
    if (x!=None):
        return x.group(0)
    
def getListOfFileNamesAndClampInputText(pathOfClampOutputFolders):
    listOfCLAMPSInputsAsText = []
    listOfCLAMPSOutputFilesNames = []
    fileCounter = 0
    batchCounter=0
    for root,dirs,files in os.walk(pathOfClampOutputFolders):
        batchCounter = batchCounter+1
        batchName = ""
        if (root!=pathOfClampOutputFolders):
            batchName = getBatch(root)
        for fileName in files:
            if fileName.endswith(".txt"):
                fileCounter=fileCounter+1
                with open(root+"\\"+fileName, 'r') as myfile:
                    data = myfile.read()
                listOfCLAMPSInputsAsText.append(data)
                
                listOfCLAMPSOutputFilesNames.append(batchName + ","+fileName)
                if fileCounter%10000==0:
                    print(str(fileCounter)+" files processed")
                    break
                
#        print(batchName+" Done")
    #    if batchCounter % 5 == 0:
    listOfFileNamesAndClampOutputDataFrames = list(zip(listOfCLAMPSOutputFilesNames,listOfCLAMPSInputsAsText))
    del listOfCLAMPSInputsAsText
    del listOfCLAMPSOutputFilesNames
    return listOfFileNamesAndClampOutputDataFrames

def replaceNE(noteString, dictelmt):
    try:
        noteString = re.sub(r'(\b({0})\b)'.format(dictelmt[0]),dictelmt[1],noteString)
    except:
        return noteString
    return noteString

def myFunc(reqTuple):
    global Root_dictionary
    global myFunCount
    fileEntitiesMapKey = reqTuple[0]
    fileText = reqTuple[1]
    entitiesMapForFile = Root_dictionary.get(fileEntitiesMapKey)
    for dictEntry in entitiesMapForFile.items():
        fileText = replaceNE(fileText, dictEntry)
    tempDict = {fileEntitiesMapKey:preprocessAndGetSectionText(fileText)}
    myFunCount +=1
    if (myFunCount%1000==0):
        print(str(myFunCount) + " preprocessAndGetSectionText run")
    return tempDict

print("hello")
fileNamesAndClamputInputText = getListOfFileNamesAndClampInputText(path)

#fileNamesAndDictions = {elem[0]:preprocessAndGetSectionText(elem[1]) for elem in fileNamesAndClamputInputText}
print("filesAndInputs created")
Root_dictionary = json.load(open(path_json + "CLamp_filename_entity_dict.json"))
print("json loaded")
listOfDictionaries = list(map(myFunc, fileNamesAndClamputInputText))
print("preprocessing completed")
reqDict={}
_ = [reqDict.update(someDict) for someDict in listOfDictionaries]
print("Diction created")
required_df = pd.DataFrame(reqDict).transpose()
print("Dataframe created")
#seq=required_df['Order of Section Header Appearence']

#cnt=Counter(str(e) for e in seq)
#df = pd.DataFrame.from_dict(cnt, orient='index').reset_index()
#perSeqCount=df
pd.DataFrame.to_csv(required_df,path+"cleanedHeaderText_withoutEn2Et.csv")
#pd.DataFrame.to_csv(df,path+"allClinicalNotesSeqCount_withoutEn2Et.csv")