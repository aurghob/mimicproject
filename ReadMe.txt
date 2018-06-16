Built and Run Instructions:
	- Run the shell script installDependency.sh to install all the libraries needed for the pyhton code to run.
			./installDependency.sh
	- The follow the instruction in file Gensim_Install_Instruction.txt install the Gensim library on the environment running.
	- Place all the files in the drive into a directory with the python scripts.
	- Choose any of the filename from the testSetDF.csv and pass as argument to the python script 
			python predictSecionHeaders_V1.py <filename> 
			
	To run phase 2 models , please find the python scripts and models at link : https://drive.google.com/open?id=1aiIJmWfIstSHJg2HTeiE_QzaxTRuLdby
		- To run on the pyhton script on different models change the model name in place holder - <Model to test> and the doc2vec in place holder - <Doc2Vec Model>:
			model_doc2vec= g.Doc2Vec.load(path+"<Doc2Vec Model>")
			save_load_utils.load_all_weights(model1,path+'<Model to test>',include_optimizer=False)
			
			--------------------------------------------------------------------------------------------------------------------------------------
			Model 1 : Load model file : ishan_test1_model2.h5 and doc2vec file : model_dbow0_nopretrained.bin
			Model 2 : Load model file : ishan_test1_model.h5 and doc2vec file : model_doc_2_vec.bin
			Model 3 : Load model file : model_dbow0_nopretrained_lstm500_dropout0_2_lr0002_noTestInTrainEmb_model.h5 and doc2vec file : model_dbow0_nopretrained.bin
			Model 4 : Load model file : model_dbow0_pretrained_lstm500_dropout0_2_lr0002_noTestInTrainEmb_model.h5 and doc2vec file : model_doc_2_vec.bin
			
		- The files getSectionHeaderCounts_V4.py and doc_2_vec_model.py need not to be runned as they were used for data preprocessing and doc2vec generation.
		
	To test the phase 3 models follow the below instruction :
		- Install Gensim version = 3.0.0 using the following command:
			pip install -Iv gensim==3.0.0
		- To run the pyhton script TestDocFile.py on different models change the model name in place holder - <Model to test> , <Doc2Vec Model> and file name to test:
			modelName = <Model Name>
			fileName = <File to test>
			model_doc2vec= g.Doc2Vec.load(dco2VecPath+"<Doc2Vec Model>")
			
			To run script from command line:
				python TestDocFile.py <FileName to test>
			
			--------------------------------------------------------------------------------------------------------------------------------------
			Model 5 : Load model file : KerasModel_lstm500_lr0.006_dropOut0.2_bSize100_epochs50_TrainEmbeddings_Infer_1exp.h5 and doc2vec file : model_dbow0_pretrainedpubWikiPMC_trained_vd100_full-36.bin
			Model 6 : Load model file : KerasModel_lstm500_lr0.006_dropOut0.2_bSize100_epochs50_AllEmbeddings_Infer_1exp.h5 and doc2vec file : model_dbow0_pretrainedpubWikiPMC_trained_vd100_full-36.bin
			
		- The file getSectionHeaderCounts_V7.py need not to be runned as they were used for data preprocessing and generate the cleaned text csv's : testSetDF.csv , 		      
			Train_Test_SectionHeader_text.csv and trainSetDF.csv .
		- The file testingKeras_V4.py is used to train the model and store it.