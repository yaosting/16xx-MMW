<<<<<<< HEAD

How to run:
	cd /BR_SVM/python
	python3 run_svm_predict.py seg3_train.txt


note:
1. This demo need 'pandas' to be installed, which is very hard to be installed on Raspberry.
2. This BR_SVM/examples/feature_extraction.ipynb is used to Serialize batch training data
3. This demo includes a trained model in BR_SVM/examples/seg3_train.txt/seg15_train.txt/seggroup_train.txt
4. Please ingore the info 'Accuaracy = ..........' 


more details in examples/...txt
#training_testing_data_svm_acc_vel_timeseg3.txt is the all data
#seg3_test.txt and seg3_train.txt  is the original data

#seg3_test.txt.scale and seg3_train.txt.scale 
 is the Scaled data
 cmd: svm-scale seg3_train.txt > seg3_train.txt.scale

#seg3_train.txt.range 
 is the scale rule
 cdm: svm-scale -s train.range seg3_train.txt > seg3_train.txt.scale
     #使用train.range对test进行同样的缩放
     svm-scale -r train.range seg3_test.txt > seg3_test.txt.scale

#seg3_train.txt.scale.out and seg3_train.txt.scale.png
 is the result of the grid

#seg3_train.txt.model
 cmd: svm-train.exe [options] training_set_file [model_file]
 1.rho #决策函数中的常数项的相反数（-b）
 2.svm的输出 y = y + model.sv_coef(i)*RBF(u,x);

#seg3_test.txt.predict 
 is the prediction for the  seg3_test.txt.scale   
 cmd: svm-predict -b 1 test_file data_file.model output_file




#we can use the model information to build the DecisionFunction (created by seg3_train.txt.model)

%% DecisionFunction
function plabel = DecisionFunction(x,model)

gamma = model.Parameters(4);

RBF = @(u,v)( exp(-gamma.*sum( (u-v).^2) ) );


len = length(model.sv_coef);
y = 0;

for i = 1:len
u = model.SVs(i,:);
y = y + model.sv_coef(i)*RBF(u,x);
end
b = -model.rho;
y = y + b;

if y >= 0
plabel = 1;
else
plabel = -1;
end
=======
### Real-time stream data fall detection by libsvm
>>>>>>> 3b87a3a23493cdbd1971114b03360e037ee896ba
