%%%%Implementation of Logistic Classification on Iris Dataset
%%%%@Author: Aadesh Neupane
%%%%Codes over here are based on Prof. Andrew Ng ML Class
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
1;
printf("Usages: octave logisticCls.m training.csv test.csv\n");
printf ("\n");
arg_list = argv();
for i = 1:nargin
  printf ("%s", arg_list{i});
  source_fname=arg_list{i};
  test_fname=arg_list{i};
endfor


function sigm=sigmoid(z)
  sigm=zeros(size(z));
  sigm=1./(1+e .^ -z);
endfunction

function pred=predict(theta,X)
  m=size(X,1);
  pred=zeros(m,1);
  pred=round(sigmoid(X*theta));
endfunction

function [theta,cost,exit_flag]=callFminunc(theta,X,y,lamda)
  options=optimset('GradObj','on','MaxIter',400);
[theta,cost,exit_flag]=fminunc(@(t)(costFunctionReg(t, X, y,lamda)),theta, options);
endfunction

function [cost,grad]=costFunctionReg(theta,X,y,lamda)
  m=length(y);
  cost=0;
  grad=zeros(size(theta));
  [cost,grad]=costFunction(theta,X,y);
  penalize=sum(theta(2:end) .^2);
  cost=cost+lamda/(2*m)*penalize;
  grad(2:end)=grad(2:end)+(lamda/m)*theta(2:end);
endfunction

function [cost,grad]=costFunction(theta,X,y)
  %Function that calculates the cost and gradient for logistic regression
  m=length(y);
  cost=0;
  grad=zeros(size(theta));
  h0=sigmoid(X*theta);
  cost=(1/m) * sum(-y.* log(h0)-(1-y).*log(1-h0));
  grad=(1/m)*(X'*(h0-y));
endfunction  

function [all_theta]=oneVsAll(X,y,num_labels,lamda)
  m=size(X,1);
  n=size(X,2);
  all_theta=zeros(num_labels,n+1);
  X=[ones(m,1) X];
  initial_theta=zeros(n+1,1);
  options=optimset('GradObj','on','MaxIter','60');
  for c = 1:num_labels
    all_theta(c,:)=fmincg(@(t)(costFunctionReg(t,X,(y==c),lamda)),initial_theta,options);
  end
endfunction

function pre=predictOneVsAll(all_theta,X)
  m=size(X,1);
  num_labels=size(all_theta,1);
  pre=zeros(size(X,1),1);
  X=[ones(m,1) X];
  [values, pre]=max((X*all_theta'),[],2);
endfunction
  
function [maindata,E] = loadData(source_fname)
  %%Function which loads the source data
  [A,B,C,D,E]=textread(source_fname,'%f %f %f %f %f','delimiter',',');
  maindata=[A B C D];
endfunction

function [m,n,X,initial_theta]=manupulateData(X)
  %%Function which manupulates data to add one base term
  [m,n]=size(X);
  X=[ones(m,1) X];
  initial_theta=zeros(n+1,1);
endfunction

%%Initial variable defination
inLyrSize=4;
num_labels=3;
lamda=0.1;
[maindata,label]=loadData(source_fname);
size(maindata)
size(label)

[all_theta]=oneVsAll(maindata,label,num_labels,lamda);
%[m,n,X,theta]=manupulateData(maindata);
%[cost,grad]=costFunction(theta,X,label);
%[thetafinal,costfinal,exit_flag]=callFminunc(theta,X,label,lamda);
%%Regression completed

%Time for prediction
[testdata,tlabel]=loadData(test_fname);
%[m1,n1,X1,theta1]=manupulateData(maindata);
pred=predictOneVsAll(all_theta,testdata);
fprintf('Train Accuracy: %f\n', mean(double(pred == tlabel)) * 100);

