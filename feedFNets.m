%%%%Implementation of Feed Forward Neural Networks
%%%%@Author: Aadesh Neupane
1;

arg_list = argv();
for i = 1:nargin
  printf ("%s", arg_list{i});
  source_fname=arg_list{i};
test_fname=arg_list{i};
endfor
printf ("\n");

function sigm=sigmoid(z)
  sigm=zeros(size(z));
  sigm=1./(1+e .^ -z);
endfunction

function sigGrad=sigmoidGradient(z)
  g=zeros(size(z));
  g=sigmoid(z).*(1-sigmoid(z));
endfunction

function [cost grad]=nnCostFunc(theta1,theta2,input_layer_size,hidden_layer_size,num_labels,X,y,lamda)
  m=size(X,1);
  cost=0;
  theta1_grad=zeros(size(theta1));
  theta2_grad=zeros(size(theta2));
  no_of_classes=length(unique(y));
  Y=zeros(no_of_classes,m);
  for i=1:m
	Y(y(i),i)=1;
  endfor

%%Forward Propagation
  A1=[ones(1,m);X'];
  Z2=theta1*A1;
  A2=[ones(1,m);sigmoid(Z2)];
  Z3=theta2*A2;
  A3=sigmoid(Z3);
  h0=A3;
  %%Cost function
  cost=(1/m)*sum(sum(-Y.*log(h0) - (1-Y).*log(1-h0)));
  #grad1=(1/m)*(theta1*(h0-y));
  #grad2=(1/m)*(theta2*(h0-y));
  grad=[theta1_grad(:); theta2_grad(:)];
endfunction

function [maindata,E] = loadData(source_fname)
  %%Function which loads the source data
  [A,B,C,D,E]=textread(source_fname,'%f %f %f %f %f','delimiter',',');
  maindata=[A B C D];
endfunction

%%Initial Parameters
input_layer_size=4;
hidden_layer_size=1;
num_labels=3;

%%load data
[X,y]=loadData(source_fname);
m=size(X,1);
n=size(y,1);
theta1=rand(hidden_layer_size,input_layer_size+1);
theta2=rand(num_labels,hidden_layer_size+1);
lamda=0;
%%size(theta1)
%%size(theta2)
%%nn_params=[theta1(:);theta2(:)];

%%Feed forward neural networks implementaion
cost=nnCostFunc(theta1,theta2,input_layer_size,hidden_layer_size,num_labels,X,y,lamda);
cost

%%Sigmoid Gradient
%%sigGra=sigmoidGradient([1 -0.5 0 0.5 1]);

%%

