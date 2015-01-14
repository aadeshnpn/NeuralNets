%%%%Implementation of Feed Forward Neural Networks
%%%%@Author: Aadesh Neupane
%%%%Codes over here are based on Prof. Andrew Ng ML Class
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
1;

arg_list = argv();
for i = 1:nargin
  printf ("%s", arg_list{i});
  source_fname=arg_list{i};
  test_fname=arg_list{i};
endfor
printf ("\n");

function pred=predict(theta1,theta2,X)
  m=size(X,1);
  num_lab=size(theta2,1);
  pred=zeros(size(X,1),1);
  h1=sigmoid([ones(m,1) X]*theta1');
  h2=sigmoid([ones(m,1) h1]*theta2');
  [dummy, pred]=max(h2,[],2);
endfunction

function m= randInitWeight(m_in,m_out)
  m=zeros(m_out,1+m_in);
  eplison_init=0.12;
  m=rand(m_out,1+m_in)*2*eplison_init-eplison_init;
endfunction

function g=sigmoidGradient(z)
  g=zeros(size(z));
  g=sigmoid(z).*(1-sigmoid(z));
endfunction

function sigm=sigmoid(z)
  sigm=zeros(size(z));
  sigm=1./(1+e .^ -z);
endfunction

function [cost grad]=nnCostFunc(nn_para,input_layer_size,hidden_layer_size,num_labels,X,y,lamda)
  m=size(X,1);
  cost=0.01;
  theta1 = reshape(nn_para(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

  theta2 = reshape(nn_para((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
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
  reg=sum(sum(theta1(:,2:end) .^2)) + sum(sum(theta2(:,2:end) .^2));
  cost=cost+(lamda/(2*m))*reg;
  %%Backward Propagation
  delta_3=A3-Y;
  delta_2=(theta2'*delta_3)(2:end,:).* sigmoidGradient(Z2);
  theta1_un_grad=(delta_2*A1')/m;
  theta2_un_grad=(delta_3*A2')/m;
  theta1_grad=theta1_un_grad+(lamda/m)*theta1;
  theta2_grad=theta2_un_grad+(lamda/m)*theta2;
  theta1_grad(:,1)=theta1_un_grad(:,1);
  theta2_grad(:,1)=theta2_un_grad(:,1);
  #grad1=(1/m)*(theta1*(h0-y));
  #grad2=(1/m)*(theta2*(h0-y));
  grad=[theta1_grad(:) ; theta2_grad(:)];
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
init_theta1=randInitWeight(input_layer_size,hidden_layer_size);
init_theta2=randInitWeight(hidden_layer_size,num_labels);
init_nn_paras=[init_theta1(:) ; init_theta2(:)];

lamda=0.1;
%%size(theta1)
%%size(theta2)


%%Feed forward neural networks implementaion
%cost=nnCostFunc(nn_para,input_layer_size,hidden_layer_size,num_labels,X,y,lamda);
%cost;
%g=sigmoidGradient([1 -0.5 0 0.5 1]);

options=optimset('MaxIter',100);

costFunc=@(p) nnCostFunc(p,input_layer_size,hidden_layer_size,num_labels,X,y,lamda);
[nn_params,cost]=fminunc(costFunc,init_nn_paras,options);
nn_params;
cost;
theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


[X,y]=loadData(test_fname);
pred=predict(theta1,theta2,X);
fprintf('\nTraining Accuracy: %f\n', mean(double(pred == y)) * 100);
