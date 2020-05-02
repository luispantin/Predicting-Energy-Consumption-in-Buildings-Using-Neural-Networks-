clear;
clc;

format long; 
testing = xlsread('testing.xlsx'); %read data set 
training = xlsread('training.xlsx'); %read data set 

x_data = training(:,1:2); %input matrix of the network
y_data = training(:,4); %target outputs of the network 
neurons_per_layer = [2 7 7 1]; %number of neurons inside each layer (input, hidden, output) 
num_of_layers = 3; %All layers but input layer 
num_samples = 290; 
L = 3; % last layer of the network  
rng(0,'twister'); 
N_ho = 10; %Number of samples in hold-out set
initial_lr = 0.09;
decay = 0.01; 

 
g_log = @(x) 1 / (1+exp(-x)); % logistic function 
g_linear = @(x) x; % linear function 

%Initialize the weigths for each layer: 
for i = 1:num_of_layers
   w_layer{i} = initialize_weigths(neurons_per_layer,i+1); %initialize weigths at the beginning, then update inside loop
   w_layer_aug{i} = aug_vec(w_layer{i}); %augment weigths in each layer
end

%number of k-fold validation
k = 30; 
x_start = 1;
x_end = 10; 


mse_val_vec1 = zeros(1,30); %Initialize a vector for all values of mse

for f = 1:k
    
x_input = x_data;
x_input(x_start:x_end,:)=[]; %Input set of all folds but fold k = (from row x to row y)
target_outputs = y_data; %Output Set of all folds but fold k = (from row x to row y)
target_outputs(x_start:x_end,:)=[];
holdOutSetInput = x_data(x_start:x_end,:); %Input Hold-out set for folf k = (from row x to row y)
holdOutSetOutput = y_data(x_start:x_end,:);


%Start Forward Propagation 
x_input = aug_vec(x_input); %augment input matrix with column of 1's 

%k =1; %sentinel variable 
%while(true)
for m = 1:2000
    
a{1} = x_input * w_layer_aug{1}.'; %compute activation matrix of layer 1 

%start for loop 
for i= 2:L
    y{i-1} = arrayfun(g_log, a{i-1});  %compute output matrix of previous layer l-1
    y_aug{i-1} = aug_vec(y{i-1}); %augment output matrix of previous layer 
    a{i} = y_aug{i-1} * w_layer_aug{i}.';  %compute activation matrix of current layer
end

y{L} = arrayfun(g_linear, a{L}); % compute output matrix  of last layer  
%End of forward propagation 

%Start Error backpropagation
%delta{L} = 2*(y{L} - target_outputs);  %compute sensitivity matrix of output layer
delta{L} = 2*(y{L} - target_outputs);  %compute sensitivity matrix of output layer

%start of for loop
for j=L-1:-1:1
    gradient{j+1} = (1/num_samples)*(delta{j+1}.' * y_aug{j}); %compute gradient of current layer
    G{j} = (y{j}).*(ones(size(y{j})) - y{j}); %compute activation matrix of previous layer
    delta{j} = (G{j}).*(delta{j+1} * w_layer{j+1}); %back propagate sensitivity matrix (find sensitivity matrix of previous layer)
end

gradient{1} = (1/num_samples)*(delta{1}.' * x_input); %gradient matrix of layer 1

%end of error back propagation 
learning_rate = initial_lr * (1 / (1 + decay * k)); 
a_lr(k) = learning_rate; 

%start updating weigths 
for i = 1:L
    
    w_layer_aug{i} = (w_layer_aug{i}) - (learning_rate*gradient{i}); %update augmented weigth matrix for each layer 
    w_layer{i} = w_layer_aug{i}; 
    w_layer{i}(:,size(w_layer{i},2)) = []; %update weigth matrix for each layer 
    
end

end

%compute forward propagation on validation set
holdOutSetInput = aug_vec(holdOutSetInput);
a_val{1} = holdOutSetInput* w_layer_aug{1}.'; %compute activation matrix of layer 1 

%start for loop 
for i= 2:L
    y_val{i-1} = arrayfun(g_log, a_val{i-1});  %compute output matrix of previous layer l-1
    y_aug_val{i-1} = aug_vec(y_val{i-1}); %augment output matrix of previous layer 
    a_val{i} = y_aug_val{i-1} * w_layer_aug{i}.';  %compute activation matrix of current layer
end

y_val{L} = arrayfun(g_linear, a_val{L}); % compute output matrix  of last layer 

mse_f_val = immse(y_val{L},holdOutSetOutput);%compute MSE
%mse_f_val = (1/N_ho)*((holdOutSetOutput.'- (w_layer{L}*holdOutSetInput.'))*(holdOutSetOutput - (holdOutSetInput*w_layer{L}.')));

x_start = x_start + 10;
x_end = x_end + 10;
 
 %save mse value of fold f in vector
mse_val_vec1(f) = mse_f_val;


end

mse_val_avg_model1 = (sum(mse_val_vec1))/k;


figure(1);
plot(err); 
title('TMSE vs iterations')
xlabel('iterations')
ylabel('TMSE')
 


%Function to initialize weigths randomly within a range (-1/Nl) <= Wl <= (1/Nl)
function w_init_range =  initialize_weigths(neurons,layer)
    x1 = (- 2 / neurons(layer)); 
    x2 = (2 / neurons(layer)); 
    w_init_range = (x2-x1).*rand(neurons(layer),neurons(layer-1)) + x1; 
end

%function to augment vectors/matrices 
function aug_vector = aug_vec(vector)
set_ones = ones(size(vector,1),1); 
aug_vector = [vector set_ones];  
end













 







