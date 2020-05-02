clear;
clc;

testing = xlsread('testing.xlsx'); %read data set 
training = xlsread('training.xlsx'); %read data set 

%The MLP Neural Network contains 4 layers. The input layer, two hidden layer,
%and the output layer. We will use this NN to predict Energy Consumption in
%buildings. 


x_input = testing(:,1:3); %input matrix of the network
target_outputs = testing(:,4); %target outputs of the network 
neurons_per_layer = [3 9 1]; %number of neurons inside each layer (input, hidden, output) 
num_of_layers = 2; %All layers but input layer 
num_samples = 300; 
L = 2; % last layer of the network  
rng(0,'twister'); 
k = 1; %sentinel variable 
initial_lr = 0.0003595946;
decay = 0.001; 

 
g_log = @(x) 1 / (1+exp(-x)); % logistic function 
%g_log = @(x) max(0,x); 
g_linear = @(x) x; % linear function 

%Initialize the weigths for each layer: 
for i = 1:num_of_layers
   w_layer{i} = initialize_weigths(neurons_per_layer,i+1); %initialize weigths at the beginning, then update inside loop
   w_layer_aug{i} = aug_vec(w_layer{i}); %augment weigths in each layer
end

%Start Forward Propagation 
x_input = aug_vec(x_input); %augment input matrix with column of 1's 

for k = 1:2000
    
a{1} = x_input * w_layer_aug{1}.'; %compute activation matrix of layer 1 

%start for loop 
for i= 2:L
    y{i-1} = arrayfun(g_log, a{i-1});  %compute output matrix of previous layer l-1
    y_aug{i-1} = aug_vec(y{i-1}); %augment output matrix of previous layer 
    a{i} = y_aug{i-1} * w_layer_aug{i}.';  %compute activation matrix of current layer
end

y{L} = arrayfun(g_linear, a{L}); % compute output matrix  of last layer 
err(k) = immse(y{L},target_outputs);%compute MSE
err_log(k) = log10(err(k));

%End of forward propagation 

%Start Error backpropagation
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


figure(1);
plot(err_log, 'r','LineWidth',2.5);
title('MSE vs iterations','FontSize', 20);
xlabel('iterations','FontSize', 18);
ylabel('MSE (log_10)', 'FontSize', 18);


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













 







