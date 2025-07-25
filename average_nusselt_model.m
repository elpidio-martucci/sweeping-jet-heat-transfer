close, clear, clc,
% Define the matrix
data1 = [
    1.7e-2, 6.25, 30, 3600, 2.0, 61.7, 45.50;
    1.7e-2, 6.25, 45, 6400, 2.0, 78.9, 58.83;
    1.7e-2, 6.25, 68, 11000, 2.0, 90.5, 71.93;
    1.7e-2, 6.25, 73, 15300, 2.0, 104.7, 85.68;
    1.5e-2, 5.00, 47, 8000, 3.0, 36.9, 38.90;
    1.5e-2, 5.00, 47, 8000, 5.0, 31.3, 31.66;
    1.5e-2, 5.00, 47, 8000, 8.0, 27.8, 27.22;
    %1.5e-2, 5.00, 47, 8000, 10.0, 26.3, 26.40;
    1.5e-2, 5.00, 70, 12000, 3.0, 44.2, 45.72;
    1.5e-2, 5.00, 70, 12000, 5.0, 34.1, 34.72;
    1.5e-2, 5.00, 70, 12000, 8.0, 29.5, 29.19;
    1.5e-2, 5.00, 70, 12000, 10.0, 27.6, 27.67;
    1.5e-2, 5.00, 74, 16000, 3.0, 57.0, 61.81;
    1.5e-2, 5.00, 74, 16000, 5.0, 40.6, 41.16;
    1.5e-2, 5.00, 74, 16000, 8.0, 32.3, 32.85;
    1.5e-2, 5.00, 74, 16000, 10.0, 30.1, 30.39;
    %1.5e-2, 5.00, 77, 24000, 3.0, 89.3, 94.42;
    1.5e-2, 5.00, 77, 24000, 5.0, 56.3, 57.06;
    1.5e-2, 5.00, 77, 24000, 8.0, 42.8, 42.62;
    1.5e-2, 5.00, 77, 24000, 10.0, 38.9, 39.66;
    %1.5e-2, 5.00, 67, 32000, 3.0, 126.3, 124.18;
    1.5e-2, 5.00, 67, 32000, 5.0, 70.1, 68.04;
    1.5e-2, 5.00, 67, 32000, 8.0, 50.0, 48.61;
    1.5e-2, 5.00, 67, 32000, 10.0, 45.9, 44.99;
    4.8e-2, 2.67, 49, 3460, 3.0, 35.6, 27.70;
    4.8e-2, 2.67, 49, 3460, 4.0, 29.7, 22.23;
    4.8e-2, 2.67, 49, 3460, 6.0, 20.8, 18.66;
    4.8e-2, 2.67, 49, 3460, 8.0, 17.5, 15.26;
    4.8e-2, 2.67, 62, 8650, 3.0, 63.5, 47.23;
    4.8e-2, 2.67, 62, 8650, 4.0, 46.5, 41.04;
    4.8e-2, 2.67, 62, 8650, 6.0, 24.4, 32.52;
    4.8e-2, 2.67, 62, 8650, 8.0, 18.4, 22.97;
    4.8e-2, 2.67, 78, 17295, 3.0, 94.3, 72.96;
    4.8e-2, 2.67, 78, 17295, 4.0, 63.5, 64.79;
    4.8e-2, 2.67, 78, 17295, 6.0, 42.3, 48.52;
    4.8e-2, 2.67, 78, 17295, 8.0, 32.9, 38.64;
    4.8e-2, 2.67, 87, 25940, 3.0, 124.2, 88.54;
    4.8e-2, 2.67, 87, 25940, 4.0, 89.8, 89.80;
    4.8e-2, 2.67, 87, 25940, 6.0, 59.0, 60.51;
    4.8e-2, 2.67, 87, 25940, 8.0, 44.2, 46.70;
    4.8e-2, 2.67, 78, 34588, 3.0, 148.0, 104.54;
    4.8e-2, 2.67, 78, 34588, 4.0, 118.8, 110.23;
    4.8e-2, 2.67, 78, 34588, 6.0, 74.7, 81.23;
    4.8e-2, 2.67, 78, 34588, 8.0, 57.5, 60.00;
    2.25e-2, 10.00, 50, 15810, 4.0, 58.67, 51.55;
    2.25e-2, 10.00, 50, 15810, 6.0, 47.31, 46.19;
    2.25e-2, 10.00, 50, 15810, 8.0, 39.63, 39.95;
    2.25e-2, 10.00, 50, 15810, 10.0, 33.17, 35.48;
];

% Remove outliers using the IQR method
Q1 = quantile(data1, 0.25); % First quartile
Q3 = quantile(data1, 0.75); % Third quartile
IQR = Q3 - Q1;               % Interquartile range

% Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR;
upper_bound = Q3 + 1.5 * IQR;

% Remove outliers
outlier_mask = all(data1 >= lower_bound & data1 <= upper_bound, 2);
cleaned_matrix = data1(outlier_mask, :);
%data = cleaned_matrix;
data = data1; % outlier rimossi manualmente perchè ne toglieva troppi

St = data(:,1);
D = data(:,2);
theta = data(:,3);
Re = data(:,4);
HD = data(:,5);
Nus = data(:,6);
Num = data(:,7);

logNus = log(Nus);
logNum = log(Num);
logSt = log(St);
logTh = log(theta);
logRe = log(Re);
logHD = log(HD);

% Construct the design matrix for linear regression
X = [logSt, logTh, logRe, logHD];

% Solve for coefficients A, B, C using linear regression
coefficientsM = (X' * X) \ (X' * logNum);

% Extract coefficients A, B, C
Am = coefficientsM(1);
Bm = coefficientsM(2);
Cm = coefficientsM(3);
Dm = coefficientsM(4);

% Display the results
disp(['Am: ', num2str(Am)]);
disp(['Bm: ', num2str(Bm)]);
disp(['Cm: ', num2str(Cm)]);
disp(['Dm: ', num2str(Dm)]);

% Predict Nu using the fitted model
predictedNum = exp(X * coefficientsM);

% Calculate residuals
residualsM = Num - predictedNum;

% Calculate R-squared
SS_total = sum((Num - mean(Num)).^2); % Total sum of squares
SS_residual = sum(residualsM.^2);     % Residual sum of squares
R_squared = 1 - (SS_residual / SS_total); % R-squared

% Calculate RMSE
RMSE = sqrt(mean(residualsM.^2)); % Root Mean Square Error

% Display R-squared and RMSE
disp(['R-squared: ', num2str(R_squared)]);
disp(['RMSE: ', num2str(RMSE)]);

% Calculate VIF for each predictor in the design matrix X
numPredictors = size(X, 2);
VIF = zeros(numPredictors, 1); % Initialize VIF array

for i = 1:numPredictors
    % Create a new design matrix excluding the i-th predictor
    X_without_i = X(:, [1:i-1, i+1:end]);
    
    % Fit a linear regression model to predict the i-th predictor
    beta = (X_without_i' * X_without_i) \ (X_without_i' * X(:, i));
    
    % Calculate the predicted values
    predicted = X_without_i * beta;
    
    % Calculate R-squared
    SS_total = sum((X(:, i) - mean(X(:, i))).^2);
    SS_residual = sum((X(:, i) - predicted).^2);
    R_squared = 1 - (SS_residual / SS_total);
    
    % Calculate VIF
    VIF(i) = 1 / (1 - R_squared);
end

% Display VIF values
disp('VIF values:');
disp(VIF);

% Check residuals for normality
figure(1);
subplot(1, 2, 1);
histogram(residualsM, 20);
xlabel('Residuals');
ylabel('Frequency');
title('Histogram of Residuals');

% Plot residuals
figure(2);
scatter(predictedNum, residualsM);
xlabel('Predicted Nu');
ylabel('Residuals');
title('Residuals vs Predicted avg. Nu');
grid on;

% Calculate the upper and lower bounds for ±15% error
lower_bound = Num * 0.85; % 85% of predicted values
upper_bound = Num * 1.15;  % 115% of predicted values

% Plot actual vs predicted Nu values
figure(3);
hold on; % Hold on to plot multiple lines
plot(Num, predictedNum, 'or', 'DisplayName', 'Predicted Nu'); % Predicted values
plot(Num, Num, '-k', 'DisplayName', 'Actual Nu'); % Actual values
plot(Num, lower_bound, '--b', 'DisplayName', 'Lower Bound (85%)'); % Lower bound
plot(Num, upper_bound, '--g', 'DisplayName', 'Upper Bound (115%)'); % Upper bound
xlabel('Actual Nu');
ylabel('Predicted Nu');
title('Actual vs Predicted Avg. Nu with ±15% Error Bounds');
grid on;
legend show; % Show legend to identify lines
hold off; % Release the hold
