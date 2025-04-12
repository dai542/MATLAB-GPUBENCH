% 获取所有可用的GPU设备信息
gpuDeviceCount = gpuDeviceCount();
fprintf('系统中发现 %d 个GPU设备\n', gpuDeviceCount);

% 用于存储每个GPU的性能结果
results = struct('DeviceID', {}, 'Name', {}, 'MatrixMultTime', {}, 'FFTTime', {});


matrixSize = 4096;          % 矩阵大小
iterations = 10;            % 迭代次数

for deviceID = 1:gpuDeviceCount
    % 选择当前GPU
    gpu = gpuDevice(deviceID);
    fprintf('\n测试GPU %d: %s\n', deviceID, gpu.Name);
    
    try
        % 1. 矩阵乘法测试
        A = gpuArray(rand(matrixSize, matrixSize, 'single'));
        B = gpuArray(rand(matrixSize, matrixSize, 'single'));
        
        % 预热GPU
        C = A * B;
        wait(gpu);
        
        % 计时测试
        matrixTime = zeros(1, iterations);
        for i = 1:iterations
            tic;
            C = A * B;
            wait(gpu);
            matrixTime(i) = toc;
        end
        avgMatrixTime = mean(matrixTime);
        
        % 2. FFT测试
        D = gpuArray(rand(matrixSize, matrixSize, 'single'));
        
        % 预热GPU
        F = fft2(D);
        wait(gpu);
        
        % 计时测试
        fftTime = zeros(1, iterations);
        for i = 1:iterations
            tic;
            F = fft2(D);
            wait(gpu);
            fftTime(i) = toc;
        end
        avgFFTTime = mean(fftTime);
        
        % 存储结果
        results(deviceID).DeviceID = deviceID;
        results(deviceID).Name = gpu.Name;
        results(deviceID).MatrixMultTime = avgMatrixTime;
        results(deviceID).FFTTime = avgFFTTime;
        
        % 清理GPU内存
        reset(gpu);
        
    catch ME
        fprintf('GPU %d 测试失败: %s\n', deviceID, ME.message);
        continue;
    end
end

% 显示结果
fprintf('\n性能测试结果汇总:\n');
fprintf('----------------------------------------\n');
fprintf('设备ID\t设备名称\t\t矩阵乘法(秒)\tFFT(秒)\n');
fprintf('----------------------------------------\n');
for i = 1:length(results)
    fprintf('%d\t%s\t%.4f\t\t%.4f\n', ...
        results(i).DeviceID, ...
        results(i).Name, ...
        results(i).MatrixMultTime, ...
        results(i).FFTTime);
end