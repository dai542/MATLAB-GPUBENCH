function GPUPerformanceTest()
    try
        % 检查是否安装了Parallel Computing Toolbox
        if ~license('test', 'Parallel_Computing_Toolbox')
            error('需要安装Parallel Computing Toolbox才能运行此测试。');
        end
        
        % 检查是否有可用的GPU
        if gpuDeviceCount() == 0
            error('未检测到可用的GPU设备。');
        end
        
        % 获取所有可用的GPU设备信息
        gpuDeviceCount = gpuDeviceCount();
        fprintf('系统中发现 %d 个GPU设备\n', gpuDeviceCount);
        
        % 测试参数设置
        matrixSize = 2048;
        iterations = 5;
        
        % 创建结构体数组存储所有结果
        results = struct();
        
        for deviceID = 1:gpuDeviceCount
            try
                gpu = gpuDevice(deviceID);
                fprintf('\n测试GPU %d: %s\n', deviceID, gpu.Name);
                
                results(deviceID).DeviceID = deviceID;
                results(deviceID).Name = gpu.Name;
                
                % 1. 带宽测试
                [uploadBW, downloadBW] = testHostToGPUBandwidth(matrixSize);
                results(deviceID).UploadBandwidth = uploadBW;
                results(deviceID).DownloadBandwidth = downloadBW;
                
                % 2. 内存密集型操作
                [transposeTime, reshapeTime, indexTime] = testMemoryIntensiveOperations(matrixSize, iterations);
                results(deviceID).TransposeTime = transposeTime;
                results(deviceID).ReshapeTime = reshapeTime;
                results(deviceID).IndexTime = indexTime;
                
                % 3. 计算密集型操作
                [multTime, fftTime, mathTime] = testComputeIntensiveOperations(matrixSize, iterations);
                results(deviceID).MatrixMultTime = multTime;
                results(deviceID).FFTTime = fftTime;
                results(deviceID).MathTime = mathTime;
                
                % 4. GPU内存操作
                [allocTime, clearTime, copyTime] = testGPUMemoryOperations(matrixSize);
                results(deviceID).AllocTime = allocTime;
                results(deviceID).ClearTime = clearTime;
                results(deviceID).CopyTime = copyTime;
                
                % 清理GPU内存
                reset(gpu);
                
            catch ME
                fprintf('GPU %d 测试失败: %s\n', deviceID, ME.message);
                % 记录失败的结果
                results(deviceID).DeviceID = deviceID;
                results(deviceID).Name = '测试失败';
                results(deviceID).Error = ME.message;
                continue;
            end
        end
        
        % 显示结果表格
        displayResultsTable(results);
        
    catch ME
        fprintf('测试过程中发生错误：\n');
        fprintf('错误信息：%s\n', ME.message);
        fprintf('错误详情：\n');
        disp(getReport(ME, 'extended'));
    end
end

function [uploadBW, downloadBW] = testHostToGPUBandwidth(n)
    try
        A = rand(n, n, 'single');
        
        % 测试上传速度
        tic;
        gpuA = gpuArray(A);
        uploadTime = toc;
        
        % 测试下载速度
        tic;
        B = gather(gpuA);
        downloadTime = toc;
        
        dataSize = numel(A) * 4 / (1024^3); % 数据大小(GB)
        uploadBW = dataSize / uploadTime;
        downloadBW = dataSize / downloadTime;
        
    catch ME
        uploadBW = NaN;
        downloadBW = NaN;
        rethrow(ME);
    end
end

function [transposeTime, reshapeTime, indexTime] = testMemoryIntensiveOperations(n, iterations)
    try
        A = gpuArray(rand(n, n, 'single'));
        B = gpuArray(rand(n, n, 'single'));
        
        % 1. 矩阵转置
        times = zeros(1, iterations);
        for i = 1:iterations
            tic;
            C = A';
            wait(gpuDevice);
            times(i) = toc;
        end
        transposeTime = mean(times);
        
        % 2. 矩阵重排
        times = zeros(1, iterations);
        for i = 1:iterations
            tic;
            C = reshape(A, n, n);
            wait(gpuDevice);
            times(i) = toc;
        end
        reshapeTime = mean(times);
        
        % 3. 矩阵索引
        times = zeros(1, iterations);
        idx = gpuArray(randperm(n));
        for i = 1:iterations
            tic;
            C = A(idx,:);
            wait(gpuDevice);
            times(i) = toc;
        end
        indexTime = mean(times);
        
    catch ME
        transposeTime = NaN;
        reshapeTime = NaN;
        indexTime = NaN;
        rethrow(ME);
    end
end

function [multTime, fftTime, mathTime] = testComputeIntensiveOperations(n, iterations)
    try
        A = gpuArray(rand(n, n, 'single'));
        B = gpuArray(rand(n, n, 'single'));
        
        % 1. 矩阵乘法
        times = zeros(1, iterations);
        for i = 1:iterations
            tic;
            C = A * B;
            wait(gpuDevice);
            times(i) = toc;
        end
        multTime = mean(times);
        
        % 2. FFT
        times = zeros(1, iterations);
        for i = 1:iterations
            tic;
            C = fft2(A);
            wait(gpuDevice);
            times(i) = toc;
        end
        fftTime = mean(times);
        
        % 3. 复杂数学运算
        times = zeros(1, iterations);
        for i = 1:iterations
            tic;
            C = exp(A) .* log(abs(B) + 1) + sqrt(abs(A.*B));
            wait(gpuDevice);
            times(i) = toc;
        end
        mathTime = mean(times);
        
    catch ME
        multTime = NaN;
        fftTime = NaN;
        mathTime = NaN;
        rethrow(ME);
    end
end

function [allocTime, clearTime, copyTime] = testGPUMemoryOperations(n)
    try
        % 1. 测试内存分配速度
        tic;
        A = gpuArray(rand(n, n, 'single'));
        allocTime = toc;
        
        % 2. 测试内存清理速度
        tic;
        clear A;
        clearTime = toc;
        
        % 3. 测试GPU内存拷贝
        A = gpuArray(rand(n, n, 'single'));
        tic;
        B = A;
        copyTime = toc;
        
    catch ME
        allocTime = NaN;
        clearTime = NaN;
        copyTime = NaN;
        rethrow(ME);
    end
end

function displayResultsTable(results)
    try
        fprintf('\n\n性能测试结果汇总:\n');
        fprintf('==================================================================================================================================\n');
        
        % 表头
        fprintf('%-3s %-20s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n', ...
            'ID', 'GPU名称', '上传带宽', '下载带宽', '转置', '重排', '索引', '矩阵乘', 'FFT', '数学运算', '内存分配', '内存拷贝');
        fprintf('%-3s %-20s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n', ...
            '', '', '(GB/s)', '(GB/s)', '(ms)', '(ms)', '(ms)', '(ms)', '(ms)', '(ms)', '(ms)', '(ms)');
        fprintf('==================================================================================================================================\n');
        
        % 数据行
        for i = 1:length(results)
            if isfield(results(i), 'Error')
                % 如果测试失败，显示错误信息
                fprintf('%-3d %-20s %-8s\n', ...
                    results(i).DeviceID, ...
                    '测试失败', ...
                    results(i).Error);
            else
                % 显示正常的测试结果
                fprintf('%-3d %-20s %-8.2f %-8.2f %-8.2f %-8.2f %-8.2f %-8.2f %-8.2f %-8.2f %-8.2f %-8.2f\n', ...
                    results(i).DeviceID, ...
                    shortenString(results(i).Name, 20), ...
                    results(i).UploadBandwidth, ...
                    results(i).DownloadBandwidth, ...
                    results(i).TransposeTime * 1000, ...
                    results(i).ReshapeTime * 1000, ...
                    results(i).IndexTime * 1000, ...
                    results(i).MatrixMultTime * 1000, ...
                    results(i).FFTTime * 1000, ...
                    results(i).MathTime * 1000, ...
                    results(i).AllocTime * 1000, ...
                    results(i).CopyTime * 1000);
            end
        end
        fprintf('==================================================================================================================================\n');
        fprintf('注: 所有时间均以毫秒(ms)为单位显示，带宽以GB/s为单位\n');
        fprintf('    NaN 表示该测试项目失败\n');
        
    catch ME
        fprintf('显示结果表格时发生错误：%s\n', ME.message);
    end
end

function str = shortenString(str, maxLength)
    if length(str) > maxLength
        str = [str(1:maxLength-3) '...'];
    end
end