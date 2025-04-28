g = gpuDevice;
maxMemory = 0.1*g.AvailableMemory/1024^3;

% Declare the matrix sizes to be a multiple of 1024.

maxSizeSingle = floor(sqrt(maxMemory*1024^3/4));
maxSizeDouble = floor(sqrt(maxMemory*1024^3/8));
step = 1024;
if maxSizeDouble/step >= 10
    step = step*floor(maxSizeDouble/(5*step));
end
sizeSingle = 1024:step:maxSizeSingle;
sizeDouble = 1024:step:maxSizeDouble;

[cpu, gpu] = executeBenchmarks('double', sizeDouble);
results.sizeDouble = sizeDouble;
results.gflopsDoubleCPU = cpu;
results.gflopsDoubleGPU = gpu;

fig = figure;
ax = axes('parent', fig);
plot(ax, results.sizeDouble, results.gflopsDoubleGPU, '-x', ...
     results.sizeDouble, results.gflopsDoubleCPU, '-o')
legend('GPU', 'CPU', 'Location', 'NorthWest');
grid on;
title(ax, 'Double-precision performance')
ylabel(ax, 'Gigaflops');
xlabel(ax, 'Matrix size');
drawnow;
saveas(gcf, 'plot.png');

function [A, b] = getData(n, clz)
    fprintf('Creating a matrix of size %d-by-%d.\n', n, n);
    A = rand(n, n, clz) + 100*eye(n, n, clz);
    b = rand(n, 1, clz);
end

function time = timeSolve(A, b, waitFcn)
    tic;
    x = A\b; % #ok<NASGU> We don't need the value of x.
    waitFcn(); % Wait for operation to complete.
    time = toc;
end

function gflops = benchFcn(A, b, waitFcn)
  numReps = 3;
  time = inf;
  % We solve the linear system a few times and calculate the Gigaflops
  % based on the best time.
  for itr = 1:numReps
      tcurr = timeSolve(A, b, waitFcn);
      time = min(tcurr, time);
  end

  % Measure the overhead introduced by calling the wait function.
  tover = inf;
  for itr = 1:numReps
      tic;
      waitFcn();
      tcurr = toc;
      tover = min(tcurr, tover);
  end
  % Remove the overhead from the measured time. Don't allow the time to
  % become negative.
  time = max(time - tover, 0);
  n = size(A, 1);
  flop = 2/3*n^3 + 3/2*n^2;
  gflops = flop/time/1e9;
end

% The CPU doesn't need to wait: this function handle is a placeholder.
function waitForCpu()
end

% On the GPU, to ensure accurate timing, we need to wait for the device
% to finish all pending operations.
function waitForGpu(theDevice)
  wait(theDevice);
end

function [gflopsCPU, gflopsGPU] = executeBenchmarks(clz, sizes)
  fprintf(['Starting benchmarks with %d different %s-precision ' ...
       'matrices of sizes\nranging from %d-by-%d to %d-by-%d.\n'], ...
          length(sizes), clz, sizes(1), sizes(1), sizes(end), ...
          sizes(end));
  gflopsGPU = zeros(size(sizes));
  gflopsCPU = zeros(size(sizes));
  gd = gpuDevice;
  for i = 1:length(sizes)
      n = sizes(i);
      [A, b] = getData(n, clz);
      gflopsCPU(i) = benchFcn(A, b, @waitForCpu);
      fprintf('Gigaflops on CPU: %f\n', gflopsCPU(i));
      A = gpuArray(A);
      b = gpuArray(b);
      gflopsGPU(i) = benchFcn(A, b, @() waitForGpu(gd));
      fprintf('Gigaflops on GPU: %f\n', gflopsGPU(i));
  end
end

