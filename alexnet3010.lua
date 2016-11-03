----------------------MODEL INIT ------------------------------------------------------------------
require 'optim'
require 'loadcaffe'
require 'cudnn';
require 'cunn'
require 'hdf5'
require 'gnuplot'
local matio = require 'matio'


model = loadcaffe.load('deploy.prototxt', 'bvlc_alexnet.caffemodel', 'cudnn')
net=model
-- --print('Modifiednet\n' .. net:__tostring());
net:remove(23)
-- print('Modifiednet\n' .. net:__tostring());
net:remove(23)
--print('Modifiednet\n' .. net:__tostring());
net:add(nn.Linear(4096, 4)) -- 7 phases
net:add(nn.LogSoftMax()) 
print('Modifiednet\n' .. net:__tostring());
--net = net:cuda()
---------------------TRAINSET AND TESTSET ---------------------------------------------------------
print('loading 1')
t1=hdf5.open('td1.mat','r')
td1 = t1:read(''):all()
t1:close()
t2 = hdf5.open('td2.mat','r')
td2 = t2:read(''):all()
t2:close()
print('done loading 1 and 2')
t3 = hdf5.open('td3.mat','r')
td3 = t3:read(''):all()
t3:close()
t4 = hdf5.open('td4.mat','r')
td4 = t4:read(''):all()
t4:close()
t5 = hdf5.open('td5.mat','r')
td5 = t5:read(''):all()
t5:close()
t6 = hdf5.open('td6.mat','r')
td6 = t6:read(''):all()
t6:close()

print('done loading')
trainDataintm = {}
--trainDataintm.data=torch.Tensor(19748,3,224,224):zero()
--trainDataintm.label=torch.Tensor(19748,1):zero()
trainDataintm.data=torch.cat({td1.td1.data,td2.td2.data,td3.td3.data,td4.td4.data,td5.td5.data,td6.td6.data},4)
trainDataintm.label=torch.cat({td1.td1.label,td2.td2.label,td3.td3.label,td4.td4.label,td5.td5.label,td6.td6.label},2)
--trainDataintm.data:transpose(2,3)
--trainDataintm.data:transpose(1,4)
--trainDataintm.label:transpose(1,2)

print('done concat')
trainset= {}
trainset.data = trainDataintm.data
trainset.label = trainDataintm.label
trainset.data=trainset.data:transpose(2,3)
trainset.data=trainset.data:transpose(1,4)
trainset.label=trainset.label:transpose(1,2)
--trainset.label=torch.ByteTensor(3173):zero()
--for i=1,3173 do
  --    trainset.label[i] = trainDataintm.label[{ {i},{1} }]
--end
print('done reshaping traindata')
---------------------------------------------------TESTSET-------------------------

--testData1 = matio.load('testData.mat')
tst = hdf5.open('testData.mat','r')
testData1  = tst:read(''):all()
tst:close()
testData2 = testData1.testData
testset = {}
testset.data = testData2.data
testset.data = testset.data:transpose(2,3)
testset.data = testset.data:transpose(1,4)
testset.label=torch.ByteTensor(554):zero()
for i=1,554 do
       testset.label[i] = testData2.label[{ {1},{i} }]
end

print('done reshaping testdata')
---------------------------------------------------------------------------------------------------------

print(trainset)

print(testset)

trainset.data = trainset.data:double() 

mean = {}
stdv={}
for i=1,3 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
   trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end



----------------------------TRAINING-------------------------------------------------------------------

function trainset:size() 
    return self.data:size(1) 
end

criterion = nn.CrossEntropyCriterion()
net=net:cuda()
criterion=criterion:cuda()



local params, gradParams = net:getParameters()
local optimState = {learningRate = .01, learningRateDecay = 1e-7}

-- Training
-----------

--local nCols = 3
local nEpochs = 100
local batchSize = 50
tranloss = torch.Tensor(nEpochs)
xepch = torch.Tensor(nEpochs)
xepch = torch.range(1,nEpochs)
xepchd= torch.Tensor(xepch)
for epoch = 1, nEpochs do
   
   lossacc = 0
   --local shuffle = torch.randperm(trainset:size())
   local shuffle = torch.randperm(22150)
   local batch = 1
   --for batchOffset = 1, trainset:size(), batchSize do
   for batchOffset = 1,443 do
   --for batchOffset = 1,31 do

      local batchInputs = torch.Tensor(batchSize,3,224,224)
      local batchResponse = torch.Tensor(batchSize)
      for i = 1, batchSize do
         --batchInputs[i] = trainset.data[shuffle[batchOffset + i]]
         --batchResponse[i] = trainset.label[shuffle[batchOffset + i]]
         batchOff=(batchOffset-1)*batchSize
         batchInputs[i] = trainset.data[shuffle[batchOff + i]]
         batchResponse[i] = trainset.label[shuffle[batchOff + i]]
      end

      local function evaluateBatch(params)
         gradParams:zero()
         batchInputs=batchInputs:cuda()
         batchResponse=batchResponse:cuda()
         local batchEstimate = net:forward(batchInputs)
         --local batchLoss = criterion:forward(batchEstimate, batchResponse)
         batchLoss = criterion:forward(batchEstimate, batchResponse)
         local nablaLossOutput = criterion:backward(
            batchEstimate, batchResponse)
         net:backward(batchInputs, nablaLossOutput)
         print('Finished epoch: ' .. epoch .. ', batch: ' ..
                  batch .. ', with loss: ' .. batchLoss)
         return batchLoss, gradParams
      end

      optim.sgd(evaluateBatch, params, optimState)
      lossacc = lossacc + batchLoss

      batch = batch + 1
      --batchOffset = batchOffset + batchSize

    end
    lossacc=lossacc/443
    tranloss[epoch] = lossacc
end
tranlossd=torch.Tensor(tranloss)
gnuplot.plot('l',xepchd,tranlossd,'-')
torch.save('tea_cam01_baldata.t7',net)


-------------------------------------------------------------------------


-------------------------------------TESTING---------------------------------------------------------------


--testset.data = testset.data:double()

for i=1,3 do -- over each image channel
  testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction  
  testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scalingend
end
class_performance={0,0,0,0}
correct=0
for i=1,554 do
    testset.data = testset.data:cuda()
    testset.label = testset.label:cuda()
    local groundtruth = testset.label[i]
    --print(groundtruth)
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    --print(indices[1])
    if groundtruth == indices[1] then
        correct = correct + 1
        class_performance[groundtruth] = class_performance[groundtruth] + 1
      end
  end
  print(correct, 100*correct/554 .. ' % ')
  print(class_performance)





--for i=1,515 do
--      local groundtruth = testset.label[i]

--      local prediction = net:forward(testset.data[i])
--      local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
--      if groundtruth == indices[1] then
--          class_performance[groundtruth] = class_performance[groundtruth] + 1
--      end
-- end




