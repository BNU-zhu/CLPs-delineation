import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4"
from time import time
from logger import Logger
import glob
import random
from dataloader import *
from losses import *
from model import Our_Model
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from accuracy import evaluate
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from torch.autograd import Variable as V
from Apollo_optimizer import Apollo

def train_models(train_path, save_path, batch_size, num_epochs, use_pretrained, pretrained_model_path,
                 augment_dataset, momentum, weight_decay, stepsize, gamma, lr, itersize):

    mylog = Logger('logs/' + 'model.log')
    tic = time()
    
    train_file_names = glob.glob(os.path.join(train_path, "*.tif"))
    random.shuffle(train_file_names)
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_file_names]
    train_file, val_file = train_test_split(img_ids, test_size=0.1, random_state=41)
    val_device = torch.device('cuda: 0' if torch.cuda.is_available() else "cpu")
    model = Our_Model(num_classes=1)

    if torch.cuda.device_count() > 0:
        print(torch.cuda.device_count(), "GPUs will be used for training")
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model.cuda()

    epoch_start = "0"
    if use_pretrained:
        print("Loading Model {}".format(os.path.basename(pretrained_model_path)))
        model.load_state_dict(torch.load(pretrained_model_path))
        epoch_start = os.path.basename(pretrained_model_path).split(".")[0]
        print(epoch_start)
    
    if augment_dataset:
        trainLoader = DataLoader(
            Augment(ImageFolder(train_path,train_file)),
            batch_size=batch_size,drop_last=True, shuffle=True, pin_memory=True
        )
    else:  
        trainLoader = DataLoader(
            ImageFolder(train_path,train_file),
            batch_size=batch_size,drop_last=True, shuffle=True, pin_memory=True
        )              
    
    valLoader = DataLoader(
        ImageFolder(train_path,val_file),drop_last=True,shuffle=False, pin_memory=True
    )
    optimizer = Apollo(params=model.parameters(), lr=lr)
    #optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)             
    
    
    for epoch in tqdm(
        range(int(epoch_start) + 1, int(epoch_start) + 1 + num_epochs)
    ):
        global_step = epoch * len(trainLoader)
        running_loss = 0.0
        counter = 0
        for i, (img_file_name, inputs, sample1, sample2, sample3) in enumerate(
            tqdm(trainLoader)
        ):
            model.train()
            with torch.no_grad():
                inputs = V(inputs.cuda()).float()
                sample1 = V(sample1.cuda()).float()
                sample2 = V(sample2.cuda()).float()
                sample3 = V(sample3.cuda()).float()           
            counter += 1
            with torch.set_grad_enabled(True):
                predict1, predict2, predict3 = model(inputs)
                loss1 = Region_Loss()(predict1,sample1)
                loss2 = Boundary_Loss()(predict2,sample2)
                loss3 = nn.MSELoss()(predict3,sample3)
                HUL = homoscedastic_uncertainty_loss(3)
                loss = HUL(loss1,loss2,loss3)                             
                loss = loss / itersize
                loss.float().cuda()
                loss.backward()
            if counter == itersize:
                optimizer.step()
                optimizer.zero_grad()
                counter = 0
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_file_names)
        scheduler.step()
        val_acc = evaluate(val_device, model, valLoader)
        mylog.write('********\n')
        mylog.write('epoch:' + str(epoch) + '    time:' + str(int(time() - tic)) + '\n')
        mylog.write('train_loss:' + str(epoch_loss) + '\n')
        mylog.write('val_acc:' + str(val_acc) + '\n')
        if epoch % 2 == 0:
            torch.save(
                model.state_dict(), os.path.join(save_path, str(epoch) + ".pt")
            )
            
    mylog.write('Finish!')
    mylog.close()








