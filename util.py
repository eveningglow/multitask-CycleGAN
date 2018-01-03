import torch
import numpy as np
import os
import time

def print_time():
    now = time.localtime()
    print("%04d-%02d-%02d / %02d:%02d:%02d\n" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    
def print_result(epoch, iters, accuracy, D_loss, G_loss, gan_loss, cyc_loss, cls_loss):
    print_time()
    print('< epoch : %d / iter : %d >\n' %(epoch, iters))
    print('===============================================')
    print('1. Classification Training\n')
    print('Accuracy : %f' % accuracy)
    print('\n')
    print('2. GAN Training\n')
    print('D loss : %f' % (D_loss.data[0]))
    print('G loss : %f' % (G_loss.data[0]))
    print('  * gan loss : %f' % (gan_loss.data[0]))
    print('  * cyc loss : %f' % (cyc_loss.data[0]))
    print('  * cls loss : %f' % (cls_loss.data[0]))
    print('===============================================\n')

# Make label using smooth labeling
def make_label(size, label=0, noisy=True):
    if noisy == True:
        out = torch.rand(size)
        
        # noisy label with range of (0, 0.3)
        if label == 0:
            out *= (3/10)
            
        # noisy label with range of (0.7, 1.2)
        else:
            out /= 2
            out += 0.7
    else:
        if label == 0:
            out = torch.zeros(size)
        else:
            out = torch.ones(size)
    
    return out
    
# Get gender classification accuracy
def get_cls_accuracy(score, label):
    total = label.size()[0]
    _, pred = torch.max(score, dim=1)
    correct = torch.sum(pred == label.cuda())
    accuracy = correct / total

    return accuracy

# Read log and find the last epoch number
def read_log():
    f = open('log.txt', 'r')
    line = f.readline()   
    epoch = int(line)
    
    f.close()
    
    return epoch

# Write current epoch number
def write_log(epoch):
    f = open('log.txt', 'w')
    data = str(epoch)
    f.write(data)
    f.close()

# Divide the batch into male and female
def gender_divider(img, label):
    dtype = type(img)

    male_num = torch.sum(label == 0)
    female_num = torch.sum(label == 1)
    
    img = img.cpu().numpy()
    label = label.cpu().numpy()
    
    if male_num is 0:
        male_img = torch.FloatTensor()
        female_img = torch.from_numpy(img[label == 1])
    elif female_num is 0:
        male_img = torch.from_numpy(img[label == 0])
        female_img = torch.FloatTensor()
    else:
        male_img = torch.from_numpy(img[label == 0])
        female_img = torch.from_numpy(img[label == 1])
    
    male_img = male_img.type(dtype)
    female_img = female_img.type(dtype)
    
    return male_img, female_img

# Save weight
def save_weight(D_M, D_F, G_MF, G_FM, num_epoch=0):
    if not os.path.exists('./pretrained'):
        os.makedirs('./pretrained')

    # Save weight at every epoch
    dir_name = './pretrained/epoch_' + str(num_epoch)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    torch.save(D_M.state_dict(), dir_name + '/D_M.pkl')
    torch.save(D_F.state_dict(), dir_name + '/D_F.pkl')
    torch.save(G_MF.state_dict(), dir_name + '/G_MF.pkl')
    torch.save(G_FM.state_dict(), dir_name + '/G_FM.pkl')       