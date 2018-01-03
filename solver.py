import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import model
import dataloader
import util

class Solver:
    def __init__(self, root_dir='data', sub_dir='train', batch_size=16, D_lr=0.0001, G_lr=0.0002, lr_decay_epoch=10, cyc_lambda=8, cls_lambda=1, num_epoch=500):

        """
            < Variables >
            
            1. self.dtype : Data type
            2. self.ltype : Label type
            3. self.D_M : Discriminator for male
            4. self.G_MF : Generator which converts male into female
            5. self.dloader : Data loader
            6. self.lr_decay_epoch : Every this epoch, learning rate will be decreased
            7. self.cyc_lambda : Weight for cycle consistency loss
            8. self.cls_lambda : Weight for gender classification loss
            9. self.criterion_gan : Loss function for GAN loss(Mean squared error). 
            10. self.criterion_cyc : Loss function for Cycle consistency loss(L1 loss).
            11. self.criterion_cls : Loss function for Gender classification loss(Cross entropy loss).
        """
        
        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
            self.ltype = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.ltype = torch.LongTensor
            
        self.D_M = model.Discriminator().type(self.dtype)
        self.D_F = model.Discriminator().type(self.dtype)
        self.G_MF = model.Generator().type(self.dtype)
        self.G_FM = model.Generator().type(self.dtype)

        self.dloader, _ = dataloader.getDataLoader(root_dir, sub_dir, batch=batch_size, shuffle=True)
        self.D_lr = D_lr
        self.G_lr = G_lr

        self.lr_decay_epoch = lr_decay_epoch
        self.cyc_lambda = cyc_lambda
        self.cls_lambda = cls_lambda
        self.num_epoch = num_epoch
        
        self.optim_D_M = optim.Adam(self.D_M.parameters(), lr=self.D_lr, betas=(0.5, 0.999))
        self.optim_D_F = optim.Adam(self.D_F.parameters(), lr=self.D_lr, betas=(0.5, 0.999))
        self.optim_G_MF = optim.Adam(self.G_MF.parameters(), lr=self.G_lr, betas=(0.5, 0.999))
        self.optim_G_FM = optim.Adam(self.G_FM.parameters(), lr=self.G_lr, betas=(0.5, 0.999))
        
        self.criterion_gan = nn.MSELoss()
        self.criterion_cyc = nn.L1Loss()
        self.criterion_cls = nn.CrossEntropyLoss()
    
    # load weight
    def load_pretrained(self, epoch=0):
        print('loading weight from epoch %d ...' % epoch)
        dir_name = 'pretrained/epoch_' + str(epoch)
        self.D_M.load_state_dict(torch.load(dir_name + '/D_M.pkl'))
        self.D_F.load_state_dict(torch.load(dir_name + '/D_F.pkl'))
        self.G_MF.load_state_dict(torch.load(dir_name + '/G_MF.pkl'))
        self.G_FM.load_state_dict(torch.load(dir_name + '/G_FM.pkl'))     
        
    # zero_grad() for all optimizers    
    def all_optim_zero_grad(self):
        self.optim_D_M.zero_grad()
        self.optim_D_F.zero_grad()
        self.optim_G_MF.zero_grad()
        self.optim_G_FM.zero_grad()
    
    # learning rate scheduler.
    def lr_scheduler(self, optim, epoch, init_lr, lr_decay_epoch=5):
        lr = init_lr * (0.1 ** (epoch / lr_decay_epoch))
        
        if epoch % lr_decay_epoch == 0:
            print(str(optim.__class__.__name__), end='')
            print(' lr is set to %f' % lr)
        
        for param_group in optim.param_groups:
            param_group['lr'] = lr
    
    # Set optimizer using lr_scheduler
    def set_optimizer(self, epoch):
        self.lr_scheduler(self.optim_D_M, epoch, self.D_lr, lr_decay_epoch=self.lr_decay_epoch)
        self.lr_scheduler(self.optim_D_F, epoch, self.D_lr, lr_decay_epoch=self.lr_decay_epoch)
        self.lr_scheduler(self.optim_G_MF, epoch, self.G_lr, lr_decay_epoch=self.lr_decay_epoch)
        self.lr_scheduler(self.optim_G_FM, epoch, self.G_lr, lr_decay_epoch=self.lr_decay_epoch)
    
    # Train
    def train(self, load_weight=False, print_every=500):
        MALE, FEMALE = 0, 1
        FAKE, REAL = 0, 1
        LAST_EPOCH = 0
        
        # If pretrained weights exist, load them and get last epoch and iteration
        if load_weight is True:
            LAST_EPOCH = int(util.read_log())
            self.load_pretrained(LAST_EPOCH)
        
        for epoch in range(LAST_EPOCH, self.num_epoch):
            self.set_optimizer(epoch)
                
            for iters, (image, label) in enumerate(self.dloader):  
                # If a batch has only female or male images not mixed, just discard that batch.
                # It seems that it makes results even worse.
                if torch.sum(label==0) == 0 or torch.sum(label==1) == 0:
                    continue

                male_num = torch.sum(label == 0)
                female_num = torch.sum(label == 1)

                image, label = image.type(self.dtype), label.type(self.ltype)
                male_img, female_img = util.gender_divider(image, label)
                
                image, label = Variable(image), Variable(label)
                male_img, female_img = Variable(male_img), Variable(female_img)
                
                
                """
                1. Train D_M (Discriminator for male)

                Step 1. Hope D_M(male_img) == 1
                Step 2. Hope D_M(fake_img) == 0
                Step 3. Minimize classification loss
                """
                # D_M(male_img) == 1
                real_loss, fake_loss = 0, 0
                if male_num is not 0:
                    real_score, _ = self.D_M(male_img)
                    real_label = util.make_label(real_score.size(), label=REAL, noisy=True).type(self.dtype)
                    real_label = Variable(real_label)

                    # Loss for real male image
                    real_loss = self.criterion_gan(real_score, real_label)
                
                # Hope D_M(fake_img) == 0
                if female_num is not 0:
                    fake_img = self.G_FM(female_img)
                    fake_score, _ = self.D_M(fake_img)
                    fake_label = util.make_label(fake_score.size(), label=FAKE, noisy=False).type(self.dtype)
                    fake_label = Variable(fake_label)

                    # Loss for fake male image
                    fake_loss = self.criterion_gan(fake_score, fake_label)
                
                # Minimize classofication loss
                _, gender_score = self.D_M(image)
                cls_loss = self.criterion_cls(gender_score, label)

                # Final D_M loss(Multitask learning)
                D_loss = real_loss + fake_loss + cls_loss
                
                # Update
                self.all_optim_zero_grad()
                D_loss.backward()
                self.optim_D_M.step()
                
                
                """
                2. Train D_F (Discriminator for female)

                Step 1. Hope D_F(female_img) == 1
                Step 2. Hope D_F(fake_img) == 0
                Step 3. Minimize classification loss
                """
                # Hope D_F(female_img) == 1
                real_loss, fake_loss = 0, 0
                if female_num is not 0:
                    real_score, _ = self.D_F(female_img)
                    real_label = util.make_label(real_score.size(), label=REAL, noisy=True).type(self.dtype)
                    real_label = Variable(real_label)

                    # Loss for real female image
                    real_loss = self.criterion_gan(real_score, real_label)
                
                # Hope D_F(fake_img) == 0
                if male_num is not 0:
                    fake_img = self.G_MF(male_img)
                    fake_score, _ = self.D_F(fake_img)
                    fake_label = util.make_label(fake_score.size(), label=FAKE, noisy=False).type(self.dtype)
                    fake_label = Variable(fake_label)

                    # Loss for fake female image
                    fake_loss = self.criterion_gan(fake_score, fake_label)
                
                # Minimize classification loss
                _, gender_score = self.D_F(image)
                cls_loss = self.criterion_cls(gender_score, label)
                          
                # Final D_F loss
                D_loss = real_loss + fake_loss + cls_loss
                
                # Get classification accuracy
                accuracy = util.get_cls_accuracy(gender_score.data, label.data)
                
                # Update
                self.all_optim_zero_grad()
                D_loss.backward()
                self.optim_D_F.step()
                
                
                """
                3. Traing G_MF, G_FM with process of
                   <1> Male(Real image) -> <2> Female(Fake image) -> <3> Male(Cycle)
                   
                Step 1. Hope D_F(<2>) == 1
                Step 2. Hope <2> to be classified as female
                Step 3. Hope <1> == <3>
                """
                if male_num is not 0:
                    fake_img = self.G_MF(male_img)
                    fake_score, gender_score = self.D_F(fake_img)

                    # Hope D_F((2)) == 1
                    real_label = util.make_label(fake_score.size(), label=REAL, noisy=False).type(self.dtype)
                    real_label = Variable(real_label)
                    gan_loss = self.criterion_gan(fake_score, real_label)

                    # Hope <2> to be classified as female
                    female_label = util.make_label(gender_score.size(0), label=FEMALE,
                                                   noisy=False).type(self.ltype)
                    female_label = Variable(female_label)
                    cls_loss = self.cls_lambda * self.criterion_cls(gender_score, female_label)

                    # Hope <1> == <3>
                    cycle_img = self.G_FM(fake_img)
                    cyc_loss = self.cyc_lambda * self.criterion_cyc(cycle_img, male_img)

                    # Final loss
                    G_loss = gan_loss + cls_loss + cyc_loss

                    # Update
                    self.all_optim_zero_grad()
                    G_loss.backward()
                    self.optim_G_MF.step()
                    self.optim_G_FM.step()

                    
                """
                4. Traing G_MF, G_FM with process of
                   <1> Female(Real image) -> <2> Male(Fake image) -> <3> Female(Cycle)
                   
                Step 1. Hope D_M(<2>) == 1
                Step 2. Hope <2> to be classified as male
                Step 3. Hope <1> == <3>
                """
                if female_num is not 0:
                    fake_img = self.G_FM(female_img)
                    fake_score, gender_score = self.D_M(fake_img)

                    # Hope D_M(<2>) == 1
                    real_label = util.make_label(fake_score.size(), label=REAL, noisy=False).type(self.dtype)
                    real_label = Variable(real_label)
                    gan_loss = self.criterion_gan(fake_score, real_label)

                    # Hope <2> to be classified as male
                    male_label = util.make_label(gender_score.size(0), label=MALE, 
                                                 noisy=False).type(self.ltype)
                    male_label = Variable(male_label)
                    cls_loss = self.cls_lambda * self.criterion_cls(gender_score, male_label)

                    # Hope <1> == <3>
                    cycle_img = self.G_MF(fake_img)
                    cyc_loss = self.cyc_lambda * self.criterion_cyc(cycle_img, female_img)

                    # Final loss
                    G_loss = gan_loss + cls_loss + cyc_loss

                    # Update
                    self.all_optim_zero_grad()
                    G_loss.backward()
                    self.optim_G_MF.step()
                    self.optim_G_FM.step()
                
                if iters % print_every == 0:
                    util.print_result(epoch, iters, accuracy, D_loss, G_loss, gan_loss, cyc_loss, cls_loss)
            
            # Save parameters
            util.save_weight(self.D_M, self.D_F, self.G_MF, self.G_FM, num_epoch=epoch)
            util.write_log(epoch)