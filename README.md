# semi-supervised-CycleGAN
Implementation of [CycleGAN : Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) using __pytorch__. 
Futhermore, this implementation is using __multitask learning with semi-supervised leaning__ which means utilize labels of data. This model converts male to female or female to male.
Following image shows improvements such as facial features(make-up, mustache, beard, etc) and image qualities.

<p align="center"><img width="100%" src="png/1.png" /></p>
<p align="center"> original : without semi-supervised learning / improved : with semi-supervised learning

## What are differences with original CycleGAN?
1. Batch size 1 -> 16
2. Instance Normalization -> Batch Normalization
3. Model architecture
4. Smooth labeling
5. Multitask learning with classification loss (semi-supervised learning)

## Influence
__1. Increasing Batch Size & Replacing Instance Norm with Batch Norm__  
This change makes the model recognize the difference of hair length between male and female. The generator started to draw or erase hair after applying this change.  

__2. Smooth Labeling & Model Architecture Change__  
Basicially, the discriminator easily overwhelms the generator. If it happens, the generator tries to fool the discriminator in an improper way which emphasizes the balance between discriminator and generator. To solve this problem, I used smooth labeling and changed model architecture.

__3. No Batch Norm in the First Convolution in Discriminator__  
DCGAN suggests not to use normalization in the first convolution. If you don't follow this, the generator will make images with range of approximately -0.7 ~ 0.7 instead of -1.0 ~ 1.0, the blurry images.

__4. Semi-Supervised Learning with Classification Loss__  
* The image quality increases in most cases.
* The model recognizes better the features of each gender such as mustache, beard, color lens and make-up
* Recognition of hair length becomes worse.
  
  
## Analysis
1. Image qualities increase in most cases.
2. Better to learn facial features such as mustache, beard, make-up and skin color.
3. Recognition of hair length become worse in case of male -> female.
4. Model collapse is removed.

## Process
### Discriminator
<p align="center"><img width="100%" src="png/discriminator.png" /></p>

### Generator
<p align="center"><img width="100%" src="png/generator.png" /></p>
  
  
  
## Results
<p align="center"><img width="100%" src="png/new_4.png" /></p>
<p align="center"><img width="100%" src="png/new_5.png" /></p>
<p align="center"><img width="100%" src="png/new_6.png" /></p>
<p align="center"><img width="100%" src="png/new_7.png" /></p>
<p align="center"><img width="100%" src="png/new_1.png" /></p>
<p align="center"><img width="100%" src="png/new_2.png" /></p>
<p align="center"><img width="100%" src="png/new_3.png" /></p>
<p align="center"><img width="100%" src="png/new_8.png" /></p>
