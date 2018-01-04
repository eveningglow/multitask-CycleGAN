# semi-supervised-CycleGAN
Implementation of [CycleGAN : Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) using __pytorch__. 
Futhermore, this implementation is using __multitask learning with semi-supervised leaning__ which means utilize labels of data. This model converts male to female or female to male.
Following image shows improvements such as facial features(make-up, mustache, beard, etc) and image qualities.

<p align="center"><img width="100%" src="png/1.png" /></p>

## What are differences with original CycleGAN?
1. Batch size 1 -> 16 (Critical)  
2. Instance Normalization -> Batch Normalization (Critical)  
3. Model architecture (Prevent from training faulure)  
4. Smooth labeling (Prevent from training faulure)  
5. Multitask learning with classification loss (Critical)  

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
