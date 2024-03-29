# Kernel Methods 

Thomas LI

Useful links :

- https://medium.com/@livajorge7/the-science-of-color-understanding-color-spaces-in-image-processing-d0e238872a0c
- https://www.kaggle.com/code/prashant111/svm-classifier-tutorial
- https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9132851&casa_token=wLX3qWTASKMAAAAA:c1pppZLjMh-gwFezSYcBr5Y6dDw-2CagtAtyGU03Ra8delEiXVlFXl7WUsXbDgysPDyu1EkN
- https://www.dbs.ifi.lmu.de/~yu_k/cvpr11_0694.pdf
- 


Ideas :
- train on validation + train
- ensembling
- create new features (sift, surf, color histogram) 24 -> 54
- try Laplace kernel


Tried :
- preprocessing minmax scaling : better without minmax scaling + 0.01
- do data augmentation (flipping image, noise, hue and saturations augmentations, translations, cropping) : 55 -> 59
- hog features 24 -> 54