# Kernel Methods 

Thomas LI

All the personal code are in the src directory.

The data folder is empty, you need to put the kaggle dataset inside this folder to run the code.

The submission used for the kaggle challenge is : submissions/0313_1031_submission_4_trainval.csv

Tried :
- ensembling
- preprocessing minmax scaling : better without minmax scaling + 0.01
- do data augmentation (flipping image, noise, hue and saturations augmentations, translations, cropping) : 55 -> 59
- hog features 24 -> 54
