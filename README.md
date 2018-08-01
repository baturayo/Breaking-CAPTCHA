# generate_captcha.py
Run this file first to create sample CAPTCHA images.
Update line 58 to set the total number of train images.

# dataset.py
You do not need to directly run this script. It is called from Breaking Captcha.ipynb script to call train image attributes,
,to split data into train and validation sets and, to create training and validation data batches.

# Breaking Captcha.ipynb
This Jupyter notebook runs Convolutional Neural Network model in TensorFlow to break CAPTCHA images.

# OpenSans-light.ttf
Font file called by generate_captcha.py

# Breaking Captcha Report.pdf
Detailed report about breaking CAPTCHA algorithm
