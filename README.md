I will be sharing a script using Keras for training a Convolutional Neural Network (CNN) with transfer learning for melanoma detection. You can find the code in <a href="https://github.com/abderhasan/cnn_melanoma_classification_transfer_learning_keras"><strong>this GitHub repository</strong></a>.

Before proceeding, make sure that you structure the data as follows (the numbers represent the number of images in each file):

![alt text](https://github.com/abderhasan/cnn_melanoma_classification_from_scratch_keras/blob/master/directory-structure.png)

You can download the data from,<strong> <a href="https://drive.google.com/drive/folders/126UgFt_xqnHpeV1Pr_qLDQLwzBbi4rrY?usp=sharing">here</a></strong>. I used two classes as you can see from the figure above (nevus and melanoma). For training, I kept 374 images in each class to keep the data balanced.

To run the code:

`$ python cnn_transfer_learning.py`

The results will not be optimal, as the purpose is to show how one can train a CNN from scratch.

<strong>What variables to edit in the code?</strong>

You need to edit the following variables to point to your data:

<em>train_directory</em> (path to your training directory)

<em>validation_directory</em> (path to your training directory)

<em>test_directory</em> (path to your testing directory)

<strong>What should you expect (outputs)?</strong>

<em>Training and validation accuracy</em>

<img class="aligncenter size-full wp-image-873" src="https://abder.io/wp-content/uploads/2019/01/accuracy-2.png" alt="" width="640" height="480" />

&nbsp;

<em>Training and validation loss</em>

<img class="aligncenter size-full wp-image-874" src="https://abder.io/wp-content/uploads/2019/01/loss-2.png" alt="" width="640" height="480" />

<em>ROC curve</em>

<img class="aligncenter size-full wp-image-875" src="https://abder.io/wp-content/uploads/2019/01/ROC-2.png" alt="" width="640" height="480" />

In addition to some other values (i.e. accuracy, confusion matrix) that will be displayed on the console.

If you would like to train a CNN from scratch, you can see <strong><a href="https://abder.io/training-a-cnn-from-scratch-for-melanoma-detection-using-keras/">this post</a></strong>. If you like to train a CNN from scratch with data augmentation, you can see <strong><a href="https://abder.io/training-a-cnn-from-scratch-with-data-augmentation-for-melanoma-detection-using-keras/">this post</a></strong>.
