# Single Image Crowd Counting via Multi Column Convolutional Neural Network（基于多列卷积神经网络的单一图像人流计数）

This is an unofficial implementation of CVPR 2016 paper ["Single Image Crowd Counting via Multi Column Convolutional Neural Network"](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)

# environment（环境）
  SYS：Windows  
  python：python3.6 (python3 is ok)  
  matlab  

# Installation（必要的安装）
1. Install pytorch
2. Install OpenCV of python
3. Install pandas
4. Install h5py
5. Clone this repository  
  We'll call the directory that you cloned crowdcount-mcnn `ROOT`  
  把该仓库中的代码下载到本地，保存在"e:/root"文件夹下

# Data Setup（数据准备）
1. Download ShanghaiTech Dataset from（下载数据，下面二者选其一）   
   Dropbox:   https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0
   
   Baidu Disk: http://pan.baidu.com/s/1nuAYslz
2. Create Directory（新建文件夹）  
  Create a new director "e:/root/data/original/shanghaitech/"
3. Save "part_A_final" under e:/root/data/original/shanghaitech/
4. Save "part_B_final" under e:/root/data/original/shanghaitech/
5. cd "e:/root/crowdcount-mcnn-master/data_preparation"  
   run create_gt_test_set_shtech.m in matlab to create ground truth files for test data
6. cd "e:/root/crowdcount-mcnn-master/data_preparation"  
   run create_training_set_shtech.m in matlab to create training and validataion set along with ground truth files

# Test
1. Follow steps 1,2,3,4 and 5 from Data Setup
2. Download pre-trained model files（下载预训练过的模型）:  

   Shanghai Tech A  
	 Dropbox:  
	     https://www.dropbox.com/s/8bxwvr4cj4bh5d8/mcnn_shtechA_660.h5?dl=0  
	 Baidu Disk:  
	     链接：https://pan.baidu.com/s/17ETn4EUNIIOXwzkHCHTZxQ  
             提取码：55t3 
   
   Shanghai Tech B  
	 Dropbox:
	     https://www.dropbox.com/s/kqqkl0exfshsw8v/mcnn_shtechB_110.h5?dl=0  
	 Baidu Disk:  
	     链接：https://pan.baidu.com/s/1GIc3k4GH0ckey7O-6KG5iw  
	     提取码：rblb 
   
   Save the model files under e:/root/final_models
   
3. Run test.py

	a. Set save_output = True to save output density maps
	
	b. Errors are saved in  output directory

4. Run count_my_image.py  
    a. Prepare your image for crowd counting  
    	Create director "e:/my_data", which is called "DATA_ROOT". Create new directors under "DATA_ROOT" and name your new directors any name you like. Put your images shoot by phone or download from Internet in your new directors.  
        新建文件夹“e:/my_data”，在该文件夹下新建你自己的文件夹，你可以给这些文件夹起任意你喜欢的名字。然后把你要测试的图片放到这些文件夹下。  
    b. Run count_my_imang.py  
        This program will show you the estimate count of every single image.
# Training
1. Follow steps 1,2,3,4 and 6 from Data Setup
2. Run train.py


# Training with TensorBoard
With the aid of [Crayon](https://github.com/torrvision/crayon),
we can access the visualisation power of TensorBoard for any 
deep learning framework.

To use the TensorBoard, install Crayon (https://github.com/torrvision/crayon)
and set `use_tensorboard = True` in `ROOT/train.py`.

# Other notes
1. During training, the best model is chosen using error on the validation set. (It is not clear how the authors in the original implementation choose the best model).
2. 10% of the training set is set asised for validation. The validation set is chosen randomly.
3. The ground truth density maps are obtained using simple gaussian maps unlike the original method described in the paper.
4. Following are the results on  Shanghai Tech A and B dataset:
		
                |     |  MAE  |   MSE  |
                ------------------------
                | A   |  110  |   169  |
                ------------------------
                | B   |   25  |    44  |
		
5. Also, please take a look at our new work on crowd counting using cascaded cnn and high-level prior (https://github.com/svishwa/crowdcount-cascaded-mtl),  which has improved results as compared to this work. 
               

