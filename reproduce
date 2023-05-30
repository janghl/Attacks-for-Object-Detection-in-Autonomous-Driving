root=""
import sys
sys.path.append(root)
sys.path.append(root+"/pytorch-caffe")
import neural_renderer as nr
from caffenet import *
from matplotlib import pyplot as plt
from pytorch.renderer import nmr
import torch
import torch.autograd as autograd
import argparse
import cv2
from c2p_segmentation import *
import loss_LiDAR
import numpy as np
import cluster
import os
from xyz2grid import *
import render
from plyfile import *
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from pytorch.yolo_models.utils_yolo import *
from pytorch.yolo_models.darknet import Darknet
r = R.from_euler('zxy', [10,80,4], degrees=True)     #旋转矩阵
lidar_rotation = torch.tensor(r.as_matrix(), dtype=torch.float).cuda()
adv = True
scale = 0.15
name = "adv_chair"
class fusion_detect():
    def __init__(self, args):
        self.args = args
        self.root_path = root+'/data/'
        self.color = 1
        # self.protofile = self.root_path + 'deploy.prototxt'
        # self.weightfile = self.root_path + 'deploy.caffemodel'
        # self.outputs = ['instance_pt', 'category_score', 'confidence_score',
        #            'height_pt', 'heading_pt', 'class_score']        #output for lidar fetection
        # namesfile = root+'/pytorch/yolo_models/data_yolo/coco.names'
        # class_names = load_class_names(namesfile)
        # single_model = Darknet(root+'/pytorch/yolo_models/cfg/yolov3.cfg')
        # single_model.load_weights(root+'/data/yolov3.weights')
        # model = single_model
        # self.model = model.cuda()
        # self.model.eval()
        # self.direction_val, self.dist_val = self.load_const_features('./data/features_1.out')
        

    def preprocess(self,image_path,point_cloud):
        #process image files
        background = cv2.imread(image_path)
        background = cv2.resize(background, (416, 416))
        background = background[:, :, ::-1] / 255.0
        background = background.astype(np.float32)
        background_tensor = torch.from_numpy(background.transpose(2, 0, 1)).cuda()
        final_image = torch.clamp(background_tensor.float(), 0, 1)[None]
        # loading ray_direction and distance for the background pcd
        self.PCL = loadPCL(point_cloud, True)    #按照shape（-1,4）包装起来，四个一组
        x_final = torch.FloatTensor(self.PCL[:, 0]).cuda()  #由x值组成的一维向量
        y_final = torch.FloatTensor(self.PCL[:, 1]).cuda()
        z_final = torch.FloatTensor(self.PCL[:, 2]).cuda()
        self.i_final = torch.FloatTensor(self.PCL[:, 3]).cuda()
        self.ray_direction, self.length = render.get_ray(x_final, y_final, z_final)#雷达上点的模长幅角
        return point_cloud,final_image




    def embed(self,image_path,cali,point_cloud,object): #embed obj inside image and pcl
        #process calibration files
        file1 = open(cali, 'r')
        Lines = file1.readlines()
        for line in Lines:
            if 'R:' in line:
                rotation = line.split('R:')[-1]
            if 'T:' in line:
                translation = line.split('T:')[-1]
        tmp_r = rotation.split(' ')
        tmp_r.pop(0)
        tmp_r[-1] = tmp_r[-1].split('\n')[0]
        # print(tmp_r)
        rota_matrix = []

        for i in range(3):
            tt = []
            for j in range(3):
                tt.append(float(tmp_r[i * 3 + j]))
            rota_matrix.append(tt)
        self.rota_matrix = np.array(rota_matrix)
        tmp_t = translation.split(' ')
        tmp_t.pop(0)
        tmp_t[-1] = tmp_t[-1].split('\n')[0]
        # print(tmp_t)
        trans_matrix = [float(tmp_t[i]) for i in range(3)]
        self.trans_matrix = np.array(trans_matrix)


        #load mesh
        self.load_mesh(object,scale)

        #process image files
        background = cv2.imread(image_path)
        background = cv2.resize(background, (416, 416))
        background = background[:, :, ::-1] / 255.0
        background = background.astype(np.float32)

        # loading ray_direction and distance for the background pcd
        self.PCL = loadPCL(point_cloud, True)    #按照shape（-1,4）包装起来，四个一组
        x_final = torch.FloatTensor(self.PCL[:, 0]).cuda()  #由x值组成的一维向量
        y_final = torch.FloatTensor(self.PCL[:, 1]).cuda()
        z_final = torch.FloatTensor(self.PCL[:, 2]).cuda()
        self.i_final = torch.FloatTensor(self.PCL[:, 3]).cuda()
        self.ray_direction, self.length = render.get_ray(x_final, y_final, z_final)#雷达上点的模长幅角
        

        #final calculate 
        self.object_f = self.object_f.cuda()
        self.i_final = self.i_final.cuda()
        self.object_v = self.object_v.cuda()
        point_cloud = render.render(self.ray_direction, self.length, self.object_v, self.object_f, self.i_final)
        camera_v = self.object_v.clone()
        camera_v = camera_v.permute(1, 0)
        r, t = torch.tensor(self.rota_matrix).cuda().float(), torch.tensor(self.trans_matrix).cuda().float()
        r_c = R.from_euler('zxy', [0, 180, 180], degrees=True)
        camera_v = torch.matmul(r, camera_v)
        camera_rotation = torch.tensor(r_c.as_matrix(), dtype=torch.float).cuda()
        camera_v = torch.matmul(camera_rotation, camera_v)
        camera_v = camera_v.permute(1, 0)
        camera_v += t
        c_v_c = camera_v.cuda()     
        image_tensor = self.renderer.render(c_v_c.unsqueeze(0), self.object_f.unsqueeze(0), self.object_t.unsqueeze(0))[0].cuda()
        # batch, channels, height, old_width = image_tensor.shape
        # new_width = self.width
        # padding = (new_width - old_width) // 2
        # padding2 = new_width - old_width - padding
        # image_tensor = torch.nn.functional.pad(image_tensor, (padding, padding2, 0, 0), mode='constant', value=0)
        mask_tensor = self.renderer.render_silhouettes(c_v_c.unsqueeze(0), self.object_f.unsqueeze(0)).cuda()  #ply的暗色轮廓
        # mask_tensor = torch.nn.functional.pad(mask_tensor, (padding, padding2, 0, 0), mode='constant', value=0)
        background_tensor = torch.from_numpy(background.transpose(2, 0, 1)).cuda() 
        fg_mask_tensor = torch.zeros(background_tensor.size())      #空的tensor
        new_mask_tensor = mask_tensor.repeat(3, 1, 1) #shape从[a,b]变成[3,a,b]
        # fg_mask_tensor[:, 0: self.height,
        # 0: self.width] = new_mask_tensor
        fg_mask_tensor[:, 0: self.image_size,
        0: self.image_size] = new_mask_tensor
        fg_mask_tensor = fg_mask_tensor.byte().cuda()
        new_mask_tensor = new_mask_tensor.byte().cuda()
        # if(self.color==1):
        #     image_tensor=image_tensor.permute(0,2,3,1)
        #     # plt.imshow(image_tensor.cpu()[0])
        #     # plt.show()
        #     # red=torch.tensor([0.396,0.262,0.129]).cuda()
        #     red=torch.tensor([1,0.647,0]).cuda()
        #     for i in range (self.height):
        #         for j in range (self.width):
        #             if(torch.sum(image_tensor[0][i][j])>0.5):
        #                 image_tensor[0][i][j]=red
        #     image_tensor=image_tensor.permute(0,3,1,2)
        background_tensor.masked_scatter_(fg_mask_tensor, image_tensor.masked_select(new_mask_tensor))
        final_image = torch.clamp(background_tensor.float(), 0, 1)[None]
        return point_cloud,final_image
    
    
    def load_mesh(self, path, r, x_of=7, y_of=0):#参数包括x,y,z,vertex_indices,vertex_indices是一个数组，3个点（int）构成一个面
        z_of = -1.73 + r / 2.
        plydata = PlyData.read(path)
        if(adv==True):
            x = torch.FloatTensor(plydata['vertex']['x']) * r 
            z = torch.FloatTensor(plydata['vertex']['y']) * r
            y = torch.FloatTensor(plydata['vertex']['z']) * r
        else:
            x = torch.FloatTensor(plydata['vertex']['x']) * r
            y = torch.FloatTensor(plydata['vertex']['y']) * r
            z = torch.FloatTensor(plydata['vertex']['z']) * r
        self.object_v = torch.stack([x, y, z], dim=1).cuda()

        self.object_f = plydata['face'].data['vertex_indices']
        for i in range(self.object_f.size):
            self.object_f[i] = self.object_f[i][:3]
        self.object_f = torch.tensor(np.vstack(self.object_f)).cuda()
        #物体平移和旋转
        rotation = lidar_rotation.cuda()
        self.object_v = self.object_v.cuda()
        self.object_v = self.object_v.permute(1, 0)  #矩阵转置，变成列向量
        self.object_v = torch.matmul(rotation, self.object_v)
        self.object_v = self.object_v.permute(1, 0)
        self.object_v[:, 0] += x_of
        self.object_v[:, 1] += y_of
        self.object_v[:, 2] += z_of

        self.object_ori = self.object_v.clone()    #点的拷贝
        self.object_t = torch.tensor(self.object_v.new_ones(self.object_f.shape[0], 1, 1, 1, 3)).cuda()
        # color red
        self.object_t[:, :, :, :, 1] = 0.3
        self.object_t[:, :, :, :, 2] = 0.3
        self.object_t[:, :, :, :, 0] = 0.3
        self.mean_gt = self.object_ori.mean(0).data.cpu().clone().numpy()

    def load_const_features(self, fname):

        print("Loading direction, dist")
        features_filename = fname

        features = np.loadtxt(features_filename)
        features = np.swapaxes(features, 0, 1)
        features = np.reshape(features, (1, 512, 512, 8))
        direction = np.reshape(features[:, :, :, 3], (1, 512, 512, 1))
        dist = np.reshape(features[:, :, :, 6], (1, 512, 512, 1))
        return torch.tensor(direction).cuda().float(), torch.tensor(dist).cuda().float()


    #outputPytorch[1] denotes score for classification, outputPytorch[2] denotes score for existance possibility
    #return sum of existance confidence
    def lidar_result(self,vertex,outputPytorch):
        x_var = vertex[:, 0]     #mean of x value for cone
        y_var = vertex[:, 1]
        fx = torch.floor(x_var * 512.0 / 120 + 512.0 / 2).long()
        fy = torch.floor(y_var * 512.0 / 120 + 512.0 / 2).long()
        #mask=1 where cone lies
        mask = torch.zeros((512, 512)).cuda().index_put((fx, fy), torch.ones(fx.shape).cuda())
        #1 if point>threshold(0.5)    
        mask1 = torch.where(torch.mul(mask, outputPytorch[1]) >= 0.5, torch.ones_like(mask), torch.zeros_like(mask))
        return torch.sum(torch.mul(mask1, outputPytorch[2]))



    def lidar(self, point_cloud):
        net = CaffeNet(self.protofile, phase='TEST')
        # torch.cuda.set_device(0)
        net.cuda()
        net.load_weights(self.weightfile)
        net.set_train_outputs(outputs)
        net.set_eval_outputs(outputs)
        net.eval()              #执行字符串中的代码
        for p in net.parameters():
            p.requires_grad = False
        grid = xyzi2grid_v2(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], point_cloud[:, 3])
        featureM = gridi2feature_v2(grid, self.direction_val, self.dist_val)
        outputPytorch = net(featureM)
        result = self.lidar_result(self.object_v,outputPytorch)
        print(f"lidar_result={result}")
        return result



    def camera(self,final_image):
        confidence=0
        num_class = 80
        threshold = 0.5
        batch_size = 1
        final, outputs = self.model(final_image)
        for index, out in enumerate(outputs):               #yolo gives out 3 tensors
                num_anchor = out.shape[1] // (num_class + 5)       
                out = out.view(batch_size * num_anchor, num_class + 5, out.shape[2], out.shape[3])
                cfs = torch.nn.functional.sigmoid(out[:, 4]).cuda()        
                mask = (cfs >= threshold).type(torch.FloatTensor).cuda()   
                total = torch.sum(mask * ((cfs - 0) ** 2 - (1 - cfs) ** 2))
                if confidence is None:
                    confidence = total
                else:
                    confidence += total
        confidence = 12 * (F.relu(torch.clamp(confidence, min=0) - 0.01) / 5.0)
        print(f"confidence={confidence}")
        return confidence

    def init_render(self, image_size = 416):        #渲染器
        self.image_size = image_size
        self.renderer = nr.Renderer(image_size=image_size, camera_mode='look_at',
                                    anti_aliasing=False, light_direction=(0, 0, 0))
        # self.renderer = nr.Renderer(image_size=self.height, camera_mode='look_at',
        #                             anti_aliasing=False, light_direction=(0, 0, 0))
        exr = cv2.imread('./data/dog.exr', cv2.IMREAD_UNCHANGED)             #训练渲染器
        self.renderer.light_direction = [1, 3, 1]

        ld, lc, ac = nmr.lighting_from_envmap(exr)              #全局光照
        self.renderer.light_direction = ld
        self.renderer.light_color = lc
        self.renderer.ambient_color = ac
    

    def calculate(self):
        success_count = 0 
        iteration = 100
        for i in range (iteration):
            j = 2*i
            k = 2*i+1
            if i<10:
                i='00%d'%i
            elif i<100:
                i='0%d'%i
            else :
                i=str(i)
            # if j<10:
            #     j='00%d'%j
            # elif j<100:
            #     j='0%d'%j
            # else :
            #     j=str(j)
            # if k<10:
            #     k='00%d'%k
            # elif k<100:
            #     k='0%d'%k
            # else :
            #     k=str(k)
            image_path = root+'/reproduce/image/000'+i+'.png'
            # cali = open(root+'/reproduce/calib/000'+i+'.txt','r')
            cali =root+'/data/cali.txt'
            point_cloud_path = root+'/reproduce/lidar/000'+i+'.bin'
            self.image = cv2.imread(image_path)
            # self.height, self.width, self.channels = self.image.shape
            self.init_render()
            adv_point_cloud,adv_image = self.embed(image_path,cali,point_cloud_path,args.object)
            adv_image = adv_image.squeeze().cpu().detach().numpy()
            adv_image = adv_image.transpose(1, 2, 0)[:,:,::-1]
            # 将张量的值从 [0, 1] 的范围映射到 [0, 255] 的范围
            adv_image = adv_image * 255.0
            # 将张量的数据类型转换为 uint8
            adv_image = adv_image.astype(np.uint8)
            print(f"0000000{i}.png'")
            # print(f"0000000{k}.png'")
            # 将 RGB 图像输出到文件
            cv2.imwrite(root+'/reproduce/out_image/'+name+'/0000000'+i+'.png', adv_image)
            # cv2.imwrite(root+'/reproduce/out_image/'+name+'/0000000'+k+'.png', adv_image)

            # 将 PyTorch 张量转换为 NumPy 数组
            adv_point_cloud_np = adv_point_cloud.cpu().detach().numpy()

            print(f"0000000{i}.bin'")
            # print(f"0000000{k}.bin'")
            # 将 NumPy 数组写入二进制文件
            np.ndarray.tofile(adv_point_cloud_np, root+'/reproduce/out_lidar/'+name+'/0000000'+i+'.bin')
            # np.ndarray.tofile(adv_point_cloud_np, root+'/reproduce/out_lidar/'+name+'/0000000'+k+'.bin')
        #     benign_point_cloud,benign_image = self.preprocess(image_path,point_cloud_path)
        #     camera_confidence_difference = self.camera(benign_image) - self.camera(adv_image)
        #     print(torch.sum(adv_image-benign_image))
        #     lidar_confidence_difference = self.lidar(benign_point_cloud) - self.lidar(adv_point_cloud)
        #     print(f"it={i}------image_cfs_diff={camera_confidence_difference}---lidar_cfs_diff={lidar_confidence_difference}")
        #     if abs(lidar_confidence_difference < 0.000003) and abs(camera_confidence_difference < 0.000003):
        #         success_count = success_count + 1
        #         print(f"success_count={success_count}")

        # print(f"success_rate = {100 * success_count / iteration}%")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj', '--obj', dest='object', default=root+"/object/"+name+".ply")
    # parser.add_argument('-obj', '--obj', dest='object', default=root+"/object/benign_cone.ply")
    parser.add_argument('-o', '--opt', dest='opt', default="pgd")
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.2)
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
    args = parser.parse_args()
    obj = fusion_detect(args)
    obj.init_render()
    obj.calculate()
    
