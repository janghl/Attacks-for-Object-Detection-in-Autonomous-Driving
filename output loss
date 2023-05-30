root=""
import sys
sys.path.append("")
from attack import *
import os
import torchgen
import argparse
import loss_LiDAR
import numpy as np
import os
from xyz2grid import *
import render
from plyfile import *
import torch.nn.functional as F


def render_image():
    camera_v = obj.object_v.clone()
    camera_v = camera_v.permute(1, 0)
    r, t = torch.tensor(obj.rota_matrix).cuda().float(), torch.tensor(obj.trans_matrix).cuda().float()
    r_c = R.from_euler('zxy', [0, 180, 180], degrees=True)
    camera_v = torch.matmul(r, camera_v)
    camera_v = camera_v.permute(1, 0)
    camera_v = camera_v.permute(1, 0)
    camera_rotation = torch.tensor(r_c.as_matrix(), dtype=torch.float).cuda()
    camera_v = torch.matmul(camera_rotation, camera_v)
    camera_v = camera_v.permute(1, 0)
    camera_v += t
    c_v_c = camera_v.cuda()            
    image_tensor = obj.renderer.render(c_v_c.unsqueeze(0), obj.object_f.unsqueeze(0), obj.object_t.unsqueeze(0))[0].cuda()
    mask_tensor = obj.renderer.render_silhouettes(c_v_c.unsqueeze(0), obj.object_f.unsqueeze(0)).cuda()  #ply的暗色轮廓
    background_tensor = torch.from_numpy(obj.background.transpose(2, 0, 1)).cuda() 
    fg_mask_tensor = torch.zeros(background_tensor.size())      #空的tensor
    new_mask_tensor = mask_tensor.repeat(3, 1, 1) #shape从[a,b]变成[3,a,b]
    fg_mask_tensor[:, 0: obj.image_size,0: obj.image_size] = new_mask_tensor
    fg_mask_tensor = fg_mask_tensor.byte().cuda()
    new_mask_tensor = new_mask_tensor.byte().cuda()
    background_tensor.masked_scatter_(fg_mask_tensor, image_tensor.masked_select(new_mask_tensor))
    final_image = torch.clamp(background_tensor.float(), 0, 1)[None]
    return final_image



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='please specify the ply file for your object')
    # parser.add_argument('-obj', '--obj', dest='object', default=root+"/object/chair.ply")
    parser.add_argument('-obj', '--obj', dest='object', default=root+"/object/adv_chair.ply")
    parser.add_argument('-lidar', '--lidar', dest='lidar', default=root+"/data/lidar.bin")
    parser.add_argument('-cam', '--cam', dest='cam', default=root+"/data/cam.png")
    parser.add_argument('-cali', '--cali', dest='cali',default=root+"/data/cali.txt")
    parser.add_argument('-o', '--opt', dest='opt', default="pgd")
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.2)
    parser.add_argument('-it', '--iteration', dest='iteration', type=int, default=1000)
    args = parser.parse_args()
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
    obj = attack_msf(args)
    obj.load_model_()      
    obj.load_LiDAR_model()
    obj.read_cali(args.cali)
    obj.load_mesh(args.object, 0.15)
    obj.load_bg(args.cam)
    obj.init_render()
    obj.load_pc_mesh(args.lidar)
    lr = 0.005
    best_it = 1e10
    num_class = 80
    batch_size = 1
    camloss=0
    obj.object_f = obj.object_f.cuda()
    obj.i_final = obj.i_final.cuda()
    obj.object_v = obj.object_v.cuda()
    adv_total_loss = None
    point_cloud = render.render(obj.ray_direction, obj.length, obj.object_v, obj.object_f, obj.i_final)
    grid = xyzi2grid_v2(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], point_cloud[:, 3])
    featureM = gridi2feature_v2(grid, obj.direction_val, obj.dist_val)
    outputPytorch = obj.LiDAR_model(featureM)
    lidarloss, loss_object, loss_distance, loss_center, loss_z = loss_LiDAR.lossRenderAttack(outputPytorch, obj.object_v, obj.object_ori, obj.object_f, 0.05)
    final_image=render_image()
    final, outputs = obj.model(final_image) #darknet框架  
    for i in range(3):             
        num_anchor = outputs[i].shape[1] // (num_class + 5)     
        outputs[i] = outputs[i].view(batch_size * num_anchor, num_class + 5, outputs[i].shape[2], outputs[i].shape[3])
        cfs=torch.nn.functional.sigmoid(outputs[i][:, 4])
        mask = (cfs >= 0.5).type(torch.FloatTensor).cuda()
        camloss+=torch.sum(mask*(2*cfs-1))
    print(f"camera_loss={camloss},lidar_loss={lidarloss},total_loss={camloss+lidarloss}")
