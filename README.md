# 针对无人驾驶场景下物体识别的攻击
# Attacks-for-Object-Detection-in-Autonomous-Driving

原论文及代码地址：https://github.com/ASGuard-UCI/MSF-ADV

代码功能：

    1.将物体的三维网格文件嵌入到图像（光线投影算法 Ray Casting）或激光雷达点云（神经三维网格渲染器 Neural 3D Mesh Renderer）中。

    2.调用YOLOv3算法和Cnn_seg算法，检测嵌入了物体的图像和激光雷达点云检测到物体的概率。
