# python3.6
# Author: Scc_hy
# Create_date: 2020-04-21

import numpy
class pic_scale():
    """
    图片缩放：包含两个方法
    INTER_NEAREST	最近邻插值
    INTER_LINEAR	双线性插值
    例子：
    pic_sc = pic_scale()
    pic_sc.pic_resize(img_dt, resize, fx=None, fy=None, interpolation = pic_sc.INTER_LINEAR)
    """
    def INTER_NEAREST(self, pic_dt, resize, x_scale=None, y_scale=None):
        """
        最近邻插值（图片 m * n * 图层）  
        param pic_dt: 为一个图片的一个图层的数据 len(pic_dt) == 2  
        param resize: set (长, 宽)  
        param x_scale: float 长度缩放大小  
        param y_scale: float 宽带缩放大小  
        """ 
        m, n = pic_dt.shape
        # 获取新的图像的大小
        if resize is None:
            n_new, m_new  =  np.round(x_scale * n).astype(int), np.round(y_scale * m).astype(int)
        else:
            n_new, m_new  = resize

        fx, fy = n / n_new, m / m_new # src_with/dst_with, Src_height/dst_heaight
        # 初始化X, Y的位置点
        idx_x_orign = np.array(list(range(n_new)) * m_new).reshape(m_new, n_new)
        idx_y_orign = np.repeat(list(range(m_new)), n_new).reshape(m_new, n_new)
        # 需要的近邻的位置
        x_indx = np.round(idx_x_orign * fx).astype(int)
        y_indx = np.round(idx_y_orign * fy).astype(int)
        return   pic_dt[y_indx, x_indx]

    def insert_linear_pos(self, img_dt, resize, x_scale=None, y_scale=None):
        """
        找位置，及位置周围四个像素点
        """
        m_, n_ = img_dt.shape
        # 获取新的图像的大小
        if resize is None:
            n_new, m_new  =  np.round(x_scale * n_).astype(int), np.round(y_scale * m_).astype(int)
        else:
            n_new, m_new  = resize

        n_scale, m_scale = n_ / n_new, m_ / m_new # src_with/dst_with, Src_height/dst_heaight
        # 一、获取位置对应的四个点
        # 1-1- 初始化位置
        m_indxs = np.repeat(np.arange(m_new), n_new).reshape(m_new, n_new)
        n_indxs = np.array(list(range(n_new))*m_new).reshape(m_new, n_new)
        # 1-2- 初始化位置
        m_indxs_c = (m_indxs + 0.5 ) * m_scale - 0.5
        n_indxs_c = (n_indxs + 0.5 ) * n_scale - 0.5
        ### 将小于零的数处理成0 
        m_indxs_c[np.where(m_indxs_c < 0)] = 0.0
        n_indxs_c[np.where(n_indxs_c < 0)] = 0.0

        # 1-3 获取正方形顶点坐标
        m_indxs_c_down = m_indxs_c.astype(int)
        n_indxs_c_down = n_indxs_c.astype(int)
        m_indxs_c_up = m_indxs_c_down + 1
        n_indxs_c_up = n_indxs_c_down + 1
        ### 溢出部分修正
        m_max = m_ - 1
        n_max = n_ - 1
        m_indxs_c_up[np.where(m_indxs_c_up > m_max)] = m_max
        n_indxs_c_up[np.where(n_indxs_c_up > n_max)] = n_max

        # 1-4 获取正方形四个顶点的位置
        pos_0_0 = img_dt[m_indxs_c_down, n_indxs_c_down].astype(int)
        pos_0_1 = img_dt[m_indxs_c_up, n_indxs_c_down].astype(int)
        pos_1_1 = img_dt[m_indxs_c_up, n_indxs_c_up].astype(int)
        pos_1_0 = img_dt[m_indxs_c_down, n_indxs_c_up].astype(int)
        # 1-5 获取浮点位置
        m, n = np.modf(m_indxs_c)[0], np.modf(n_indxs_c)[0]
        return pos_0_0, pos_0_1, pos_1_1, pos_1_0, m, n

    def INTER_LINEAR(self, img_dt, resize, fx=None, fy=None):
        """
        双线性插值算法
        """
        pos_0_0, pos_0_1, pos_1_1, pos_1_0, m, n = self.insert_linear_pos(img_dt=img_dt, resize=resize, x_scale=fx, y_scale=fy)
        a = (pos_1_0 - pos_0_0)
        b = (pos_0_1 - pos_0_0)
        c = pos_1_1 + pos_0_0 - pos_1_0 - pos_0_1
        return np.round(a * n + b * m + c * n * m + pos_0_0).astype(int)

    def pic_resize(self, img_dt, resize, fx=None, fy=None, interpolation = None):
        """
        # 三个通道分开处理再合并
        param interpolation 插值方法:self.INTER_NEAREST	最近邻插值  self.INTER_LINEAR	双线性插值
        """
        # 三个通道分开处理再合并
        if interpolation is None:
           interpolation  = self.INTER_LINEAR
        if len(img_dt.shape) == 3:
            out_img0 = interpolation(img_dt[:,:,0],  resize,  fx,  fy)
            out_img1 = interpolation(img_dt[:,:,1],  resize,  fx,  fy)
            out_img2 = interpolation(img_dt[:,:,2],  resize,  fx,  fy)
            out_img_all = np.c_[out_img0[:,:,np.newaxis], out_img1[:,:,np.newaxis], out_img2[:,:,np.newaxis]]
        else:
            out_img_all = interpolation(img_dt,  resize,  fx,  fy)
        return out_img_all.astype(np.uint8)


pic_sc = pic_scale()
pic_ = pic_sc.pic_resize(img, resize=None, fx=1.5, fy=1.5, interpolation = pic_sc.INTER_LINEAR)

out_img_all = linear_insert(img, resize=None, fx=1.5, fy=1.5)
resized1_5_img_linear = cv2.resize(img, dsize=None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

cv2.imshow('Origin', img)
cv2.imshow('linear_insert*1.5', out_img_all)
cv2.imshow('linear_insert*1.5', pic_)
cv2.imshow('Origin-[INTER_LINEAR] * 1.5', resized1_5_img_linear)
cv2.waitKey(0)
