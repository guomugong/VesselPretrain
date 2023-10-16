# coding=utf-8
import bezier, mmcv
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import math
import os

from vlkit.geometry import random_perspective_matrix
import argparse, cv2, random

from PIL import Image, ImageFile
import copy
ImageFile.LOAD_TRUNCATED_IMAGES = True


def wrap_points(points):
    assert isinstance(points, np.ndarray)
    n = points.shape[0]
    augmented_points = np.concatenate((points, np.ones((n, 1))), axis=1).astype(points.dtype)
    points = (augmented_points.T).T
    points = points / points[:,-1].reshape(-1, 1)
    return points[:, :2]

def sample_edge(low, high):
    """
    sample points on edges of a unit square
    """
    offset = min(low, high)
    low, high = map(lambda x: x - offset, [low, high])
    t = np.random.uniform(low, high) + offset

    if t >= 4:
        t = t % 4
    if t < 0:
        t = t + 4

    if t <= 1:
        x, y = t, 0
    elif 1 < t <= 2:
        x, y = 1, t - 1
    elif 2 < t <= 3:
        x, y = 3 - t, 1
    else:
        x, y = 0, 4 - t
    return np.array([x, y]), t

def control_point(head, tail, t=0.5, s=0):
    head = np.array(head)
    tail = np.array(tail)
    l = np.sqrt(((head - tail) ** 2).sum())
    assert head.size == 2 and tail.size == 2
    assert l >= 0
    c = head * t + (1 - t) * tail
    x, y = head - tail
    v = np.array([-y, x])
    v /= max(np.sqrt((v ** 2).sum()), 1e-6)
    return c + s * l * v

def get_bezier(p0, p1, t=0.5, s=1):
    assert -1 < s < 1, 's=%f'%s
    c = control_point(p0, p1, t, s)
    nodes = np.vstack((p0, c, p1)).T
    return bezier.Curve(nodes, degree=2)

def pixel_equal(image1, image2, x, y, b):
    if (math.isclose(image1[x, y, 0],b[2]*255,rel_tol=1e-1)) and (math.isclose(image1[x, y, 1], b[1]*255, rel_tol=1e-1)) and (math.isclose(image1[x, y, 2],b[0]*255,rel_tol=1e-1)):
        return True
    else:
        return False

def compare(image1, image2, b):
    Xlenth = image1.shape[1] 
    Ylenth = image1.shape[0]
    label = np.zeros([Xlenth, Ylenth], dtype=int)

    for i in range(Xlenth):
        for j in range(Ylenth):
            if pixel_equal(image1, image2, i, j, b):
                label[i, j] = 255
            else:
                label[i, j] = 0
    return label

def generate_parameters_up(number):
    nodes = []
    for i in range(number):
        startx = random.uniform(0, 1/4)
        starty = random.uniform(1.3/4, 2/4)
        # startx = np.random.uniform(low=0, high=0, size=(len(nodes1))).tolist()
        # starty = np.random.uniform(low=1, high=1, size=(len(nodes1))).tolist()
        offx = random.uniform(0.6, 0.8)
        offy = random.uniform(-0.1, 0)
        endx = startx + offx
        endy = starty + offy
        c1 = control_point([startx, starty], [endx, endy], s=np.random.uniform(0.5, 0.7))
        tmp_node = np.asfortranarray([
            [startx, starty],  # start point
            c1,
            [endx, endy],  # end point
        ])
        nodes.append(tmp_node)

    return nodes

def generate_parameters_down(number):
    nodes = []
    for i in range(number):
        startx = random.uniform(0, 1/4)
        starty = random.uniform(2/4, 3/4)
        offx = random.uniform(0.6, 0.8)
        offy = random.uniform(-0.1, 0)
        endx = startx + offx
        endy = starty + offy
        c1 = control_point([startx, starty], [endx, endy], s=np.random.uniform(-0.7, -0.5))
        tmp_node = np.asfortranarray([
            [startx, starty],  # start point
            c1,
            [endx, endy],  # end point
        ])
        nodes.append(tmp_node)

    return nodes



# 给一根主线生成随机弯曲的直线，但目前直线都是二折的
def generate_sencond_parameters(c1):

    n = random.randint(5, 8)
    second_line = []
    for i in range(n):
        location = c1.evaluate(random.uniform(0, 1))
        s1 = [location[0][0]/512, location[1][0]/512]
        e1 = [random.uniform(location[0][0]/512, location[0][0]/512+np.random.uniform(0.2, 0.4)),
              random.uniform(location[1][0]/512, location[1][0]/512+np.random.uniform(0.2, 0.4))]
        control_1 = control_point([s1[0], s1[1]], [e1[0], e1[1]], s=np.random.uniform(0.2, 0.3))
        nodes = np.asfortranarray([
            [s1[0], s1[1]],
            [control_1[0], control_1[1]],
            [e1[0], e1[1]],
        ])
        second_line.append(nodes)
    return second_line


def process_img(filename, out_filename, label_outfile, Ecolor, lwm, lws, degree): #out_filename = N = ID数目
    EPS = 1e-2

    # start/end points of main creases
    # 生成一级血管(up)
    # 生成一级血管的数量作为参数
    nodes_up = generate_parameters_up(1) # 算法1 m
    start_up = np.random.uniform(low=0, high=0, size=(len(nodes_up))).tolist()
    end_up = np.random.uniform(low=1, high=1, size=(len(nodes_up))).tolist()

    # 生成一级血管(up)
    nodes_down = generate_parameters_down(1) # 算法1 m
    start_down = np.random.uniform(low=0, high=0, size=(len(nodes_down))).tolist()
    end_down = np.random.uniform(low=1, high=1, size=(len(nodes_down))).tolist()


    fig = plt.figure(frameon=False)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    fig.set_size_inches((512+ EPS) / dpi, (512 + EPS) / dpi)
    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.axis('off')

    bg_im = np.array(Image.open(filename).resize(size=(512,)*2))

    bg_im2 = copy.deepcopy(bg_im) # array
    bg_im = Image.fromarray(bg_im)
    ax.imshow(bg_im)
    bg_im2 = bg_im2.astype('uint8')
    bg_im2 = mmcv.rgb2bgr(bg_im2)


    # main creases
    # 一级血管-up
    curves_up = [bezier.Curve(n.T * 512, degree=degree) for n in nodes_up]
    # 一级血管-down
    curves_down = [bezier.Curve(n.T * 512, degree=degree) for n in nodes_down]


    ## 生成二级血管
    # 生成一级血管(up)的二级血管
    # up0的二级血管
    nodes_s_up0 = generate_sencond_parameters(curves_up[0])
    curves_s_up0 = [bezier.Curve(n.T * 512, degree=degree) for n in nodes_s_up0]
    start_s_up0 = np.random.uniform(low=0, high=0, size=(len(nodes_s_up0))).tolist()
    end_s_up0 = np.random.uniform(low=1, high=1, size=(len(nodes_s_up0))).tolist()
    # 生成一级血管(down)的二级血管
    # down0的二级血管
    nodes_s_down0 = generate_sencond_parameters(curves_down[0])
    curves_s_down0 = [bezier.Curve(n.T * 512, degree=degree) for n in nodes_s_down0]
    start_s_down0 = np.random.uniform(low=0, high=0, size=(len(nodes_s_down0))).tolist()
    end_s_down0 = np.random.uniform(low=1, high=1, size=(len(nodes_s_down0))).tolist()


    # 合成一级血管(up)
    points_up = [c.evaluate_multi(np.linspace(s, e, 50)).T for c, s, e in zip(curves_up, start_up, end_up)]
    points_up = [wrap_points(p) for p in points_up]
    paths_up = [Path(p) for p in points_up]
    patches_up = [patches.PathPatch(p, edgecolor=Ecolor, facecolor='none', lw=lwm) for p in paths_up]
    #patches1 = [patches.PathPatch(p, edgecolor=Ecolor, facecolor='none', lw=np.random.uniform(1,4)) for p in paths1]
    for p in patches_up:
        ax.add_patch(p) #画线的操作 ax.image.show 写背景图像
    # 合成一级血管(down)
    points_down = [c.evaluate_multi(np.linspace(s, e, 50)).T for c, s, e in zip(curves_down, start_down, end_down)]
    points_down = [wrap_points(p) for p in points_down]
    paths_down = [Path(p) for p in points_down]
    patches_down = [patches.PathPatch(p, edgecolor=Ecolor, facecolor='none', lw=lwm) for p in paths_down]
    for p in patches_down:
        ax.add_patch(p) #画线的操作 ax.image.show 写背景图像


    ## 合成二级血管
    # 生成一级血管(up)的二级血管
    # up0的二级血管
    points_s_up0 = [c.evaluate_multi(np.linspace(s, e, 50)).T for c, s, e in zip(curves_s_up0, start_s_up0, end_s_up0)]
    points_s_up0 = [wrap_points(p) for p in points_s_up0]
    paths_s_up0 = [Path(p) for p in points_s_up0]
    patches_s_up0 = [patches.PathPatch(p, edgecolor=Ecolor, facecolor='none', lw=lws) for p in paths_s_up0]
    for p in patches_s_up0:
        ax.add_patch(p)
    # 生成一级血管(down)的二级血管
    # down0的二级血管
    points_s_down0 = [c.evaluate_multi(np.linspace(s, e, 50)).T for c, s, e in zip(curves_s_down0, start_s_down0, end_s_down0)]
    points_s_down0 = [wrap_points(p) for p in points_s_down0]
    paths_s_down0 = [Path(p) for p in points_s_down0]
    patches_s_down0 = [patches.PathPatch(p, edgecolor=Ecolor, facecolor='none', lw=lws) for p in paths_s_down0]
    for p in patches_s_down0:
        ax.add_patch(p)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(512, 512, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    # label
    label = compare(img, bg_im2, Ecolor)

    # save 
    img = cv2.GaussianBlur(img, (7, 7), 0)
    mmcv.imwrite(img, out_filename)
    mmcv.imwrite(label, label_outfile)

    #
    plt.close()


if __name__ == '__main__':
    np.random.seed(2023)
    random.seed(2023)

    num_gen_train = 5000
    raw_img_path='/data/lesion/ddr/train/img/'
    save_image_path='./gendata/train/img/'
    save_label_path='./gendata/train/label/'
    
    raw_files = os.listdir(raw_img_path)

    for i in range(num_gen_train):
        print('processing %d' %i)
        ind = np.random.randint(0,len(raw_files))
        raw_img = os.path.join(raw_img_path, raw_files[ind])
        gen_img_path = save_image_path+str(i)+'.jpg'
        gen_label_path = save_label_path+str(i)+'.png'

        ## 更改生成主血管的弯的数量
        degree = 2

        # 更改颜色
        edgecolor = (random.randint(30, 100)/100, random.randint(0, 80)/100, random.randint(0, 80)/100)

        # 更改主线宽度
        lw_main = np.random.uniform(2, 4)
        # 更改支线宽度
        lw_second = np.random.uniform(0.5, 1.5)

        process_img(raw_img, gen_img_path, gen_label_path, edgecolor, lw_main, lw_second, degree)

    num_gen_test = 100
    raw_img_path='/data/lesion/ddr/train/img/'
    save_image_path='./gendata/test/img/'
    save_label_path='./gendata/test/label/'
    
    
    raw_files = os.listdir(raw_img_path)

    for i in range(num_gen_test):
        print('processing %d' %i)
        ind = np.random.randint(0, len(raw_files))
        raw_img = os.path.join(raw_img_path, raw_files[ind])
        gen_img_path = save_image_path+str(i)+'.jpg'
        gen_label_path = save_label_path+str(i)+'.png'

        ## 更改生成主血管的弯的数量
        degree = 2

        # 更改颜色
        edgecolor = (random.randint(30, 100)/100, random.randint(0, 80)/100, random.randint(0, 80)/100)

        # 更改主线宽度
        lw_main = np.random.uniform(2, 4)
        # 更改支线宽度
        lw_second = np.random.uniform(0.5, 1.5)

        process_img(raw_img, gen_img_path, gen_label_path, edgecolor, lw_main, lw_second, degree)
