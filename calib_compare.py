# -*- coding: utf-8 -*-
import open3d as o3d
import numpy as np
import os
import copy
import pickle
import transforms3d
import json

#
def bbox(x,y,z,x_d,y_d,z_d,heading,l):

    #center [x,y,z] +/- [w,h,l/2]

    points = [[x-x_d/2, y-y_d/2, z-z_d/2], #0
              [x+x_d/2, y-y_d/2, z-z_d/2], #1
              [x-x_d/2, y+y_d/2, z-z_d/2], #2
              [x+x_d/2, y+y_d/2, z-z_d/2], #3
              [x-x_d/2, y-y_d/2, z+z_d/2], #4
              [x+x_d/2, y-y_d/2, z+z_d/2], #5
              [x-x_d/2, y+y_d/2, z+z_d/2], #6
              [x+x_d/2, y+y_d/2, z+z_d/2]]#7

    lines = [[0, 1],
             [0, 2],
             [1, 3],
             [2, 3],
             [4, 5],
             [4, 6],
             [5, 7],
             [6, 7],
             [0, 4],
             [1, 5],
             [2, 6],
             [3, 7],]

    if l==1: # 星尘-0 云测-1
        colors = [[1, 0, 0] for i in range(len(lines))]
    else:
        colors = [[0, 0, 1] for i in range(len(lines))]

    points=rigid_translate(np.asarray(points),ex([-x,-y,-z],[0,0,0]))
    points=rigid_translate(np.asarray(points),ex([0,0,0,],[0,0,heading]))
    points=rigid_translate(np.asarray(points),ex([x,y,z],[0,0,0]))

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    #o3d.visualization.draw_geometries([line_set], zoom=0.8)
    return line_set

def rigid_translate(pc_input, extrinsic):
    # projection
    scan = np.row_stack([pc_input[:, :3].T, np.ones_like(pc_input[:, 0])])
    scan = np.matmul(extrinsic, scan)
    points = np.row_stack([scan[:3, :], pc_input[:, 3:].T]).T
    return points

def ex(r,t):
    r=np.asarray(r)
    t=np.asarray(t)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = transforms3d.euler.euler2mat(t[0],t[1],t[2])
    extrinsic[:3, 3] = r.reshape(-1)
    return extrinsic

def readlabels(path):
    with open(path, "r") as f:
        data = f.readlines()
    coords=[]
    labels=[]
    for d in data:
        line = d.split(' ')
        coord=[float(line[4]),float(line[5]),float(line[6]),float(line[7]),float(line[8]),float(line[9]),float(line[10])]
        coords.append(coord)
        labels.append(line[2])
    return coords,labels

def remove_ground(pc):
    plane_model, inliers = pc.segment_plane(distance_threshold=0.2,
                                                   ransac_n=3,
                                                   num_iterations=100)
    #plane_pc=remain_pc.select_by_index(inliers) #plane
    inlier_pc=pc.select_by_index(inliers,invert=True) #non-plane
    #plane_pc.paint_uniform_color([0,1,1]) #plane-black
    return inlier_pc

def pc_show(pc,norm_flag=False):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800,height=500)
    opt = vis.get_render_option()
    opt.point_size = 3
    opt.point_show_normal=norm_flag
    for p in pc:
        vis.add_geometry(p)
    vis.run()
    vis.destroy_window()

labelspath1=os.listdir(r'./云测')

pcpath=os.listdir(r'./试标注数据/pcd')

#print(labelspath)
#print(pcpath)

count=0

for c_pc in pcpath:
    pc=o3d.io.read_point_cloud(os.path.join('./试标注数据/pcd',c_pc))
    pc=remove_ground(pc)

    #云测 txt labels
    cur_file1=os.path.join('./云测',c_pc.replace(".pcd",'.txt'))
    co1,labels1=readlabels(cur_file1)
    #print(co1,labels1)
    bbox_list=[]
    bbox_list.append(pc)
    yunce_count=0
    for cur in range(0,len(labels1)):
        c=co1[cur]
        #l=labels1[cur]
        l=1
        heading=c[6]
        #heading=c[6]/180*np.pi
        box=bbox(c[0],c[1],c[2],c[3],c[4],c[5],heading,l)
        #print(heading,l)
        bbox_list.append(box)
        yunce_count+=1

    #星x json labels 解析
    cur_file2=os.path.join('./星尘',c_pc.replace(".pcd",'.json'))
    assert os.path.exists(cur_file2)
    xingchen_count=0
    with open(cur_file2,'r',encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
        #print(load_dict)
        labellist=load_dict['task_result']['annotations'][0]['slotsChildren']
        for labels2 in labellist:
            cur_label=labels2['slot']
            #print(cur_label)
            position=cur_label['position']
            rotation=cur_label['rotation']
            scale=cur_label['scale']

            x=position['x']
            y=position['y']
            z=position['z']
            w=scale['width']
            h=scale['height']
            l=scale['depth']
            heading=rotation['_z']
            box=bbox(x,y,z,w,h,l,heading,0)
            bbox_list.append(box)
            xingchen_count+=1
    print(c_pc,"云测(red)：",yunce_count,"星尘(blue)：",xingchen_count)
    pc_show(bbox_list)

    count+=1

    



'''
for l in labelspath1:
    cur_file1=os.path.join('./云测',l)
    co1,labels1=readlabels(cur_file1)
    print(co1,labels1)
    pc=o3d.io.read_point_cloud(os.path.join('./试标注数据/pcd',pcpath[count]))
    pc=remove_ground(pc)
    bbox_list=[]
    for cur in range(0,len(labels1)):
        c=co1[cur]
        l=labels1[cur]
        heading=c[6]
        #heading=c[6]/180*np.pi
        box=bbox(c[0],c[1],c[2],c[3],c[4],c[5],heading,l)
        print(heading,l)
        bbox_list.append(box)
    bbox_list.append(pc)
    pc_show(bbox_list)
    count+=1
'''















