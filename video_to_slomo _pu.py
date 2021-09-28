import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#!/usr/bin/env python3
import argparse
# import os
import os.path
import ctypes
from shutil import rmtree, move
from PIL import Image
import torch
import torchvision.transforms as transforms
import model
import dataloader
import platform
from tqdm import tqdm
import dxchange
import numpy as np
import cv2
import shutil
# For parsing commandline arguments
def red_stack_flow_out(path,n):
    prj = []
    for n in range(1,n+1):
        file = path + '%s.png'% n
        p = cv2.imread(file,0)
        prj.append(p)
    pr = np.array(prj)
    return pr
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".tiff",".tif"])

def red_stack_tiff_inter(path,inter):
    files00 = os.listdir(path)
    files0 = []
    for file in files00:
        if is_image_file(file):
            files0.append(file)
    files = files0[::inter]
    prj = []
    for n,file in enumerate(files):
        if is_image_file(file):
            p = dxchange.read_tiff(path + file)
            prj.append(p)
    pr = np.array(prj)
    return pr
def red_stack_tiff(path):
    files = os.listdir(path)
    prj = []
    for n,file in enumerate(files):
        if is_image_file(file):
            p = dxchange.read_tiff(path + file)
            prj.append(p)
    pr = np.array(prj)
    return pr
def get_args(inter,path_in,path_out):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg_dir", type=str, default="", help='path to ffmpeg.exe')
    parser.add_argument("--inputPath", type=str, required=True, help='input path')
    parser.add_argument("--checkpoint", type=str, required=True, help='path of checkpoint for pretrained model')
    parser.add_argument("--fps", type=float, default=30, help='specify fps of output video. Default: 30.')
    parser.add_argument("--sf", type=int, required=True,
                        help='specify the slomo factor N. This will increase the frames by Nx. Example sf=2 ==> 2x frames')
    parser.add_argument("--batch_size", type=int, default=1,
                        help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
    parser.add_argument("--outputPath", type=str, default="output.mkv",
                        help='Specify output file name. Default: output.mkv')
    # parser.add_argument("--extractionDir", type=str, default="", help='zhongjianjieguo')
    return parser.parse_args((['--ffmpeg=G:/SloMo/Super-SloMo-master/ffmpeg/bin/',
                              '--inputPath=%s'%path_in,
                              '--sf=%s'%inter, '--fps=30',
                              '--checkpoint=G:/SloMo/Super-SloMo-master/mode/SuperSloMo.ckpt',
                              '--outputPath=%s'%path_out]))
def main(inter,path_in,path_out):

    args = get_args(inter,path_in,path_out)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean = [0.429, 0.431, 0.397]
    std  = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean,
                                     std=std)

    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)

    # Temporary fix for issue #7 https://github.com/avinashpaliwal/Super-SloMo/issues/7 -
    # - Removed per channel mean subtraction for CPU.
    if (device == "cpu"):
        transform = transforms.Compose([transforms.ToTensor()])
        TP = transforms.Compose([transforms.ToPILImage()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        TP = transforms.Compose([transforms.ToPILImage()])
    if not os.path.exists(args.outputPath):
        os.makedirs(args.outputPath)
    # Load data
    videoFrames = dataloader.Video(root=args.inputPath, transform=transform)
    videoFramesloader = torch.utils.data.DataLoader(videoFrames, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    flowComp = model.UNet(6, 4)
    flowComp.to(device)
    for param in flowComp.parameters():
        param.requires_grad = False
    ArbTimeFlowIntrp = model.UNet(20, 5)
    ArbTimeFlowIntrp.to(device)
    for param in ArbTimeFlowIntrp.parameters():
        param.requires_grad = False

    flowBackWarp = model.backWarp(videoFrames.dim[0], videoFrames.dim[1], device)
    flowBackWarp = flowBackWarp.to(device)

    dict1 = torch.load(args.checkpoint, map_location='cpu')
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])

    # Interpolate frames
    frameCounter = 1

    with torch.no_grad():
        for _, (frame0, frame1) in enumerate(tqdm(videoFramesloader), 0):

            I0 = frame0.to(device)
            I1 = frame1.to(device)

            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]

            # Save reference frames in output folder
            for batchIndex in range(args.batch_size):
                (TP(frame0[batchIndex].detach())).\
                    resize(videoFrames.origDim, Image.BILINEAR).\
                    save(os.path.join(args.outputPath, str(frameCounter + args.sf * batchIndex) + ".png"))
            frameCounter += 1

            # Generate intermediate frames
            for intermediateIndex in range(1, args.sf):
                t = float(intermediateIndex) / args.sf
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = flowBackWarp(I1, F_t_1)

                intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1   = 1 - V_t_0

                g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)

                wCoeff = [1 - t, t]

                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                # Save intermediate frame
                for batchIndex in range(args.batch_size):
                    (TP(Ft_p[batchIndex].cpu().detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(args.outputPath, str(frameCounter + args.sf * batchIndex) + ".png"))
                frameCounter += 1

            # Set counter accounting for batching of frames
            frameCounter += args.sf * (args.batch_size - 1)
    (TP(frame1[-1].detach())).resize(videoFrames.origDim, Image.BILINEAR).save(
        os.path.join(args.outputPath, str(frameCounter + args.sf * batchIndex) + ".png"))
def fty_make_rename(path,ph_re,file):
    flies = os.listdir(path)
    lens = len(flies)
    tomo = red_stack_flow_out(path,lens)
    for i_na, re in enumerate(tomo):
        if i_na < 10:
            dxchange.writer.write_tiff(re, ph_re + '%s000%s.tiff' % (file, i_na))
        elif 9 < i_na < 100:

            dxchange.writer.write_tiff(re, ph_re + '%s00%s.tiff' % (file, i_na))
        elif 99 < i_na < 1000:

            dxchange.writer.write_tiff(re, ph_re + '%s0%s.tiff' % (file, i_na))
        else:
            dxchange.writer.write_tiff(re, ph_re + '%s%s.tiff' % (file, i_na))
def fty_make_renames():
    path_input = r'H:\LirMn_charged_Ni_imghandled_flow360\output/'
    files = os.listdir(path_input)
    for i_na, file in enumerate(files):
        ph_re = 'H:\LirMn_charged_Ni_imghandled_flow360\output_rename/%s/'%file
        tomo = red_stack_flow_out(path_input+file+'/', 359)
        for i_na, re in enumerate(tomo):
            if i_na < 10:
                dxchange.writer.write_tiff(re, ph_re + '%s000%s.tiff' % (file, i_na))
            elif 9 < i_na < 100:

                dxchange.writer.write_tiff(re, ph_re + '%s00%s.tiff' % (file, i_na))
            elif 99 < i_na < 1000:

                dxchange.writer.write_tiff(re, ph_re + '%s0%s.tiff' % (file, i_na))
            else:
                dxchange.writer.write_tiff(re, ph_re + '%s%s.tiff' % (file, i_na))
def fty_dao_ru_180_zhang():
    path_input = r'H:\LirMn_charged_Ni_imghandled_flow360\output_rename/'
    files = os.listdir(path_input)
    for i_na, file in enumerate(files):
        ph_re = r'H:\LirMn_charged_Ni_imghandled_flow\output_rename/%s/'%file
        tomo = dxchange.read_tiff(path_input+file+'/'+'%s0358.tiff' % (file))
        dxchange.writer.write_tiff(tomo, ph_re + '%s0179.tiff' % (file.split('_')[0]))
def fty_dao_chu_slomo_make_ot():
    path_input = r'H:\LirMn_charged_Ni_imghandled_flow360\output/'
    files = os.listdir(path_input)
    for i_na, file in enumerate(files):
        ph_re = r'H:\LirMn_charged_Ni_imghandled_flow_or\%s/' % file
        tomo = red_stack_tiff(path_tomo, 2)
        for i_na, proj in enumerate(tomo):
            dxchange.write_tiff(proj, ph_re + 'pro_{}_{:05d}.tiff'.format(file, i_na))
def fty_test_pu():
    path_all = r'H:\LirMn_charged_Ni_imghandled_out/'
    files = os.listdir(path_all)
    for i_na, file in enumerate(files):
        path_tomo = r'H:\LirMn_charged_Ni_imghandled_out\%s/' % file
        path_in = r'H:\LirMn_charged_Ni_imghandled_flow360\input/%s_eV/' % file.split('_')[0]
        path_out = r'H:\LirMn_charged_Ni_imghandled_flow360\output/%s_eV/' % file.split('_')[0]
        path_out_rename = r'H:\LirMn_charged_Ni_imghandled_flow360\output_rename/%s_ev/' % file.split('_')[0]
        inter = 1
        name_ev = file.split('_')[0]
        tomo = red_stack_tiff(path_tomo, inter)
        tomo = tomo * 255.
        tomo = tomo.astype(np.uint32)
        for i_na, proj in enumerate(tomo):
            dxchange.write_tiff(proj, path_in + 'pro_{}_{:05d}.tiff'.format(name_ev, i_na))
        inter = 2
        main(inter, path_in, path_out)
        fty_make_rename(path_out, path_out_rename, file=file.split('_')[0])
def fty_test_ss():
    path_in = r'H:\LirMN\ss\aug_90\08349.00_ev_tomo_out1_rename/'
    for i_inter in range(2,3):
        path_out = r'H:\LirMN\ss\aug_90\08349.00_ev_tomo_out%s/'%i_inter
        path_out_rename = r'H:\LirMN\ss\aug_90\08349.00_ev_tomo_out%s_rename/'%i_inter
        main(i_inter, path_in, path_out)
        fty_make_rename(path_out, path_out_rename, file='tomo_out%s' % (i_inter))
def fty_test_180_ss():
    path_in = r'H:\LirMN\ss\aug_60\tomo180/'
    index = 355
    for i_inter in range(2,7):
        path_out = r'H:\LirMN\ss\aug_60\08349.00_ev_180tomo_out%s/'%(i_inter*3)
        path_out_rename = r'H:\LirMN\ss\aug_60\08349.00_ev_tomo_out%s_rename/'%(i_inter*3)
        main(i_inter, path_in, path_out)
        fty_make_rename180(path_out, path_out_rename, file='tomo_out%s'%(i_inter*3),index =index)
        index += 177
def fty_out_nmc_pu():
    path_all = r'H:\LirMn_charged_Ni_imghandled_out/'
    files = os.listdir(path_all)
    for i_na, file in enumerate(files):
        print(file)
        name_ev = file.split('_')[0]
        path_tomo = r'H:\LirMn_charged_Ni_imghandled_out\%s/' % file
        path_in = r'H:\LirMN\ss_pu_out/tomo_or/%s_eV/' % name_ev
        tomo = red_stack_tiff(path_tomo)
        tomo = tomo * 255.
        tomo = tomo.astype(np.uint32)
        for i_na, proj in enumerate(tomo):
            dxchange.write_tiff(proj, path_in + 'pro_{}_{:05d}.tiff'.format(file, i_na))
        # path_in = r'H:\LirMN\ss_pu_out/%s_eV/' % file.split('_')[0]
        for i_inter in range(4,17):
            print(i_inter)
            path_out = r'H:\LirMN\ss_pu_out/flow_out/%s_eV/%s_eV_tomo_out%s/' %(name_ev,name_ev,i_inter)
            main(i_inter, path_in, path_out)
            path_out_rename = r'H:\LirMN\ss_pu_out/flow_out/%s_eV/%s_eV_tomo_out%s_rename/'%(name_ev,name_ev,i_inter)
            fty_make_rename(path_out, path_out_rename, file='%s_eV_tomo_out%s'%(name_ev,i_inter))
            #
            #
def fty_out_nmc_pu_silme():
    path_all = r'H:\LirMn_charged_Ni_imghandled_out/'
    files = os.listdir(path_all)
    for i_na, file in enumerate(files):
        print(file)
        name_ev = file.split('_')[0]
        path_tomo = r'H:\LirMn_charged_Ni_imghandled_out\%s/' % file
        path_in = r'H:\LirMN\ss_pu_out/tomo_or/%s_eV/' % name_ev
        tomo = red_stack_tiff(path_tomo)
        tomo = tomo * 255.
        tomo = tomo.astype(np.uint32)
        for i_na, proj in enumerate(tomo):
            dxchange.write_tiff(proj, path_in + 'pro_{}_{:05d}.tiff'.format(file, i_na))
        # path_in = r'H:\LirMN\ss_pu_out/%s_eV/' % file.split('_')[0]
        for i_inter in range(4,13,2):
            print(i_inter)
            path_out = r'H:\LirMN\ss_pu_out/flow_out/%s_eV/%s_eV_tomo_out%s/' %(name_ev,name_ev,i_inter)
            main(i_inter, path_in, path_out)
            path_out_rename = r'H:\LirMN\ss_pu_out/flow_out/%s_eV/%s_eV_tomo_out%s_rename/'%(name_ev,name_ev,i_inter)
            fty_make_rename(path_out, path_out_rename, file='%s_eV_tomo_out%s'%(name_ev,i_inter))
            #
            #
def fty_out_nmc_pu_silme90():
    path_all = r'H:\LirMn_charged_Ni_imghandled_out/'
    files = os.listdir(path_all)
    for i_na, file in enumerate(files):
        print(file)
        name_ev = file.split('_')[0]
        path_tomo = r'H:\LirMn_charged_Ni_imghandled_out\%s/' % file
        path_in = r'H:\LirMN\ss_pu_out60/tomo_or/%s_eV/' % name_ev
        tomo = red_stack_tiff_inter(path_tomo,3)
        tomo = tomo * 255.
        tomo = tomo.astype(np.uint32)
        for i_na, proj in enumerate(tomo):
            dxchange.write_tiff(proj, path_in + 'pro_{}_{:05d}.tiff'.format(file, i_na))
        # path_in = r'H:\LirMN\ss_pu_out/%s_eV/' % file.split('_')[0]
        for i_inter in range(3,19,3):
            print(i_inter)
            path_out = r'H:\LirMN\ss_pu_out60/flow_out/%s_eV/%s_eV_tomo_out%s/' %(name_ev,name_ev,i_inter)
            main(i_inter, path_in, path_out)
            path_out_rename = r'H:\LirMN\ss_pu_out60/flow_out/%s_eV/%s_eV_tomo_out%s_rename/'%(name_ev,name_ev,i_inter)
            fty_make_rename(path_out, path_out_rename, file='%s_eV_tomo_out%s'%(name_ev,i_inter))
            #
            #
def fty_make_rename180(path,ph_re,file,index):
    flies = os.listdir(path)
    lens = len(flies)
    tomo0 = red_stack_flow_out(path,lens)
    tomo =tomo0[1:]
    for i_na, re in enumerate(tomo,index):
        if i_na < 10:
            dxchange.writer.write_tiff(re, ph_re + '%s000%s.tiff' % (file, i_na))
        elif 9 < i_na < 100:

            dxchange.writer.write_tiff(re, ph_re + '%s00%s.tiff' % (file, i_na))
        elif 99 < i_na < 1000:

            dxchange.writer.write_tiff(re, ph_re + '%s0%s.tiff' % (file, i_na))
        else:
            dxchange.writer.write_tiff(re, ph_re + '%s%s.tiff' % (file, i_na))
def fty_out_nmc_pu_silme90_180():
    path_all = r'H:\LirMN\ss_pu_out60\tomo180/'
    files = os.listdir(path_all)
    for i_na, file in enumerate(files):
        print(file)
        name_ev = file.split('_')[0]
        path_tomo = r'H:\LirMN\ss_pu_out60\tomo180\%s/' % file
        index = 355
        for i_inter in range(2,7):

            print(i_inter)
            path_out = r'H:\LirMN\ss_pu_out60/flow_out180/%s_eV/%s_eV_tomo_out%s/' %(name_ev,name_ev,i_inter*3)
            main(i_inter, path_tomo, path_out)
            path_out_rename = r'H:\LirMN\ss_pu_out60/flow_out/%s_eV/%s_eV_tomo_out%s_rename/'%(name_ev,name_ev,i_inter*3)
            fty_make_rename180(path_out, path_out_rename, file='%s_eV_tomo_out%s'%(name_ev,i_inter*3),index =index)
            index+=177
def fty_nomal(data):
    data_min = np.min(data)
    data_max = np.max(data)
    data = (data-data_min)/(data_max-data_min)*255.
    return data
if __name__ == '__main__':
    path_in = r'G:\SloMo\Adobe240fps\CT_um\tutorial\tomo_ringt_255_denoise\result2/'
    path_in1 = r'G:\SloMo\Adobe240fps\CT_um\tutorial\tomo_ringt_255_denoise\result2_or/'
    data = red_stack_tiff(path_in)
    data = fty_nomal(data)
    tomo = data.astype(np.uint32)
    for i_na, proj in enumerate(tomo):
        dxchange.write_tiff(proj, path_in1 + 'pro_{:05d}.tiff'.format(i_na))
    path_out = r'G:\SloMo\Adobe240fps\CT_um\tutorial\tomo_ringt_255_denoise\result2_or_out/'
    i_inter = 5
    main(i_inter, path_in1, path_out)
    ph_re = r'G:\SloMo\Adobe240fps\CT_um\tutorial\tomo_ringt_255_denoise\result2_or_rename/'
    file = 'sele_5_'
    fty_make_rename(path_out, ph_re, file)

    # path_out = r'G:\MIRNet\MIRNet-master\data_LiNi_8354\tomo_or_255_inter10/'
    # path_out_rename = r'G:\MIRNet\MIRNet-master\data_LiNi_8354\tomo_or_255_inter10_rename/'
    # fty_make_rename(path_out, path_out_rename, file='8354_eV_tomo_out10')
    # path_in = r'G:\MIRNet\MIRNet-master\data_LiNi_8354\tomo_or/'
    # path_in_255 = r'G:\MIRNet\MIRNet-master\data_LiNi_8354\tomo_or_255/'
    # data = red_stack_tiff(path_in)
    # max = np.max(data)
    # data = data/max*255.
    # data = data.astype(np.uint32)
    # for i_na, proj in enumerate(data):
    #     dxchange.write_tiff(proj, path_in_255 + 'pro_{:05d}.tiff'.format(i_na))
    # path_out = r'G:\MIRNet\MIRNet-master\data_LiNi_8354\tomo_or_255_inter10/'
    # i_inter = 10
    # main(i_inter, path_in_255, path_out)
    # data = red_stack_tiff(path_in)
    # max = np.max(data)
    # data = data/max*255.
    # data = data.astype(np.uint32)
    # for i_na, proj in enumerate(data):
    #     dxchange.write_tiff(proj, path_in_255 + 'pro_{:05d}.tiff'.format(i_na))
    # for i_inter in range(4, 17):
    #     print(i_inter)
    #     path_out = r'G:\MIRNet\MIRNet-master\data_LiNi_8354\tomo_or_denoise_result2_inter%s/'%(i_inter)
    #     main(i_inter, path_in_255, path_out)
    #     path_out_rename = r'G:\MIRNet\MIRNet-master\data_LiNi_8354\tomo_or_denoise_result2_inter%s_rename/'%(i_inter)

