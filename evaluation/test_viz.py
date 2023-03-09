import matplotlib.pyplot as plt

def create_juxt_vid(filepath, inp_imgs, jts_order, all_2d_jts=None, all_3d_jts_rt=None, all_3d_jts_cam=None,
                    all_3d_jts_prcst_algnd=None):
    seq_len = inp_imgs.shape[0]
    tmp_dir = 'out/juxt/imgs'
    os.makedirs(tmp_dir, exist_ok=True)

    imgs = []
    res = None
    for i in tqdm(range(seq_len)):
        # breakpoint()
        if all_2d_jts is not None:
            img1 = draw_projtd_handjts(im=inp_imgs[i], jts2d=all_2d_jts[i].reshape(-1, 21, 2), jts_order=jts_order,
                                       line_type=cv2.LINE_AA, colors=((255, 0, 0), (255, 0, 0)))  # red color
            res = img1.shape[0] // 2
            imgs.append(img1)

        if all_3d_jts_rt is not None:
            # img2 = viz_hand_jts3d(hand_jts=all_3d_jts_rt[i], jts_order='DOPE', grid_axis='ON', line_sz=2, dot_sz=2,
            # elev=15, azim=45, xlim=[-0.2, 0.2], ylim=[-0.2, 0.2], zlim=[-0.2, 0.2], resln=img1.shape[0] // 2,
            # title='3D Joints in wrist frame') elev = 15, azim = 45,
            img2 = viz_hand_jts3d(hand_jts=all_3d_jts_rt[i], jts_order=jts_order, grid_axis='ON',
                                  line_sz=2, dot_sz=4, elev=-90, azim=-90,
                                  xlim=[all_3d_jts_rt[:, :, 0].min() - 0.1, all_3d_jts_rt[:, :, 0].max() + 0.1],
                                  ylim=[all_3d_jts_rt[:, :, 1].min() - 0.1, all_3d_jts_rt[:, :, 1].max() + 0.1],
                                  zlim=[all_3d_jts_rt[:, :, 2].min() - 0.1, all_3d_jts_rt[:, :, 2].max() + 0.1],
                                  resln=res, title='3D Joints in wrist frame')
            if res is not None:
                img2 = np.concatenate([np.zeros((img1.shape[0] // 2, *img2.shape[1:]), np.uint8), img2])
            imgs.append(img2)

        if all_3d_jts_cam is not None:
            # img3 = viz_hand_jts3d(hand_jts=all_3d_jts_cam[i], jts_order='DOPE', grid_axis='ON', line_sz=1, dot_sz=1,
            #                       elev=-90, azim=-90, xlim=[-1.0, 1.0], ylim=[-1.0, 1.0], zlim=[-1.0, 1.0],
            #                       resln=img1.shape[0] // 2, title='3D Joints in camera Frame (wrist to camera frame)')
            img3 = viz_hand_jts3d(hand_jts=all_3d_jts_cam[i], jts_order=jts_order, grid_axis='ON',
                                  line_sz=2, dot_sz=2, elev=-45, azim=-90,
                                  xlim=[all_3d_jts_cam[:, :, 0].min() - 0.1, all_3d_jts_cam[:, :, 0].max() + 0.1],
                                  ylim=[all_3d_jts_cam[:, :, 1].min() - 0.1, all_3d_jts_cam[:, :, 1].max() + 0.1],
                                  zlim=[all_3d_jts_cam[:, :, 2].min() - 0.1, all_3d_jts_cam[:, :, 2].max() + 0.1],
                                  resln=res, title='3D Joints in camera Frame (wrist to camera frame)')
            if res is not None:
                img3 = np.concatenate([np.zeros((img1.shape[0] // 2, *img3.shape[1:]), np.uint8), img3])
            imgs.append(img3)

        if all_3d_jts_prcst_algnd is not None:
            img4 = viz_hand_jts3d(hand_jts=all_3d_jts_prcst_algnd[i], jts_order=jts_order, grid_axis='ON',
                                  line_sz=2, dot_sz=4, elev=-90, azim=-90,
                                  xlim=[all_3d_jts_prcst_algnd[:, :, 0].min() - 0.1,
                                        all_3d_jts_prcst_algnd[:, :, 0].max() + 0.1],
                                  ylim=[all_3d_jts_prcst_algnd[:, :, 1].min() - 0.1,
                                        all_3d_jts_prcst_algnd[:, :, 1].max() + 0.1],
                                  zlim=[all_3d_jts_prcst_algnd[:, :, 2].min() - 0.1,
                                        all_3d_jts_prcst_algnd[:, :, 2].max() + 0.1],
                                  resln=res, title='3D Joints after procrust alignment')
            if res is not None:
                img4 = np.concatenate([np.zeros((img1.shape[0] // 2, *img4.shape[1:]), np.uint8), img4])
            imgs.append(img4)

        disp_img = np.hstack(imgs)
        # breakpoint()
        Image.fromarray(disp_img).save(f"{tmp_dir}/{i:03d}.jpg")
        imgs = []

        return disp_img
    
def viz_hand_jts3d(hand_jts, jts_order, grid_axis='ON', line_sz=2, dot_sz=2, elev=15, azim=45, xlim=1., ylim=1.,
                   zlim=1., resln=None, title=None):
    """
    @param resln: resolution
    @param hand_jts: (N, K, 3)
    @return: visualized joints img
    """
    plt.style.use('dark_background')
    fig = plt.figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(projection='3d')

    if jts_order == 'CP':
        # hand skeleton structure
        # Contactpose labels order
        thumb_idxs = [0, 1, 2, 3, 4]
        indxfing_idxs = [0, 5, 6, 7, 8]
        midfing_idxs = [0, 9, 10, 11, 12]
        ringfing_idxs = [0, 13, 14, 15, 16]
        litfing_idxs = [0, 17, 18, 19, 20]
    elif jts_order == 'DOPE':
        # DOPE dets order
        thumb_idxs = [0, 1, 6, 7, 8]
        indxfing_idxs = [0, 2, 9, 10, 11]
        midfing_idxs = [0, 3, 12, 13, 14]
        ringfing_idxs = [0, 4, 15, 16, 17]
        litfing_idxs = [0, 5, 18, 19, 20]
    elif jts_order == 'OURS':
        thumb_idxs = [0, 1, 6, 11, 16]
        indxfing_idxs = [0, 2, 7, 12, 17]
        midfing_idxs = [0, 3, 8, 13, 18]
        ringfing_idxs = [0, 4, 9, 14, 19]
        litfing_idxs = [0, 5, 10, 15, 20]
    else:
        ValueError("Specify hand joints order type like ('DOPE', 'CONACTPOSE'")

    hand_jts = hand_jts.reshape(-1, 21, 3)
    for joints in hand_jts:
        ax.scatter3D(joints[:, 0], joints[:, 1], joints[:, 2], color='r', s=dot_sz)
        plt.plot(joints[thumb_idxs][:, 0], joints[thumb_idxs][:, 1], joints[thumb_idxs][:, 2], marker='.',
                 markersize=line_sz, color='r')
        plt.plot(joints[indxfing_idxs][:, 0], joints[indxfing_idxs][:, 1], joints[indxfing_idxs][:, 2], marker='.',
                 markersize=line_sz, color='g')
        plt.plot(joints[midfing_idxs][:, 0], joints[midfing_idxs][:, 1], joints[midfing_idxs][:, 2], marker='.',
                 markersize=line_sz, color='b')
        plt.plot(joints[ringfing_idxs][:, 0], joints[ringfing_idxs][:, 1], joints[ringfing_idxs][:, 2], marker='.',
                 markersize=line_sz, color='y')
        plt.plot(joints[litfing_idxs][:, 0], joints[litfing_idxs][:, 1], joints[litfing_idxs][:, 2], marker='.',
                 markersize=line_sz, color='m')

    # legends and ticks
    ax.view_init(elev, azim)  # 0 45 90 315
    ax.dist = 8

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # ax.set_xlabel('X axis', labelpad=-12)
    # ax.set_ylabel('Y axis', labelpad=-12)
    # ax.set_zlabel('Z axis', labelpad=-12)

    if isinstance(xlim, list) and len(xlim) == 2:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim([-1 * xlim, xlim])
    if isinstance(ylim, list) and len(ylim) == 2:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([-1 * ylim, ylim])
    if isinstance(zlim, list) and len(zlim) == 2:
        ax.set_zlim(zlim)
    else:
        ax.set_zlim([-1 * zlim, zlim])

    plt.title(title)

    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])

    # switch off grid lines and axis info
    if grid_axis == 'OFF':
        plt.axis('off')
        plt.grid(b=None)

    # ax.set_proj_type('ortho')

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.clf()
    plt.close('all')

    if img.shape[0] != resln and resln is not None:
        from PIL import Image
        img = np.asarray(Image.fromarray(img).resize((resln, resln)))

    return img


jts3d = jts3d_minhand
jts_img = viz_hand_jts3d(hand_jts=jts3d, jts_order='CP', grid_axis='ON',
                                  line_sz=2, dot_sz=4, elev=-90, azim=-90,
                                  xlim=[jts3d[:, 0].min(), jts3d[:, 0].max()],
                                  ylim=[jts3d[:, 1].min(), jts3d[:, 1].max()],
                                  zlim=[jts3d[:, 2].min(), jts3d[:, 2].max()],
                                  resln=None, title='3D Joints min-hand')

gt = jts3d_ann
gt_img = viz_hand_jts3d(hand_jts=gt, jts_order='OURS', grid_axis='ON',
                                  line_sz=2, dot_sz=4, elev=-90, azim=-90,
                                  xlim=[gt[:, 0].min(), gt[:, 0].max()],
                                  ylim=[gt[:, 1].min(), gt[:, 1].max()],
                                  zlim=[gt[:, 2].min(), gt[:, 2].max()],
                                  resln=None, title='GT 3D Joints')