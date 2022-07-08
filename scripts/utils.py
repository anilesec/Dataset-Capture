import open3d as o3d

def read_o3d_pcd(file_path):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(file_path)

        return pcd

def write_o3d_pcd(file_path, pcd_o3d):
        import open3d as o3d
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        o3d.io.write_point_cloud(file_path, pcd_o3d, write_ascii=True)

        return print(f"saved: {file_path}")

def imgs2vid_ffmpeg(imgs_dir, file_pth, ext='png',  frm_rate=10):
    import os
    print(f"ffmpeg creating video...")
    cmd = f"ffmpeg -hide_banner -loglevel error -framerate {frm_rate} -pattern_type glob -i '{imgs_dir}/*.{ext}' -c:v " f"libx264 -vf fps=30 -pix_fmt yuv420p {file_pth} -y "
    os.system(cmd)
   # os.system(f"rm {imgs_dir}/*.jpg")s
    return print(f"video saved here: {file_pth}")