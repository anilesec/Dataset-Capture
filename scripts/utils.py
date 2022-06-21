import open3d as o3d

def read_o3d_pcd(file_path):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(file_path)

        return pcd