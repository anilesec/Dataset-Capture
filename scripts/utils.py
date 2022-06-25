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

