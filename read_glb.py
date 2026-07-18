import trimesh
import argparse

def main():
    parser = argparse.ArgumentParser(description="Load a GLB file and print its vertices.")
    parser.add_argument("file_path", type=str, help="Path to the .glb file")
    args = parser.parse_args()

    try:
        # Load the GLB file
        scene = trimesh.load(args.file_path)

        # Iterate through all geometries in the scene
        if isinstance(scene, trimesh.Scene):
            for name, geometry in scene.geometry.items():
                # geometry.vertices contains an Nx3 array of (x, y, z) coordinates
                print(f"Mesh: {name}")
                print(f"Vertices:\n{geometry.vertices}")
        else:
            # If it's a single mesh rather than a scene
            print(f"Loaded single mesh with {len(scene.vertices)} vertices.")
            print(f"Vertices:\n{scene.vertices}")
            
    except Exception as e:
        print(f"Error loading {args.file_path}: {e}")

if __name__ == "__main__":
    main()
