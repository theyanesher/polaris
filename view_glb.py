import trimesh
import argparse

def main():
    parser = argparse.ArgumentParser(description="View a GLB file in 3D.")
    parser.add_argument("file_path", type=str, help="Path to the .glb file")
    args = parser.parse_args()

    try:
        print(f"Loading {args.file_path}...")
        scene = trimesh.load(args.file_path)
        
        print("Opening 3D viewer. You can click and drag to rotate.")
        print("Tip: Middle-click or shift-click might allow you to pan.")
        
        # This will open a window on your local machine to view the mesh
        scene.show()
            
    except Exception as e:
        print(f"Error loading {args.file_path}: {e}")
        print("\nNote: If you get a display error (like 'pyglet.canvas.xlib.NoSuchDisplayException'),")
        print("it means your terminal doesn't support opening UI windows.")

if __name__ == "__main__":
    main()
