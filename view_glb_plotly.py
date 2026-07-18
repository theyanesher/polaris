import trimesh
import argparse
import plotly.graph_objects as go

def main():
    parser = argparse.ArgumentParser(description="View a GLB file interactively with hover coordinates.")
    parser.add_argument("file_path", type=str, help="Path to the .glb file")
    args = parser.parse_args()

    try:
        print(f"Loading {args.file_path}...")
        scene = trimesh.load(args.file_path)
        
        if isinstance(scene, trimesh.Scene):
            # We assume the first geometry is the one we want
            geometry = list(scene.geometry.values())[0]
            vertices = geometry.vertices
            faces = geometry.faces
        else:
            vertices = scene.vertices
            faces = scene.faces
            
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        
        # Create a 3D mesh using Plotly
        fig = go.Figure(data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.5,
                color='lightblue',
                hoverinfo='x+y+z'  # This enables coordinate hovering!
            ),
            # Also overlay the vertices as tiny scatter points to make hovering easier
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(size=2, color='red'),
                hoverinfo='x+y+z'
            )
        ])

        fig.update_layout(
            title="Interactive 3D Mesh (Hover for Coordinates)",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        print("Opening in your web browser... You can hover over any point to see its exact coordinates.")
        fig.show()

    except Exception as e:
        print(f"Error loading {args.file_path}: {e}")
        print("Make sure you have plotly installed: pip install plotly")

if __name__ == "__main__":
    main()
