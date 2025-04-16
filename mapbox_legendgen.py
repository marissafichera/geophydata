import json
import svgwrite
import os

def generate_legend_from_mapbox_style(style_json_path, output_svg_path, include_layers=None):
    """
    Generate an SVG legend from a Mapbox Studio style JSON.

    Parameters:
    - style_json_path (str): Path to Mapbox style JSON file.
    - output_svg_path (str): Path to save the output SVG file.
    - include_layers (list): Optional. List of layer IDs to include in the legend.
    """
    # Load Mapbox style JSON
    with open(style_json_path, 'r', encoding='utf-8') as f:
        style = json.load(f)

    # Prepare drawing
    dwg = svgwrite.Drawing(output_svg_path, profile='tiny', size=("300px", "auto"))

    # Styling for legend
    rect_size = 20
    spacing = 10
    text_offset = 5
    y = spacing

    # Track added layers to avoid duplicates
    added_layers = set()

    # Process each layer in style JSON
    for layer in style.get('layers', []):
        layer_id = layer.get('id')
        paint = layer.get('paint', {})

        # Optional: Filter specific layers
        if include_layers and layer_id not in include_layers:
            continue

        # Skip already processed layers
        if layer_id in added_layers:
            continue

        # Try to get fill or line color
        color = paint.get('fill-color') or paint.get('line-color')

        # Skip if no color found
        if not color:
            continue

        # Handle expressions in color
        if isinstance(color, list):
            color = color[1] if len(color) > 1 else "#000000"

        # Draw color rectangle
        dwg.add(dwg.rect(insert=(spacing, y), size=(rect_size, rect_size), fill=color))

        # Draw text label
        dwg.add(dwg.text(layer_id, insert=(spacing + rect_size + text_offset, y + rect_size * 0.75),
                         font_size="12px", fill="black"))

        # Update positions
        y += rect_size + spacing
        added_layers.add(layer_id)

    # Save SVG
    dwg.save()
    print(f"✅ Legend saved to: {output_svg_path}")

# === Example usage ===
generate_legend_from_mapbox_style(
    style_json_path=r"C:\path\to\your\mapbox_style.json",
    output_svg_path=r"C:\path\to\output_legend.svg"
)
