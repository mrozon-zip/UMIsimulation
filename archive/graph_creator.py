"""
Helper function to combine 4 different svg graphs into one.
"""

import xml.etree.ElementTree as ET

def parse_svg_length(length_str):
    if length_str.endswith('px'):
        return float(length_str.replace('px', ''))
    elif length_str.endswith('pt'):
        return float(length_str.replace('pt', '')) * 96 / 72
    elif length_str.endswith('cm'):
        return float(length_str.replace('cm', '')) * 96 / 2.54
    elif length_str.endswith('mm'):
        return float(length_str.replace('mm', '')) * 96 / 25.4
    elif length_str.endswith('in'):
        return float(length_str.replace('in', '')) * 96
    else:
        # Try to just parse float if no units
        try:
            return float(length_str)
        except ValueError:
            raise ValueError(f"Unknown SVG length unit in '{length_str}'")

def get_svg_size(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()
    width = parse_svg_length(root.attrib['width'])
    height = parse_svg_length(root.attrib['height'])
    return width, height, tree, root


def join_svgs(svg_files, output_svg="combined.svg"):
    # Parse all SVGs and get their sizes
    svgs = [get_svg_size(f) for f in svg_files]
    width, height = svgs[0][0], svgs[0][1]  # Assume all are the same size

    # Create a new SVG root
    svg_ns = "http://www.w3.org/2000/svg"
    ET.register_namespace('', svg_ns)
    new_svg = ET.Element('{%s}svg' % svg_ns, {
        'width': str(width * 2),
        'height': str(height * 2),
        'xmlns': svg_ns,
    })

    positions = [
        (0, 0),
        (width, 0),
        (0, height),
        (width, height),
    ]

    for idx, (w, h, tree, root) in enumerate(svgs):
        # Create a group to wrap each SVG with translation
        g = ET.Element('g', {'transform': f'translate({positions[idx][0]},{positions[idx][1]})'})
        # Add all children (skip <svg> root itself)
        for child in list(root):
            g.append(child)
        new_svg.append(g)

    # Write to file
    ET.ElementTree(new_svg).write(output_svg, encoding='utf-8', xml_declaration=True)
    print(f"Combined SVG saved as {output_svg}")


if __name__ == "__main__":
    print("Paste the four SVG filenames (one per line):")
    svg_files = [input().strip() for _ in range(4)]
    join_svgs(svg_files)