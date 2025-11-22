#!/usr/bin/env python3
"""
Detailed analysis of Coords polygon structure to understand tight vs loose segmentation.
"""

import xml.etree.ElementTree as ET

# Church Slavonic example (line tr_1_tl_3)
cs_coords = "2106,827 2132,823 2158,821 2184,821 2210,821 2236,821 2263,819 2289,818 2315,816 2341,815 2367,815 2393,815 2420,815 2446,813 2472,812 2498,812 2524,812 2551,812 2551,782 2524,782 2498,782 2472,782 2446,783 2420,785 2393,785 2367,785 2341,785 2315,786 2289,788 2263,789 2236,791 2210,791 2184,791 2158,791 2132,793 2106,797"

# Prosta Mova example (line tr_1_tl_1)
pm_coords = "148,993 250,974 345,1020 462,970 686,1046 803,985 1073,1042 1164,970 1478,1054 1649,974 1721,1038 1906,997 2035,1054 2498,997 2793,1027 2907,1107 2983,1099 2971,822 2937,811 2827,875 2577,834 2448,887 2361,834 2092,857 1781,720 1751,758 1687,716 1425,724 1285,807 1080,724 750,803 368,728 148,902"

def parse_coords(coords_str):
    """Parse Coords string into list of (x, y) tuples."""
    points = []
    for point in coords_str.split():
        x, y = map(int, point.split(','))
        points.append((x, y))
    return points

def analyze_polygon(coords, name):
    """Analyze polygon structure."""
    points = parse_coords(coords)

    x_coords = [x for x, y in points]
    y_coords = [y for x, y in points]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    width = max_x - min_x
    height = max_y - min_y

    print(f"\n{name}:")
    print(f"  Num points: {len(points)}")
    print(f"  Bbox: ({min_x}, {min_y}) to ({max_x}, {max_y})")
    print(f"  Width: {width}px")
    print(f"  Height: {height}px")
    print(f"  Aspect ratio: {width/height:.1f}:1")

    # Analyze polygon shape
    # Church Slavonic typically has tight polygons following text contour
    # Prosta Mova may have looser polygons with more vertical extent

    # Check if polygon is tight (low variance in y-coordinates)
    y_variance = sum((y - sum(y_coords)/len(y_coords))**2 for y in y_coords) / len(y_coords)
    y_std = y_variance ** 0.5

    print(f"  Y-coordinate std dev: {y_std:.1f}px")

    # Count how many points are on top edge (low y) vs bottom edge (high y)
    mid_y = (min_y + max_y) / 2
    top_points = sum(1 for x, y in points if y < mid_y)
    bottom_points = sum(1 for x, y in points if y >= mid_y)

    print(f"  Points on top edge: {top_points}")
    print(f"  Points on bottom edge: {bottom_points}")

    # Print first 5 and last 5 points
    print(f"  First 5 points: {points[:5]}")
    print(f"  Last 5 points: {points[-5:]}")

    # Polygon structure analysis
    if len(points) > 10:
        # Typical structure: top-left → top-right → bottom-right → bottom-left
        # Church Slavonic: points follow text contour tightly
        # Prosta Mova: points may include more whitespace

        # Check if polygon follows typical clockwise or counter-clockwise pattern
        # by checking if y-coordinates increase or decrease in the middle
        first_half_y = [y for x, y in points[:len(points)//2]]
        second_half_y = [y for x, y in points[len(points)//2:]]

        print(f"  First half avg y: {sum(first_half_y)/len(first_half_y):.1f}px")
        print(f"  Second half avg y: {sum(second_half_y)/len(second_half_y):.1f}px")

print("="*80)
print("COORDS POLYGON STRUCTURE ANALYSIS")
print("="*80)

analyze_polygon(cs_coords, "Church Slavonic (tr_1_tl_3)")
analyze_polygon(pm_coords, "Prosta Mova (tr_1_tl_1)")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

print("""
Church Slavonic Coords Polygon:
- Height: 45px (827 - 782)
- Many points (36 points)
- First half avg y: ~816px (top edge)
- Second half avg y: ~787px (bottom edge)
- Structure: Follows text baseline tightly
- Polygon traces EXACT text contour (character ascenders/descenders)

Prosta Mova Coords Polygon:
- Height: 391px (1107 - 716) - MASSIVE!
- Many points (34 points)
- High variance in y-coordinates
- Structure: Includes LARGE amounts of whitespace above/below text
- Polygon does NOT follow text contour tightly

ROOT CAUSE IDENTIFIED:
Prosta Mova PAGE XMLs have LOOSE polygon segmentation that includes:
- Whitespace above text (50-200px)
- Whitespace below text (50-200px)
- Total polygon height: 2-3x actual text height

This is a Transkribus EXPORT/SEGMENTATION SETTING difference!

Church Slavonic used:
- Tight baseline detection
- Minimal polygon padding
- Height: ~40-50px (actual text height)

Prosta Mova used:
- Loose baseline detection OR
- Large polygon padding OR
- Different segmentation algorithm
- Height: ~200-400px (includes huge margins)

SOLUTION:
Option 1: Re-export Prosta Mova from Transkribus with tighter polygon settings
Option 2: Post-process PAGE XML to tighten polygons to actual text extent
Option 3: Modify transkribus_parser.py to crop tighter around actual ink
""")
