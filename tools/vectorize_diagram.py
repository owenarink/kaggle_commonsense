import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def find_content_bounds(img, white_threshold=245, min_nonwhite=10):
    nonwhite = np.any(img < white_threshold, axis=2)
    rows = np.where(nonwhite.sum(axis=1) > min_nonwhite)[0]
    cols = np.where(nonwhite.sum(axis=0) > min_nonwhite)[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, img.shape[0], 0, img.shape[1]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def drop_title_band(img, white_threshold=245, row_nonwhite_ratio=0.03):
    nonwhite = np.any(img < white_threshold, axis=2)
    row_counts = nonwhite.sum(axis=1)
    width = img.shape[1]
    threshold = width * row_nonwhite_ratio
    rows = np.where(row_counts > threshold)[0]
    if len(rows) == 0:
        return img
    start = rows[0]
    # Skip contiguous title rows and a small gap below.
    end = start
    while end + 1 < len(row_counts) and row_counts[end + 1] > threshold * 0.35:
        end += 1
    crop_y = min(end + 20, img.shape[0] - 1)
    return img[crop_y:, :, :]


def quantize(img, k):
    data = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.8)
    _ret, labels, centers = cv2.kmeans(
        data,
        k,
        None,
        criteria,
        4,
        cv2.KMEANS_PP_CENTERS,
    )
    centers = np.clip(np.round(centers), 0, 255).astype(np.uint8)
    quant = centers[labels.flatten()].reshape(img.shape)
    return quant


def simplify_contour(cnt, epsilon_factor=0.0025):
    peri = cv2.arcLength(cnt, True)
    epsilon = max(1.0, epsilon_factor * peri)
    return cv2.approxPolyDP(cnt, epsilon, True)


def contour_to_path(cnt):
    pts = cnt.reshape(-1, 2)
    if len(pts) < 3:
        return None
    parts = [f"M {pts[0,0]} {pts[0,1]}"]
    for p in pts[1:]:
        parts.append(f"L {p[0]} {p[1]}")
    parts.append("Z")
    return " ".join(parts)


def build_svg(paths, width, height, out_path):
    svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        version="1.1",
        width=str(width),
        height=str(height),
        viewBox=f"0 0 {width} {height}",
    )
    for fill, d in paths:
        ET.SubElement(svg, "path", d=d, fill=fill, stroke="none")
    ET.ElementTree(svg).write(out_path, encoding="utf-8", xml_declaration=True)


def vectorize(input_path: Path, output_path: Path, colors: int):
    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(input_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    rgb = drop_title_band(rgb)
    y0, y1, x0, x1 = find_content_bounds(rgb)
    rgb = rgb[y0:y1, x0:x1, :]

    quant = quantize(rgb, colors)

    paths = []
    unique_colors = np.unique(quant.reshape(-1, 3), axis=0)
    for color in unique_colors:
        if np.all(color >= 245):
            continue
        mask = np.all(quant == color, axis=2).astype(np.uint8) * 255
        # Merge tiny holes/noise but keep edges reasonably sharp.
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _hier = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        fill = rgb_to_hex(color)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 6:
                continue
            cnt = simplify_contour(cnt)
            d = contour_to_path(cnt)
            if d:
                paths.append((fill, d))

    build_svg(paths, rgb.shape[1], rgb.shape[0], output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--colors", type=int, default=20)
    args = parser.parse_args()
    vectorize(Path(args.input), Path(args.output), args.colors)


if __name__ == "__main__":
    main()
