import cv2
import numpy as np
import os
from math import pi


def _adaptive_and_close(gray, block_size=15, C=10, closing_kernel=(7, 7)):
    """Adaptive threshold + morphological closing to fill holes in filled bubbles."""
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, closing_kernel)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def _filter_contours(contours, min_area=50, max_area=5000, min_circularity=0.25):
    kept = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circularity = 4 * pi * area / (peri * peri)
        # relaxed filter: allow lower circularity for small areas
        if circularity < min_circularity and area < (min_area * 4):
            continue
        M = cv2.moments(c)
        if M.get("m00", 0) == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        kept.append({
            "contour": c,
            "area": area,
            "peri": peri,
            "circularity": circularity,
            "center": (cx, cy)
        })
    return kept


def _cluster_into_grid(centers, rows=100, cols=4):
    """Cluster centers into a grid for OMR sheets.
    For 100 questions with 4 options each, arranged in 5 columns of 20 questions.
    """
    if len(centers) == 0:
        return [[None] * cols for _ in range(rows)]

    # Sort centers by position (left to right, top to bottom)
    centers_sorted = sorted(centers, key=lambda c: (c[0], c[1]))
    
    # Group into clusters of 4 (each representing one question's options A,B,C,D)
    questions = []
    for i in range(0, len(centers_sorted), 4):
        question_bubbles = centers_sorted[i:i+4]
        if len(question_bubbles) == 4:
            # Sort by Y position (top to bottom) for this question
            question_bubbles = sorted(question_bubbles, key=lambda c: c[1])
            questions.append(question_bubbles)
    
    # Create the grid
    grid = [[None for _ in range(cols)] for _ in range(rows)]
    for q_idx, question_bubbles in enumerate(questions):
        if q_idx < rows:
            for opt_idx, center in enumerate(question_bubbles):
                if opt_idx < cols:
                    grid[q_idx][opt_idx] = center
    
    return grid


def _visualize(image, detections, grid=None, page_box=None, out_path=None):
    vis = image.copy()
    for i, d in enumerate(detections):
        cnt = d["contour"]
        # draw contour only (no red centroid dots)
        cv2.drawContours(vis, [cnt], -1, (0, 180, 0), 1)
    if grid is not None:
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell is None:
                    continue
                x, y = cell
                # draw small hollow marker for grid positions (magenta)
                cv2.circle(vis, (x, y), 5, (255, 0, 255), 1)
    # draw detected OMR bounding box if available (red rectangle)
    if page_box is not None:
        px, py, pw, ph = page_box
        cv2.rectangle(vis, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, vis)
    return vis


def detect_bubbles(gray_img, expected_grid=(20, 20), debug=False, use_answers_box=True):
    """Detect bubbles using adaptive threshold + closing, relaxed filtering.

    Returns a dict: {count, detections, centers, grid, thresh, visualization}
    """
    # Try to find the OMR bounding box (prefer large rectangular contours)
    h, w = gray_img.shape[:2]
    img_area = float(w * h)
    page_box = None

    # Primary: edge-based rectangular detection (fast)
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # quick rotation estimate (median angle of near-horizontal Hough lines)
    rotation_angle = 0.0
    rotated = False
    try:
        lines = cv2.HoughLinesP(edged, 1, np.pi/180.0, threshold=80, minLineLength=100, maxLineGap=20)
        if lines is not None:
            angles = []
            for l in lines:
                x1, y1, x2, y2 = l[0]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                ang = np.degrees(np.arctan2(dy, dx))
                # keep near-horizontal lines only
                if abs(ang) < 45:
                    angles.append(ang)
            if angles:
                rotation_angle = float(np.median(angles))
                if abs(rotation_angle) > 0.5:
                    rotated = True
    except Exception:
        rotation_angle = 0.0
        rotated = False
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts_sorted:
            area = cv2.contourArea(c)
            if area < 0.05 * img_area:
                break
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and area > 0.20 * img_area:
                x, y, ww, hh = cv2.boundingRect(approx)
                page_box = (x, y, ww, hh)
                break

    # Fallback A: morphological mask-based search for large rectangular region
    if page_box is None:
        # create a mask that highlights large page-like areas
        thr = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 51, 10)
        thr = 255 - thr
        # close gaps to form a single connected page region
        big_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 55))
        closed_mask = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, big_kernel, iterations=2)
        cnts2, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts2:
            cnts2_sorted = sorted(cnts2, key=cv2.contourArea, reverse=True)
            for c in cnts2_sorted:
                area = cv2.contourArea(c)
                if area < 0.10 * img_area:
                    break
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) >= 4:
                    x, y, ww, hh = cv2.boundingRect(approx)
                    # accept if it covers reasonable area
                    if area > 0.18 * img_area:
                        page_box = (x, y, ww, hh)
                        break

    # Fallback B: infer page box from density of candidate bubble centers
    if page_box is None:
        # do a quick full-image detection to get centers
        full_closed = _adaptive_and_close(gray_img)
        cnts3, _ = cv2.findContours(full_closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cand = _filter_contours(cnts3)
        if cand:
            centers = [d["center"] for d in cand]
            xs = [c[0] for c in centers]
            ys = [c[1] for c in centers]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            # add margin
            margin_x = int(0.03 * w)
            margin_y = int(0.03 * h)
            x0 = max(0, x_min - margin_x)
            y0 = max(0, y_min - margin_y)
            x1 = min(w, x_max + margin_x)
            y1 = min(h, y_max + margin_y)
            area_box = (x1 - x0) * (y1 - y0)
            if area_box > 0.15 * img_area:
                page_box = (x0, y0, x1 - x0, y1 - y0)

    # Final fallback: use largest detected contour bounding rect
    if page_box is None and cnts:
        largest = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
        if len(approx) >= 4:
            x, y, ww, hh = cv2.boundingRect(approx)
        else:
            x, y, ww, hh = cv2.boundingRect(largest)
        page_box = (x, y, ww, hh)

    # If page_box found, crop to that region for detection
    offset_x, offset_y = 0, 0
    page_box_display = None
    if page_box is not None:
        x, y, ww, hh = page_box
        # add a small padding but clamp to image
        pad = 10
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + ww + pad)
        y1 = min(h, y + hh + pad)
        crop = gray_img[y0:y1, x0:x1]
        offset_x, offset_y = x0, y0
        page_box_display = (x0, y0, x1 - x0, y1 - y0)
        if debug:
            print(f"Using detected OMR box: x={x0}, y={y0}, w={x1-x0}, h={y1-y0}")
    else:
        crop = gray_img

    closed = _adaptive_and_close(crop)
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = _filter_contours(contours)
    # translate centers back to original image coords if we cropped
    for d in detections:
        cx, cy = d["center"]
        d["center"] = (cx + offset_x, cy + offset_y)
        # also translate contour points
        d["contour"] = d["contour"] + np.array([[[offset_x, offset_y]]])
    centers = [d["center"] for d in detections]

    # Compute an answers-box robustly using percentile trimming to ignore outliers
    answers_box = None
    if centers:
        pts = np.array(centers)
        xs = pts[:, 0]
        ys = pts[:, 1]

        def try_percentiles(low_pct, high_pct):
            x_min = np.percentile(xs, low_pct)
            x_max = np.percentile(xs, high_pct)
            y_min = np.percentile(ys, low_pct)
            y_max = np.percentile(ys, high_pct)
            pad_x = int(0.02 * w)
            pad_y = int(0.02 * h)
            x0 = int(max(0, x_min - pad_x))
            y0 = int(max(0, y_min - pad_y))
            x1 = int(min(w, x_max + pad_x))
            y1 = int(min(h, y_max + pad_y))
            return (x0, y0, x1 - x0, y1 - y0)

        # Try wider percentiles first to keep edge rows (1st-99th)
        tried = []
        candidates = [(1, 99), (3, 97), (8, 92)]
        accepted = None
        for low, high in candidates:
            cand = try_percentiles(low, high)
            ax, ay, aw, ah = cand
            area_frac = (aw * ah) / float(w * h)
            count_inside = sum(1 for (cx, cy) in centers if ax <= cx <= ax + aw and ay <= cy <= ay + ah)
            min_centers = max(8, int(0.45 * len(centers)))
            # accept if it contains a reasonable share of centers and not almost whole image
            if count_inside >= min_centers and area_frac < 0.85:
                accepted = (ax, ay, aw, ah)
                break
            tried.append((low, high, cand, count_inside, area_frac))

        if accepted is None:
            # give up if none matched stricter criteria
            answers_box = None
        else:
            ax, ay, aw, ah = accepted
            # iterative expansion to include missing edge rows: expand until we have
            # approximately the expected number of rows (or a sensible limit)
            rows_expected, _ = expected_grid

            def _count_row_bins(ax_i, ay_i, aw_i, ah_i):
                ys_in = [cy for (cx, cy) in centers if ax_i <= cx <= ax_i + aw_i and ay_i <= cy <= ay_i + ah_i]
                if not ys_in:
                    return 0
                hist, _ = np.histogram(ys_in, bins=rows_expected)
                return int(np.count_nonzero(hist))

            bins_present = _count_row_bins(ax, ay, aw, ah)
            # target: at least rows_expected-1 bins present (allow one missing), or at least 90% of expected
            target_bins = max(rows_expected - 1, int(0.9 * rows_expected))
            expand_step = int(0.02 * h)
            if expand_step < 2:
                expand_step = 2
            expanded = 0
            expand_limit = int(0.20 * h)
            while bins_present < target_bins and expanded < expand_limit:
                ax = max(0, ax - expand_step)
                ay = max(0, ay - expand_step)
                aw = min(w - ax, aw + 2 * expand_step)
                ah = min(h - ay, ah + 2 * expand_step)
                expanded += expand_step
                bins_present = _count_row_bins(ax, ay, aw, ah)

            # final safety clamp: don't let answers_box cover entire image
            area_frac_final = (aw * ah) / float(w * h)
            if area_frac_final >= 0.95:
                # too big — discard
                answers_box = None
            else:
                answers_box = (ax, ay, aw, ah)

    # Filter detections to those inside the answers_box (if computed and enabled)
    if use_answers_box and answers_box is not None:
        ax, ay, aw, ah = answers_box
        filtered = []
        for d in detections:
            cx, cy = d["center"]
            if ax <= cx <= ax + aw and ay <= cy <= ay + ah:
                filtered.append(d)
        detections = filtered
        centers = [d["center"] for d in detections]
    rows, cols = expected_grid
    grid = _cluster_into_grid(centers, rows=rows, cols=cols)
    vis = _visualize(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR), detections, grid=grid, page_box=page_box_display)
    # draw answers box (blue) if present — only horizontal borders (top and bottom)
    if answers_box is not None:
        ax, ay, aw, ah = answers_box
        # top horizontal
        cv2.line(vis, (ax, ay), (ax + aw, ay), (255, 0, 0), 2)
        # bottom horizontal
        cv2.line(vis, (ax, ay + ah), (ax + aw, ay + ah), (255, 0, 0), 2)
    if debug:
        print(f"✅ Detected {len(detections)} candidate bubbles (after relaxed filtering)")
    return {
        "count": len(detections),
        "detections": detections,
        "centers": centers,
        "grid": grid,
        "thresh": closed,
    "page_box": page_box_display,
    "answers_box": answers_box,
    "visualization": vis,
    "rotation_angle": rotation_angle,
    "rotated": rotated,
    "use_answers_box": use_answers_box
    }

