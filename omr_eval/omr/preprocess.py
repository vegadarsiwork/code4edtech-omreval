import cv2
import numpy as np


def _find_signature_and_instructions_boxes_and_orient(img, debug_out=None):
    """
    Detect both the signature box (vertical, left) and instructions box (horizontal, right).
    Rotate so both are at the top (signature left, instructions right).
    Returns: (rotated_img, angle_applied, [sig_box, instr_box])
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    vis = img.copy()
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.005 * w * h or area > 0.4 * w * h:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, ww, hh = cv2.boundingRect(approx)
            aspect = ww / hh if hh > 0 else 0
            # Only consider boxes in the top/bottom 40%
            if y < 0.4 * h or (y + hh) > 0.6 * h:
                candidates.append({'area': area, 'rect': (x, y, ww, hh), 'aspect': aspect, 'approx': approx.reshape(4,2)})
                cv2.drawContours(vis, [approx], -1, (0, 0, 255), 2)
                cv2.putText(vis, f"A={int(area)}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    if debug_out is not None:
        debug_out['instructions_candidates_img'] = vis
        debug_out['instructions_candidates'] = [c['rect'] for c in candidates]
    if len(candidates) < 2:
        # fallback: use previous logic if only one box
        if not candidates:
            return img, 0, []
        c = candidates[0]
        return img, 0, [c['rect']]
    # Heuristic: signature box is tall and thin, instructions box is wide and short
    sig_cand = min(candidates, key=lambda c: c['aspect'])
    instr_cand = max(candidates, key=lambda c: c['aspect'])
    # Try all 0/90/180/270 rotations to get both at the top, sig left, instr right
    def get_box_centers(boxes):
        return [(x + ww/2, y + hh/2) for (x, y, ww, hh) in boxes]
    def rotate_img_and_boxes(img, boxes, angle):
        if angle == 0:
            return img, boxes
        elif angle == 90:
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            new_boxes = [(h - (y + hh), x, hh, ww) for (x, y, ww, hh) in boxes]
            return rotated, new_boxes
        elif angle == 180:
            rotated = cv2.rotate(img, cv2.ROTATE_180)
            new_boxes = [(w - (x + ww), h - (y + hh), ww, hh) for (x, y, ww, hh) in boxes]
            return rotated, new_boxes
        elif angle == -90:
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            new_boxes = [(y, w - (x + ww), hh, ww) for (x, y, ww, hh) in boxes]
            return rotated, new_boxes
        return img, boxes
    best_angle = 0
    for angle in [0, 90, 180, -90]:
        test_img, test_boxes = rotate_img_and_boxes(img, [sig_cand['rect'], instr_cand['rect']], angle)
        (sx, sy, sww, shh), (ix, iy, iww, ihh) = test_boxes
        # Both boxes should be near the top (y < 0.25*h), and sig left of instr (sx < ix)
        if sy < 0.25 * h and iy < 0.25 * h and sx < ix:
            best_angle = angle
            break
    final_img, final_boxes = rotate_img_and_boxes(img, [sig_cand['rect'], instr_cand['rect']], best_angle)
    if debug_out is not None:
        debug_out['final_signature_box'] = final_boxes[0]
        debug_out['final_instructions_box'] = final_boxes[1]
        debug_out['final_orientation_angle'] = best_angle
    return final_img, best_angle, final_boxes
    """
    Detect the largest rectangle near the top 30% of the image (instructions box),
    and determine its orientation (0, 90, 180, 270 degrees).
    Returns: (rotated_img, angle_applied, box_coords)
    If debug_out is a dict, adds 'instructions_candidates_img' and 'instructions_candidates' for inspection.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    vis = img.copy()
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.005 * w * h or area > 0.4 * w * h:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, ww, hh = cv2.boundingRect(approx)
            # Only consider boxes in the top 40% or bottom 40% (for upside-down)
            if y < 0.4 * h or (y + hh) > 0.6 * h:
                candidates.append((area, (x, y, ww, hh), approx.reshape(4,2)))
                cv2.drawContours(vis, [approx], -1, (0, 0, 255), 2)
                cv2.putText(vis, f"A={int(area)}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    if debug_out is not None:
        debug_out['instructions_candidates_img'] = vis
        debug_out['instructions_candidates'] = [c[1] for c in candidates]
    if not candidates:
        return img, 0, None
    # Pick the largest such box
    candidates.sort(reverse=True)
    _, (x, y, ww, hh), box_pts = candidates[0]
    rect = _order_points(box_pts)
    (tl, tr, br, bl) = rect
    # Find which corner is closest to (0,0) (top left) and (w,0) (top right)
    dists_top_left = [np.linalg.norm(pt - np.array([0,0])) for pt in rect]
    dists_top_right = [np.linalg.norm(pt - np.array([w,0])) for pt in rect]
    idx_top_left = int(np.argmin(dists_top_left))
    idx_top_right = int(np.argmin(dists_top_right))
    # If the closest corner to top right is not actually the top right of the box, rotate accordingly
    # The order is tl, tr, br, bl. We want tr (index 1) to be closest to (w,0)
    for rot in [0, 1, 2, 3]:
        rect_rot = np.roll(rect, -rot, axis=0)
        tr_rot = rect_rot[1]
        dist_tr = np.linalg.norm(tr_rot - np.array([w,0]))
        if dist_tr < 0.1 * w:
            if rot == 0:
                return img, 0, (x, y, ww, hh)
            elif rot == 1:
                rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # recalc box for rotated image
                new_box = (y, w - (x + ww), hh, ww)
                return rotated, -90, new_box
            elif rot == 2:
                rotated = cv2.rotate(img, cv2.ROTATE_180)
                new_box = (w - (x + ww), h - (y + hh), ww, hh)
                return rotated, 180, new_box
            elif rot == 3:
                rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                new_box = (h - (y + hh), x, hh, ww)
                return rotated, 90, new_box
    # fallback: return as is
    return img, 0, (x, y, ww, hh)

    def _find_signature_and_instructions_boxes_and_orient(img, debug_out=None):
        """
        Detect both the signature box (vertical, left) and instructions box (horizontal, right).
        Rotate so both are at the top (signature left, instructions right).
        Returns: (rotated_img, angle_applied, [sig_box, instr_box])
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 50, 150)
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        vis = img.copy()
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 0.005 * w * h or area > 0.4 * w * h:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, ww, hh = cv2.boundingRect(approx)
                aspect = ww / hh if hh > 0 else 0
                # Only consider boxes in the top/bottom 40%
                if y < 0.4 * h or (y + hh) > 0.6 * h:
                    candidates.append({'area': area, 'rect': (x, y, ww, hh), 'aspect': aspect, 'approx': approx.reshape(4,2)})
                    cv2.drawContours(vis, [approx], -1, (0, 0, 255), 2)
                    cv2.putText(vis, f"A={int(area)}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        if debug_out is not None:
            debug_out['instructions_candidates_img'] = vis
            debug_out['instructions_candidates'] = [c['rect'] for c in candidates]
        if len(candidates) < 2:
            # fallback: use previous logic if only one box
            if not candidates:
                return img, 0, []
            c = candidates[0]
            return img, 0, [c['rect']]
        # Heuristic: signature box is tall and thin, instructions box is wide and short
        sig_cand = min(candidates, key=lambda c: c['aspect'])
        instr_cand = max(candidates, key=lambda c: c['aspect'])
        # Try all 0/90/180/270 rotations to get both at the top, sig left, instr right
        def get_box_centers(boxes):
            return [(x + ww/2, y + hh/2) for (x, y, ww, hh) in boxes]
        def rotate_img_and_boxes(img, boxes, angle):
            if angle == 0:
                return img, boxes
            elif angle == 90:
                rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                new_boxes = [(h - (y + hh), x, hh, ww) for (x, y, ww, hh) in boxes]
                return rotated, new_boxes
            elif angle == 180:
                rotated = cv2.rotate(img, cv2.ROTATE_180)
                new_boxes = [(w - (x + ww), h - (y + hh), ww, hh) for (x, y, ww, hh) in boxes]
                return rotated, new_boxes
            elif angle == -90:
                rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                new_boxes = [(y, w - (x + ww), hh, ww) for (x, y, ww, hh) in boxes]
                return rotated, new_boxes
            return img, boxes
        best_angle = 0
        for angle in [0, 90, 180, -90]:
            test_img, test_boxes = rotate_img_and_boxes(img, [sig_cand['rect'], instr_cand['rect']], angle)
            (sx, sy, sww, shh), (ix, iy, iww, ihh) = test_boxes
            # Both boxes should be near the top (y < 0.25*h), and sig left of instr (sx < ix)
            if sy < 0.25 * h and iy < 0.25 * h and sx < ix:
                best_angle = angle
                break
        final_img, final_boxes = rotate_img_and_boxes(img, [sig_cand['rect'], instr_cand['rect']], best_angle)
        if debug_out is not None:
            debug_out['final_signature_box'] = final_boxes[0]
            debug_out['final_instructions_box'] = final_boxes[1]
            debug_out['final_orientation_angle'] = best_angle
        return final_img, best_angle, final_boxes
import cv2
import numpy as np


def _order_points(pts):
    # initial ordering of points: tl, tr, br, bl
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image, pts, dst_size=(1000, 1400)):
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB), dst_size[0])

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB), dst_size[1])

    dst = np.array([
        [0, 0],
        [dst_size[0] - 1, 0],
        [dst_size[0] - 1, dst_size[1] - 1],
        [0, dst_size[1] - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, dst_size)
    return warped


def preprocess_image(image_path, target_size=(1000, 1400), auto_rotate=True, fix_orientation_by_instructions=True):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found at " + image_path)

    orig = image.copy()
    # Step 0: fix orientation using instructions box if requested
    if fix_orientation_by_instructions:
        orig, instr_angle, instr_boxes = _find_signature_and_instructions_boxes_and_orient(orig, debug_out=None)

    # Robust deskew: detect dominant line angle (up to ±45°) and rotate
    def _robust_deskew(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 120)
        if lines is None:
            return img
        angles = []
        for l in lines:
            rho, theta = l[0]
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)
        if not angles:
            return img
        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:
            return img
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), -median_angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    try:
        orig = _robust_deskew(orig)
    except Exception:
        orig = orig
    h0, w0 = orig.shape[:2]

    # Auto-rotate by 90 degrees if the detected page appears sideways.
    def _rotate90_if_needed(img, target_size):
        # small fast detection to decide orientation
        ih, iw = img.shape[:2]
        scale_tmp = 800.0 / float(iw) if iw > 800 else 1.0
        if scale_tmp != 1.0:
            tmp = cv2.resize(img, (int(iw * scale_tmp), int(ih * scale_tmp)))
        else:
            tmp = img.copy()
        gray_tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        blur_tmp = cv2.GaussianBlur(gray_tmp, (5, 5), 0)
        edged_tmp = cv2.Canny(blur_tmp, 50, 150)
        cnts_tmp, _ = cv2.findContours(edged_tmp.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        page_w, page_h = None, None
        if cnts_tmp:
            cnts_sorted = sorted(cnts_tmp, key=cv2.contourArea, reverse=True)
            for c in cnts_sorted:
                area = cv2.contourArea(c)
                if area < 0.10 * (tmp.shape[0] * tmp.shape[1]):
                    break
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) >= 4:
                    x, y, ww, hh = cv2.boundingRect(approx)
                    # scale back to original reference
                    page_w = int(ww / scale_tmp)
                    page_h = int(hh / scale_tmp)
                    break
        # if we didn't find contour, use image shape
        if page_w is None or page_h is None:
            page_w, page_h = img.shape[1], img.shape[0]

        tgt_w, tgt_h = target_size[0], target_size[1]
        # decide rotation: if target is portrait but detected page is landscape -> rotate
        if tgt_h > tgt_w and page_w > page_h:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 90
        if tgt_w > tgt_h and page_h > page_w:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), -90
        return img, None

    try:
        if auto_rotate:
            orig, auto_rot = _rotate90_if_needed(orig, target_size)
        else:
            auto_rot = None
    except Exception:
        auto_rot = None

    # Resize a bit for faster processing while keeping aspect
    scale = 1000.0 / float(w0) if w0 > 1000 else 1.0
    if scale != 1.0:
        image_small = cv2.resize(image, (int(w0 * scale), int(h0 * scale)))
    else:
        image_small = image.copy()

    gray_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_small, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # find contours and look for the largest 4-point contour
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pageCnt = None
    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        resized_area = float(image_small.shape[0] * image_small.shape[1])
        for c in cnts:
            area = cv2.contourArea(c)
            # skip small contours (likely signature boxes or small decorations)
            if area < 0.20 * resized_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                pageCnt = approx.reshape(4, 2).astype('float32')
                break

    if pageCnt is not None:
        # scale points back to original size if we resized
        if scale != 1.0:
            pageCnt = pageCnt / scale
        # perform perspective transform to fixed target size
        warped = _four_point_transform(orig, pageCnt, dst_size=target_size)
        image = warped
    else:
        # fallback: center-crop or resize to target_size
        image = cv2.resize(orig, target_size)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return image, gray, thresh


def preprocess_debug(image_path, target_size=(1000, 1400), auto_rotate=True, fix_orientation_by_instructions=True):
    """Run preprocessing and return intermediate images for debugging.

    Returns a dict with keys:
      original, deskewed, small_resized, gray_small, edged, page_mask, warped, gray, blurred, thresh
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found at " + image_path)
    out = {}
    orig = image.copy()
    # Step 0: fix orientation using instructions box if requested
    if fix_orientation_by_instructions:
        orig, instr_angle, instr_boxes = _find_signature_and_instructions_boxes_and_orient(orig, debug_out=out)
        out["signature_box"] = instr_boxes[0] if len(instr_boxes) > 0 else None
        out["instructions_box"] = instr_boxes[1] if len(instr_boxes) > 1 else None
        out["instructions_angle"] = instr_angle
    out["original"] = orig.copy()

    # Robust deskew (debug): detect dominant line angle (up to ±45°) and rotate
    def _robust_deskew_debug(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 120)
        if lines is None:
            return img, edges, None
        angles = []
        for l in lines:
            rho, theta = l[0]
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)
        if not angles:
            return img, edges, None
        median_angle = float(np.median(angles))
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), -median_angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated, edges, median_angle

    deskewed, edges, angle = _robust_deskew_debug(out["original"])
    out["deskewed"] = deskewed
    out["edges_after_deskew"] = edges
    out["deskew_angle"] = angle

    # Auto-rotate if necessary (90-degree correction)
    def _rotate90_decision(img, target_size):
        ih, iw = img.shape[:2]
        scale_tmp = 800.0 / float(iw) if iw > 800 else 1.0
        if scale_tmp != 1.0:
            tmp = cv2.resize(img, (int(iw * scale_tmp), int(ih * scale_tmp)))
        else:
            tmp = img.copy()
        gray_tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        blur_tmp = cv2.GaussianBlur(gray_tmp, (5, 5), 0)
        edged_tmp = cv2.Canny(blur_tmp, 50, 150)
        cnts_tmp, _ = cv2.findContours(edged_tmp.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        page_w, page_h = None, None
        if cnts_tmp:
            cnts_sorted = sorted(cnts_tmp, key=cv2.contourArea, reverse=True)
            for c in cnts_sorted:
                area = cv2.contourArea(c)
                if area < 0.10 * (tmp.shape[0] * tmp.shape[1]):
                    break
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) >= 4:
                    x, y, ww, hh = cv2.boundingRect(approx)
                    page_w = int(ww / scale_tmp)
                    page_h = int(hh / scale_tmp)
                    break
        if page_w is None or page_h is None:
            page_w, page_h = img.shape[1], img.shape[0]
        tgt_w, tgt_h = target_size[0], target_size[1]
        if tgt_h > tgt_w and page_w > page_h:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 90
        if tgt_w > tgt_h and page_h > page_w:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), -90
        return img, None

    warped_input = out["deskewed"]
    try:
        if auto_rotate:
            warped_input, auto_rot = _rotate90_decision(warped_input, target_size)
        else:
            auto_rot = None
    except Exception:
        auto_rot = None
    out["auto_rotated"] = auto_rot

    h0, w0 = image.shape[:2]
    scale = 1000.0 / float(w0) if w0 > 1000 else 1.0
    if scale != 1.0:
        image_small = cv2.resize(deskewed, (int(w0 * scale), int(h0 * scale)))
    else:
        image_small = deskewed.copy()
    out["small_resized"] = image_small

    gray_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
    blurred_small = cv2.GaussianBlur(gray_small, (5, 5), 0)
    edged = cv2.Canny(blurred_small, 50, 150)
    out["gray_small"] = gray_small
    out["edged_small"] = edged

    # find contours and page mask
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    page_mask = np.zeros_like(gray_small)
    pageCnt = None
    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        resized_area = float(image_small.shape[0] * image_small.shape[1])
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 0.20 * resized_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                pageCnt = approx.reshape(4, 2).astype('float32')
                cv2.drawContours(page_mask, [c], -1, 255, -1)
                break
    out["page_mask"] = page_mask

    if pageCnt is not None:
        if scale != 1.0:
            pageCnt = pageCnt / scale
        warped = _four_point_transform(warped_input, pageCnt, dst_size=target_size)
    else:
        warped = cv2.resize(warped_input, target_size)
    out["warped"] = warped

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    out["gray"] = gray
    out["blurred"] = blurred
    out["thresh"] = thresh
    return out
