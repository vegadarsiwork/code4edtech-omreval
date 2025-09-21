from omr.preprocess import preprocess_image, preprocess_debug
from omr.detect_bubbles import detect_bubbles
import cv2
from pathlib import Path
import os

# ---------- Resolve Image Path ----------
BASE = Path(__file__).parent
expected_name = "Img6.jpeg"
rel_path = Path("data") / "uploads" / expected_name
img_path = (BASE / rel_path).resolve()
print("Resolved image path:", img_path)

# Case-insensitive fallback
if not img_path.exists():
    uploads_dir = (BASE / "data" / "uploads")
    if uploads_dir.exists():
        matches = [p for p in uploads_dir.iterdir() if p.name.lower() == expected_name.lower()]
        if matches:
            img_path = matches[0]
            print("Found case-insensitive match:", img_path)
        else:
            print("Uploads directory contents:", [p.name for p in uploads_dir.iterdir()])
            raise FileNotFoundError(f"Image not found at {img_path}")
    else:
        raise FileNotFoundError(f"Uploads directory not found at {uploads_dir}")

# ---------- Step 1: Preprocess & debug steps ----------
# Use instructions box to fix orientation (default True)
debug_steps = preprocess_debug(str(img_path), auto_rotate=False, fix_orientation_by_instructions=True)

# Print instructions box/angle if found
if "instructions_box" in debug_steps:
    print("Instructions box (top right after correction):", debug_steps["instructions_box"])
    print("Instructions angle:", debug_steps["instructions_angle"])

# create results dir
results_dir = Path(__file__).parent / ".." / "results" / "preprocessing"
results_dir = results_dir.resolve()
results_dir.mkdir(parents=True, exist_ok=True)

# save each debugging image (skip None values)
for k, v in debug_steps.items():
    if v is None:
        continue
    out_path = results_dir / f"step_{k}.png"
    # v may be grayscale or color; convert if needed
    # ensure v is an image (numpy array)
    import numpy as _np
    if not isinstance(v, _np.ndarray):
        # skip non-image debug values (e.g. floats like deskew_angle)
        continue
    if v.ndim == 2:
        cv2.imwrite(str(out_path), v)
    elif v.ndim == 3:
        # convert BGR->RGB if needed when saving? OpenCV expects BGR so save directly
        cv2.imwrite(str(out_path), v)
    else:
        # unexpected shape, skip
        continue


# main outputs
original = debug_steps["original"]
gray = debug_steps["gray"]
thresh = debug_steps["thresh"]

# If answer box is detected, crop all images to that region
answer_box = None
if "answers_box" in debug_steps and debug_steps["answers_box"] is not None:
    answer_box = debug_steps["answers_box"]
elif "answers_box" in debug_steps and debug_steps["answers_box"] is None and "instructions_box" in debug_steps:
    # fallback: use instructions_box if available
    answer_box = debug_steps["instructions_box"]

if answer_box is not None:
    ax, ay, aw, ah = answer_box
    original = original[ay:ay+ah, ax:ax+aw]
    gray = gray[ay:ay+ah, ax:ax+aw]
    thresh = thresh[ay:ay+ah, ax:ax+aw]

# ---------- Step 2: Detect All Bubbles (new API) ----------
# set use_answers_box=False to run detection over the entire image and keep other parts
# Use True to filter to the computed answers area (recommended for structured sheets)
use_answers_box = True
# For 100 questions with 4 options each, arranged in 5 columns of 20 questions
result = detect_bubbles(gray, expected_grid=(100, 4), debug=True, use_answers_box=use_answers_box)
print("Detected bubbles:", result["count"])


# --- Simple answer classification ---
import numpy as np
def detect_filled_bubbles(grid, thresh, roi_radius=10, fill_ratio=0.5):
    filled = []
    for row in grid:
        row_filled = []
        for cell in row:
            if cell is None:
                row_filled.append(False)
                continue
            x, y = map(int, cell)
            x0, x1 = max(0, x-roi_radius), min(thresh.shape[1], x+roi_radius)
            y0, y1 = max(0, y-roi_radius), min(thresh.shape[0], y+roi_radius)
            roi = thresh[y0:y1, x0:x1]
            if roi.size == 0:
                row_filled.append(False)
                continue
            fill = np.mean(roi < 128)
            row_filled.append(fill > fill_ratio)
        filled.append(row_filled)
    return filled


# Enhanced answer detection with fill ratio info
def detect_filled_bubbles_with_debug(grid, thresh, roi_radius=10, fill_ratio=0.5):
    filled = []
    debug_info = []
    for row_idx, row in enumerate(grid):
        row_filled = []
        row_debug = []
        for col_idx, cell in enumerate(row):
            if cell is None:
                row_filled.append(False)
                row_debug.append({'fill_pct': 0, 'status': 'missing'})
                continue
            x, y = map(int, cell)
            x0, x1 = max(0, x-roi_radius), min(thresh.shape[1], x+roi_radius)
            y0, y1 = max(0, y-roi_radius), min(thresh.shape[0], y+roi_radius)
            roi = thresh[y0:y1, x0:x1]
            if roi.size == 0:
                row_filled.append(False)
                row_debug.append({'fill_pct': 0, 'status': 'invalid_roi'})
                continue
            fill_pct = np.mean(roi < 128) * 100
            is_filled = fill_pct > (fill_ratio * 100)
            row_filled.append(is_filled)
            status = 'filled' if is_filled else 'empty'
            if 20 < fill_pct < 80:  # Edge case: partially filled
                status = 'partial'
            row_debug.append({'fill_pct': fill_pct, 'status': status})
        filled.append(row_filled)
        debug_info.append(row_debug)
    return filled, debug_info

def get_definitive_answers(filled_answers, debug_info, confidence_threshold=60):
    """
    Extract definitive answers with confidence scoring.
    Returns clean final answers suitable for Excel export.
    """
    options = ['A', 'B', 'C', 'D']
    results = []
    
    for q_idx, (row, debug_row) in enumerate(zip(filled_answers, debug_info), 1):
        question_data = {
            'Question': q_idx,
            'Answer': None,
            'Confidence': 'Low',
            'Status': 'Clear',
            'Fill_Percentages': {},
            'Multiple_Marks': False,
            'Raw_Data': []
        }
        
        # Collect all fill percentages and determine answers
        marked_options = []
        high_confidence_marks = []
        
        for i, (filled, debug) in enumerate(zip(row, debug_row)):
            if i >= len(options):
                break
                
            option = options[i]
            fill_pct = debug['fill_pct']
            question_data['Fill_Percentages'][option] = fill_pct
            question_data['Raw_Data'].append(f"{option}:{fill_pct:.1f}%")
            
            if filled:  # Above 50% threshold
                marked_options.append(option)
                if fill_pct >= confidence_threshold:
                    high_confidence_marks.append(option)
        
        # Determine final answer based on analysis
        if len(high_confidence_marks) == 1:
            # Single high-confidence answer - BEST case
            question_data['Answer'] = high_confidence_marks[0]
            question_data['Confidence'] = 'High'
            question_data['Status'] = 'Clear'
        elif len(high_confidence_marks) > 1:
            # Multiple high-confidence marks - take highest fill %
            best_option = max(high_confidence_marks, 
                            key=lambda opt: question_data['Fill_Percentages'][opt])
            question_data['Answer'] = best_option
            question_data['Confidence'] = 'Medium'
            question_data['Status'] = 'Multiple_High_Confidence'
            question_data['Multiple_Marks'] = True
        elif len(marked_options) == 1:
            # Single low-confidence answer
            question_data['Answer'] = marked_options[0]
            question_data['Confidence'] = 'Medium'
            question_data['Status'] = 'Low_Confidence'
        elif len(marked_options) > 1:
            # Multiple low-confidence marks - take highest
            best_option = max(marked_options, 
                            key=lambda opt: question_data['Fill_Percentages'][opt])
            question_data['Answer'] = best_option
            question_data['Confidence'] = 'Low'
            question_data['Status'] = 'Multiple_Low_Confidence'
            question_data['Multiple_Marks'] = True
        else:
            # No clear answer
            # Check if any option has >40% (might be lightly filled)
            potential = [(opt, pct) for opt, pct in question_data['Fill_Percentages'].items() 
                        if pct > 40]
            if potential:
                best_option = max(potential, key=lambda x: x[1])[0]
                question_data['Answer'] = best_option
                question_data['Confidence'] = 'Low'
                question_data['Status'] = 'Weak_Signal'
            else:
                question_data['Answer'] = None
                question_data['Confidence'] = 'None'
                question_data['Status'] = 'No_Answer'
        
        results.append(question_data)
    
    return results

# Create visualization of filled vs empty bubbles
def create_answer_visualization(grid, filled_answers, debug_info, gray):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
            if cell is None:
                continue
            x, y = map(int, cell)
            
            # Get debug info
            if row_idx < len(debug_info) and col_idx < len(debug_info[row_idx]):
                info = debug_info[row_idx][col_idx]
                fill_pct = info['fill_pct']
                status = info['status']
                
                # Color coding based on status
                if status == 'filled':
                    color = (0, 255, 0)  # Green for filled
                elif status == 'partial':
                    color = (0, 165, 255)  # Orange for partial
                elif status == 'empty':
                    color = (0, 0, 255)  # Red for empty
                else:
                    color = (128, 128, 128)  # Gray for missing/invalid
                
                # Draw circle and fill percentage
                cv2.circle(vis, (x, y), 12, color, 2)
                cv2.putText(vis, f"{fill_pct:.0f}%", (x-15, y-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Mark question number for first option of each question
                if col_idx == 0:
                    cv2.putText(vis, f"Q{row_idx+1}", (x-30, y+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis

filled_answers, debug_info = detect_filled_bubbles_with_debug(result["grid"], result["thresh"])

# Create and save detailed visualization
answer_vis = create_answer_visualization(result["grid"], filled_answers, debug_info, gray)
answer_vis_path = os.path.join(Path(__file__).parent, "..", "results", "filled_answers_visualization.png")
answer_vis_path = str(Path(answer_vis_path).resolve())
cv2.imwrite(answer_vis_path, answer_vis)
print(f"Answer visualization saved to: {answer_vis_path}")

# Get definitive answers with confidence analysis
definitive_results = get_definitive_answers(filled_answers, debug_info, confidence_threshold=60)

# Create Excel export with comprehensive data
import pandas as pd

# Prepare data for Excel
excel_data = []
for result in definitive_results:
    excel_row = {
        'Question_Number': result['Question'],
        'Final_Answer': result['Answer'] if result['Answer'] else 'NO_ANSWER',
        'Confidence_Level': result['Confidence'],
        'Status': result['Status'],
        'Multiple_Marks_Detected': 'Yes' if result['Multiple_Marks'] else 'No',
        'A_Fill_Percent': f"{result['Fill_Percentages'].get('A', 0):.1f}%",
        'B_Fill_Percent': f"{result['Fill_Percentages'].get('B', 0):.1f}%",
        'C_Fill_Percent': f"{result['Fill_Percentages'].get('C', 0):.1f}%",
        'D_Fill_Percent': f"{result['Fill_Percentages'].get('D', 0):.1f}%",
        'Raw_Detection_Data': ' | '.join(result['Raw_Data'])
    }
    excel_data.append(excel_row)

# Create DataFrame and save to Excel
df = pd.DataFrame(excel_data)
excel_path = os.path.join(Path(__file__).parent, "..", "results", "omr_answers.xlsx")
excel_path = str(Path(excel_path).resolve())

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # Main results sheet
    df.to_excel(writer, sheet_name='OMR_Results', index=False)
    
    # Summary statistics sheet
    summary_data = {
        'Total_Questions': len(definitive_results),
        'High_Confidence_Answers': len([r for r in definitive_results if r['Confidence'] == 'High']),
        'Medium_Confidence_Answers': len([r for r in definitive_results if r['Confidence'] == 'Medium']),
        'Low_Confidence_Answers': len([r for r in definitive_results if r['Confidence'] == 'Low']),
        'No_Answer_Detected': len([r for r in definitive_results if r['Answer'] is None]),
        'Multiple_Marks_Questions': len([r for r in definitive_results if r['Multiple_Marks']]),
        'Clear_Single_Answers': len([r for r in definitive_results if r['Status'] == 'Clear']),
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_excel(writer, sheet_name='Summary_Stats', index=False)
    
    # Clean answers only (for frontend use)
    clean_data = []
    for result in definitive_results:
        clean_data.append({
            'Question': result['Question'],
            'Answer': result['Answer'] if result['Answer'] else '',
            'Confidence': result['Confidence']
        })
    
    clean_df = pd.DataFrame(clean_data)
    clean_df.to_excel(writer, sheet_name='Clean_Answers', index=False)

print(f"✅ Excel file created: {excel_path}")

# Print summary for console
print(f"\n=== DEFINITIVE RESULTS SUMMARY ===")
print(f"Total Questions: {summary_data['Total_Questions']}")
print(f"High Confidence: {summary_data['High_Confidence_Answers']}")
print(f"Medium Confidence: {summary_data['Medium_Confidence_Answers']}")
print(f"Low Confidence: {summary_data['Low_Confidence_Answers']}")
print(f"No Answer: {summary_data['No_Answer_Detected']}")
print(f"Multiple Marks: {summary_data['Multiple_Marks_Questions']}")

# Show first 10 definitive answers
print(f"\n=== FIRST 10 DEFINITIVE ANSWERS ===")
for i, result in enumerate(definitive_results[:10]):
    answer = result['Answer'] if result['Answer'] else 'NO_ANSWER'
    confidence = result['Confidence']
    status = result['Status']
    print(f"Q{result['Question']}: {answer} ({confidence} confidence, {status})")

print(f"\nSaved detailed visualization to: {answer_vis_path}")
print("Green = Filled, Red = Empty, Orange = Partial, Gray = Missing/Invalid")
print(f"\n📊 Excel file ready for Streamlit frontend: {excel_path}")

# Save bubble detection visualization if available
if "visualization" in result:
    out_vis = os.path.join(Path(__file__).parent, "..", "results", "detected_bubbles.png")
    out_vis = str(Path(out_vis).resolve())
    cv2.imwrite(out_vis, result["visualization"])
    print("Bubble detection visualization written to:", out_vis)
