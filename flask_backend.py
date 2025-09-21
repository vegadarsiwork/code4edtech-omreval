from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import uuid
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import zipfile
import io

# Import OMR processing modules
from omr_eval.omr.preprocess import preprocess_image
from omr_eval.omr.detect_bubbles import detect_bubbles

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
RESULTS_FOLDER = 'temp_results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

class OMRProcessor:
    def __init__(self):
        self.options = ['A', 'B', 'C', 'D']
    
    def preprocess_image_array(self, image, target_size=(1000, 1400)):
        """
        Simplified preprocessing for image arrays (not file paths)
        This is a streamlined version for the Flask API
        """
        orig = image.copy()
        
        # Convert to grayscale if needed
        if len(orig.shape) == 3:
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        else:
            gray = orig.copy()
        
        # Basic deskewing
        try:
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 120)
            
            if lines is not None:
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = np.degrees(theta) - 90
                    if abs(angle) < 45:
                        angles.append(angle)
                
                if angles:
                    median_angle = float(np.median(angles))
                    if abs(median_angle) > 0.5:
                        (h, w) = orig.shape[:2]
                        M = cv2.getRotationMatrix2D((w // 2, h // 2), -median_angle, 1.0)
                        orig = cv2.warpAffine(orig, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            pass  # Continue without deskewing if it fails
        
        # Resize to target size
        warped = cv2.resize(orig, target_size)
        
        # Final processing
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return {
            "original": orig,
            "warped": warped,
            "gray": gray,
            "blurred": blurred,
            "thresh": thresh
        }
    
    def detect_filled_bubbles_with_debug(self, grid, thresh, roi_radius=10, fill_ratio=0.5):
        """Enhanced answer detection with fill ratio info"""
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

    def get_definitive_answers(self, filled_answers, debug_info, confidence_threshold=60):
        """Extract definitive answers with confidence scoring"""
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
                if i >= len(self.options):
                    break
                    
                option = self.options[i]
                fill_pct = debug['fill_pct']
                question_data['Fill_Percentages'][option] = fill_pct
                question_data['Raw_Data'].append(f"{option}:{fill_pct:.1f}%")
                
                if filled:  # Above 50% threshold
                    marked_options.append(option)
                    if fill_pct >= confidence_threshold:
                        high_confidence_marks.append(option)
            
            # Determine final answer based on analysis
            if len(high_confidence_marks) == 1:
                question_data['Answer'] = high_confidence_marks[0]
                question_data['Confidence'] = 'High'
                question_data['Status'] = 'Clear'
            elif len(high_confidence_marks) > 1:
                best_option = max(high_confidence_marks, 
                                key=lambda opt: question_data['Fill_Percentages'][opt])
                question_data['Answer'] = best_option
                question_data['Confidence'] = 'Medium'
                question_data['Status'] = 'Multiple_High_Confidence'
                question_data['Multiple_Marks'] = True
            elif len(marked_options) == 1:
                question_data['Answer'] = marked_options[0]
                question_data['Confidence'] = 'Medium'
                question_data['Status'] = 'Low_Confidence'
            elif len(marked_options) > 1:
                best_option = max(marked_options, 
                                key=lambda opt: question_data['Fill_Percentages'][opt])
                question_data['Answer'] = best_option
                question_data['Confidence'] = 'Low'
                question_data['Status'] = 'Multiple_Low_Confidence'
                question_data['Multiple_Marks'] = True
            else:
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

    def create_answer_visualization(self, grid, filled_answers, debug_info, gray):
        """Create visualization of filled vs empty bubbles"""
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

    def load_answer_key(self):
        """Load answer key from data/uploads/sample.xlsx"""
        try:
            answer_key_path = os.path.join("omr_eval", "data", "sample.xlsx")
            if os.path.exists(answer_key_path):
                df = pd.read_excel(answer_key_path)
                
                # Try different possible column names for questions and answers
                question_col = None
                answer_col = None
                
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'question' in col_lower or 'q' in col_lower or 'number' in col_lower:
                        question_col = col
                    elif 'answer' in col_lower or 'correct' in col_lower or 'key' in col_lower:
                        answer_col = col
                
                if question_col is None or answer_col is None:
                    # Try first two columns as fallback
                    if len(df.columns) >= 2:
                        question_col = df.columns[0]
                        answer_col = df.columns[1]
                    else:
                        return None
                
                # Create answer key dictionary
                answer_key = {}
                for _, row in df.iterrows():
                    q_num = int(row[question_col])
                    answer = str(row[answer_col]).upper().strip()
                    if answer in ['A', 'B', 'C', 'D']:
                        answer_key[q_num] = answer
                
                print(f"DEBUG: Loaded answer key with {len(answer_key)} questions")
                return answer_key
            else:
                print("DEBUG: No answer key found at", answer_key_path)
                return None
        except Exception as e:
            print(f"DEBUG: Error loading answer key: {e}")
            return None

    def calculate_score_if_answer_key_exists(self, definitive_results):
        """Calculate score against answer key if available"""
        answer_key = self.load_answer_key()
        if not answer_key:
            return None
        
        correct = 0
        total = 0
        detailed_comparison = []
        
        for result in definitive_results:
            q_num = result['Question']
            student_answer = result['Answer']
            
            if q_num in answer_key:
                correct_answer = answer_key[q_num]
                is_correct = student_answer == correct_answer
                
                if student_answer is not None:  # Only count answered questions
                    total += 1
                    if is_correct:
                        correct += 1
                
                detailed_comparison.append({
                    'Question': q_num,
                    'Student_Answer': student_answer if student_answer else 'NO_ANSWER',
                    'Correct_Answer': correct_answer,
                    'Is_Correct': is_correct,
                    'Confidence': result['Confidence'],
                    'Status': result['Status']
                })
        
        if total == 0:
            return None
        
        score_percentage = (correct / total) * 100
        
        return {
            'scoring_available': True,
            'total_answered': total,
            'total_correct': correct,
            'score_percentage': round(score_percentage, 2),
            'grade': self.get_letter_grade(score_percentage),
            'answer_key_questions': len(answer_key),
            'detailed_comparison': detailed_comparison
        }

    def get_letter_grade(self, percentage):
        """Convert percentage to letter grade"""
        if percentage >= 90:
            return 'A'
        elif percentage >= 80:
            return 'B'
        elif percentage >= 70:
            return 'C'
        elif percentage >= 60:
            return 'D'
        else:
            return 'F'

    def process_omr_image(self, image_data, job_id, filename="uploaded_image"):
        """Complete OMR processing pipeline"""
        try:
            # Handle different input types
            if isinstance(image_data, str):
                # If it's a file path
                img = cv2.imread(image_data)
                if img is None:
                    raise ValueError(f"Could not load image: {image_data}")
            else:
                # If it's bytes/buffer data (from file upload)
                # Convert bytes to numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                # Decode image from numpy array
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Could not decode image data for: {filename}")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Preprocessing - create a custom preprocessing pipeline for arrays
            result = self.preprocess_image_array(img)
            if result is None:
                raise ValueError("Preprocessing failed")
            
            # Detect bubbles
            bubble_result = detect_bubbles(result["thresh"], expected_grid=(100, 4))
            if bubble_result is None or "grid" not in bubble_result:
                raise ValueError("Bubble detection failed")
            
            # Debug: Check bubble detection
            print(f"DEBUG: Detected {len([cell for row in bubble_result['grid'] for cell in row if cell is not None])} bubble positions")
            print(f"DEBUG: Grid shape: {len(bubble_result['grid'])} rows x {len(bubble_result['grid'][0])} cols")
            
            # Merge results
            result.update(bubble_result)
            
            # Answer detection with debug info
            filled_answers, debug_info = self.detect_filled_bubbles_with_debug(result["grid"], result["thresh"])
            
            # Debug: Check answer detection
            total_filled = sum(sum(row) for row in filled_answers)
            print(f"DEBUG: Found {total_filled} filled bubbles across all questions")
            
            # Show sample fill percentages from first few questions
            for i in range(min(3, len(debug_info))):
                percentages = [f"{info['fill_pct']:.1f}%" for info in debug_info[i][:4]]
                print(f"DEBUG: Q{i+1} fill percentages: {percentages}")
            
            # Let's also try with lower threshold
            filled_answers_low, debug_info_low = self.detect_filled_bubbles_with_debug(result["grid"], result["thresh"], fill_ratio=0.3)
            total_filled_low = sum(sum(row) for row in filled_answers_low)
            print(f"DEBUG: With 30% threshold: {total_filled_low} filled bubbles")
            
            # Use the lower threshold results if they seem more reasonable
            if total_filled < 20 and total_filled_low > total_filled:
                print("DEBUG: Using lower threshold results")
                filled_answers, debug_info = filled_answers_low, debug_info_low
            
            # Get definitive answers
            definitive_results = self.get_definitive_answers(filled_answers, debug_info, confidence_threshold=45)
            
            # Create visualizations
            answer_vis = self.create_answer_visualization(result["grid"], filled_answers, debug_info, gray)
            
            # Save results
            results_dir = os.path.join(RESULTS_FOLDER, job_id)
            os.makedirs(results_dir, exist_ok=True)
            
            # Save visualization
            vis_path = os.path.join(results_dir, "answer_visualization.png")
            cv2.imwrite(vis_path, answer_vis)
            
            # Create Excel file
            excel_data = []
            for result_item in definitive_results:
                excel_row = {
                    'Question_Number': result_item['Question'],
                    'Final_Answer': result_item['Answer'] if result_item['Answer'] else 'NO_ANSWER',
                    'Confidence_Level': result_item['Confidence'],
                    'Status': result_item['Status'],
                    'Multiple_Marks_Detected': 'Yes' if result_item['Multiple_Marks'] else 'No',
                    'A_Fill_Percent': f"{result_item['Fill_Percentages'].get('A', 0):.1f}%",
                    'B_Fill_Percent': f"{result_item['Fill_Percentages'].get('B', 0):.1f}%",
                    'C_Fill_Percent': f"{result_item['Fill_Percentages'].get('C', 0):.1f}%",
                    'D_Fill_Percent': f"{result_item['Fill_Percentages'].get('D', 0):.1f}%",
                    'Raw_Detection_Data': ' | '.join(result_item['Raw_Data'])
                }
                excel_data.append(excel_row)
            
            df = pd.DataFrame(excel_data)
            excel_path = os.path.join(results_dir, "omr_results.xlsx")
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Main results sheet
                df.to_excel(writer, sheet_name='OMR_Results', index=False)
                
                # Summary statistics
                summary_data = {
                    'Total_Questions': len(definitive_results),
                    'High_Confidence_Answers': len([r for r in definitive_results if r['Confidence'] == 'High']),
                    'Medium_Confidence_Answers': len([r for r in definitive_results if r['Confidence'] == 'Medium']),
                    'Low_Confidence_Answers': len([r for r in definitive_results if r['Confidence'] == 'Low']),
                    'No_Answer_Detected': len([r for r in definitive_results if r['Answer'] is None]),
                    'Multiple_Marks_Questions': len([r for r in definitive_results if r['Multiple_Marks']]),
                    'Processing_Timestamp': datetime.now().isoformat(),
                    'Image_Processed': filename
                }
                
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_excel(writer, sheet_name='Summary_Stats', index=False)
                
                # Clean answers only
                clean_data = []
                for result_item in definitive_results:
                    clean_data.append({
                        'Question': result_item['Question'],
                        'Answer': result_item['Answer'] if result_item['Answer'] else '',
                        'Confidence': result_item['Confidence']
                    })
                
                clean_df = pd.DataFrame(clean_data)
                clean_df.to_excel(writer, sheet_name='Clean_Answers', index=False)
            
            # Check for answer key and calculate score if available
            score_data = self.calculate_score_if_answer_key_exists(definitive_results)
            
            # Add scoring sheet to Excel if available
            if score_data:
                comparison_df = pd.DataFrame(score_data['detailed_comparison'])
                comparison_df.to_excel(writer, sheet_name='Score_Comparison', index=False)
                
                # Add summary to the summary_data
                summary_data.update({
                    'Scoring_Available': True,
                    'Total_Answered': score_data['total_answered'],
                    'Total_Correct': score_data['total_correct'],
                    'Score_Percentage': score_data['score_percentage'],
                    'Letter_Grade': score_data['grade']
                })
            else:
                summary_data['Scoring_Available'] = False
            
            # Create processing summary
            processing_summary = {
                'job_id': job_id,
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'total_questions': len(definitive_results),
                'high_confidence': summary_data['High_Confidence_Answers'],
                'medium_confidence': summary_data['Medium_Confidence_Answers'], 
                'low_confidence': summary_data['Low_Confidence_Answers'],
                'no_answer': summary_data['No_Answer_Detected'],
                'multiple_marks': summary_data['Multiple_Marks_Questions'],
                'files_generated': {
                    'excel': 'omr_results.xlsx',
                    'visualization': 'answer_visualization.png'
                }
            }
            
            # Add score data if available
            if score_data:
                processing_summary.update(score_data)
            
            return processing_summary, definitive_results
            
        except Exception as e:
            error_summary = {
                'job_id': job_id,
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            return error_summary, None

# Initialize processor
processor = OMRProcessor()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'OMR Processing API',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/process', methods=['POST'])
def process_omr():
    """Process OMR image and return results"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Get file data directly (no need to save to disk first)
        file_data = file.read()
        filename = file.filename
        
        # Process the image directly from memory
        processing_summary, detailed_results = processor.process_omr_image(file_data, job_id, filename)
        
        if processing_summary['status'] == 'error':
            return jsonify(processing_summary), 500
        
        # Return processing summary
        return jsonify(processing_summary)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/results/<job_id>/excel', methods=['GET'])
def download_excel(job_id):
    """Download Excel results file"""
    try:
        excel_path = os.path.join(RESULTS_FOLDER, job_id, 'omr_results.xlsx')
        if not os.path.exists(excel_path):
            return jsonify({'error': 'Results file not found'}), 404
        
        return send_file(
            excel_path,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'omr_results_{job_id}.xlsx'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<job_id>/visualization', methods=['GET'])
def download_visualization(job_id):
    """Download visualization image"""
    try:
        vis_path = os.path.join(RESULTS_FOLDER, job_id, 'answer_visualization.png')
        if not os.path.exists(vis_path):
            return jsonify({'error': 'Visualization file not found'}), 404
        
        return send_file(
            vis_path,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'omr_visualization_{job_id}.png'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<job_id>/package', methods=['GET'])
def download_package(job_id):
    """Download complete results package (Excel + Visualization)"""
    try:
        results_dir = os.path.join(RESULTS_FOLDER, job_id)
        if not os.path.exists(results_dir):
            return jsonify({'error': 'Results not found'}), 404
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add Excel file
            excel_path = os.path.join(results_dir, 'omr_results.xlsx')
            if os.path.exists(excel_path):
                zip_file.write(excel_path, 'omr_results.xlsx')
            
            # Add visualization
            vis_path = os.path.join(results_dir, 'answer_visualization.png')
            if os.path.exists(vis_path):
                zip_file.write(vis_path, 'answer_visualization.png')
        
        zip_buffer.seek(0)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'omr_results_{job_id}.zip'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    """Get job processing status"""
    try:
        results_dir = os.path.join(RESULTS_FOLDER, job_id)
        if os.path.exists(results_dir):
            excel_exists = os.path.exists(os.path.join(results_dir, 'omr_results.xlsx'))
            vis_exists = os.path.exists(os.path.join(results_dir, 'answer_visualization.png'))
            
            return jsonify({
                'job_id': job_id,
                'status': 'completed' if excel_exists else 'processing',
                'files_ready': {
                    'excel': excel_exists,
                    'visualization': vis_exists
                }
            })
        else:
            return jsonify({
                'job_id': job_id,
                'status': 'not_found'
            }), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/answer-key/upload', methods=['POST'])
def upload_answer_key():
    """Upload answer key file"""
    try:
        if 'answer_key' not in request.files:
            return jsonify({'error': 'No answer key file provided'}), 400
        
        file = request.files['answer_key']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save the answer key
        answer_key_path = os.path.join("omr_eval", "data", "sample.xlsx")
        os.makedirs(os.path.dirname(answer_key_path), exist_ok=True)
        file.save(answer_key_path)
        
        # Validate the answer key
        try:
            df = pd.read_excel(answer_key_path)
            question_count = len(df)
            
            return jsonify({
                'status': 'success',
                'message': 'Answer key uploaded successfully',
                'questions_loaded': question_count,
                'timestamp': datetime.now().isoformat()
            })
        
        except Exception as validation_error:
            return jsonify({
                'status': 'error',
                'error': f'Invalid answer key format: {str(validation_error)}'
            }), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/answer-key/status', methods=['GET'])
def get_answer_key_status():
    """Check if answer key is available"""
    try:
        answer_key_path = os.path.join("omr_eval", "data", "sample.xlsx")
        
        if os.path.exists(answer_key_path):
            df = pd.read_excel(answer_key_path)
            return jsonify({
                'available': True,
                'questions_count': len(df),
                'last_modified': datetime.fromtimestamp(os.path.getmtime(answer_key_path)).isoformat()
            })
        else:
            return jsonify({
                'available': False,
                'message': 'No answer key found'
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting OMR Processing API Server...")
    print("📋 Available endpoints:")
    print("  - POST /api/process - Upload and process OMR image")
    print("  - GET /api/results/<job_id>/excel - Download Excel results")
    print("  - GET /api/results/<job_id>/visualization - Download visualization")
    print("  - GET /api/results/<job_id>/package - Download complete package")
    print("  - GET /api/results/<job_id>/status - Check job status")
    print("  - POST /api/answer-key/upload - Upload answer key file")
    print("  - GET /api/answer-key/status - Check answer key availability")
    print("  - GET /api/health - Health check")
    print("\n🌐 Server running on: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
