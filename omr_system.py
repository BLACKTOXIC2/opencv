import cv2
import numpy as np
import argparse
import os
import sys

def extract_bubble_features(image):
    """Extract features from a bubble image for classification"""
    # Resize to standard size
    img_resized = cv2.resize(image, (32, 32))
    
    # Extract histogram features
    hist = cv2.calcHist([img_resized], [0], None, [16], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    # Extract some statistics
    mean = np.mean(img_resized)
    std = np.std(img_resized)
    
    # Calculate black pixel ratio
    black_pixels = np.sum(img_resized < 127)
    total_pixels = img_resized.size
    black_ratio = black_pixels / total_pixels
    
    # Calculate moments
    moments = cv2.moments(img_resized)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Combine features
    features = np.concatenate([hist, [mean, std, black_ratio], hu_moments])
    
    return features

def process_omr_sheet(image_path, answer_key=None, output_dir="./results", bubble_classifier=None, threshold_model=None):
    """
    Process an OMR sheet image and grade it against an optional answer key.
    
    Args:
        image_path (str): Path to the OMR sheet image
        answer_key (list, optional): List of correct answers (0=A, 1=B, 2=C, 3=D)
        output_dir (str, optional): Directory to save output images
        bubble_classifier (object, optional): Trained classifier model for bubble detection
        threshold_model (object, optional): Trained model for threshold determination
        
    Returns:
        dict: Results including marked answers and score
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image from {image_path}")
        return None
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Save thresholded image for debugging
    cv2.imwrite(os.path.join(output_dir, "debug_threshold.jpg"), thresh)
    
    # Use Hough Circle Transform to detect circles (bubbles)
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1,           # Inverse ratio of accumulator resolution
        minDist=20,     # Minimum distance between detected centers
        param1=50,      # Upper threshold for edge detection
        param2=15,      # Threshold for center detection (lower = more circles)
        minRadius=8,    # Minimum radius
        maxRadius=15    # Maximum radius
    )
    
    result_image = image.copy()
    
    # Process detected circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(f"Found {len(circles)} potential bubble contours")
        
        # Sort circles by y-coordinate (top to bottom) first
        circles = sorted(circles, key=lambda x: x[1])
        
        # Group circles by rows (questions)
        rows = []
        current_row = [circles[0]]
        y_threshold = 20  # Adjust as needed for your form
        
        for i in range(1, len(circles)):
            # If this circle is close to the y-coordinate of the previous row
            if abs(circles[i][1] - current_row[0][1]) < y_threshold:
                current_row.append(circles[i])
            else:
                # Sort the circles in the current row by x-coordinate
                current_row = sorted(current_row, key=lambda x: x[0])
                rows.append(current_row)
                current_row = [circles[i]]
                
        # Add the last row
        if current_row:
            current_row = sorted(current_row, key=lambda x: x[0])
            rows.append(current_row)
        
        # Filter rows that don't have exactly 4 circles (assuming 4 options)
        valid_rows = [row for row in rows if len(row) == 4]
        
        print(f"Found {len(valid_rows)} valid questions")
        
        # Check which bubbles are filled
        marked_answers = []
        answer_confidences = {}
        
        for row_idx, row in enumerate(valid_rows):
            max_filled_score = 0
            marked_option = None
            
            for i, (x, y, r) in enumerate(row):
                # Extract the bubble region
                bubble_roi = thresh[y-r:y+r, x-r:x+r]
                
                # Calculate the percentage of filled pixels
                if bubble_roi.size > 0:  # Ensure ROI is not empty
                    filled_score = np.count_nonzero(bubble_roi) / bubble_roi.size
                    
                    # Use bubble classifier if available
                    if bubble_classifier is not None:
                        try:
                            # Extract features for classification
                            bubble_features = extract_bubble_features(bubble_roi)
                            # Predict if bubble is filled using the classifier
                            is_filled_prob = bubble_classifier.predict_proba(bubble_features.reshape(1, -1))[0][1]
                            filled_score = is_filled_prob  # Use probability from classifier
                        except Exception as e:
                            print(f"Error using bubble classifier: {str(e)}")
                            # Fallback to default method if classifier fails
                    
                    # Check if this is the most filled bubble for this question
                    if filled_score > max_filled_score:
                        max_filled_score = filled_score
                        marked_option = i
            
            # Determine if a bubble is actually marked
            # Use threshold model if available, otherwise use static threshold
            marking_threshold = 0.45  # Default threshold
            if threshold_model is not None:
                try:
                    # The threshold model might provide a dynamic threshold based on image properties
                    # This is a placeholder for custom threshold logic
                    marking_threshold = 0.45  # Replace with threshold model logic
                except Exception as e:
                    print(f"Error using threshold model: {str(e)}")
            
            if max_filled_score > marking_threshold:
                marked_answers.append(marked_option)
                answer_confidences[row_idx] = int(max_filled_score * 100)  # Store confidence as percentage
            else:
                marked_answers.append(None)  # No answer marked for this question
                answer_confidences[row_idx] = 0
        
        # Draw circles and annotate the result image
        for row_idx, row in enumerate(valid_rows):
            for i, (x, y, r) in enumerate(row):
                # Extract the bubble region
                bubble_roi = thresh[y-r:y+r, x-r:x+r]
                
                # Calculate the percentage of filled pixels
                if bubble_roi.size > 0:
                    filled_score = np.count_nonzero(bubble_roi) / bubble_roi.size
                    
                    # Determine the color based on whether this bubble is marked
                    is_marked = (marked_answers[row_idx] == i)
                    
                    # Check if we have an answer key and this is the correct answer
                    is_correct_answer = False
                    if answer_key and row_idx < len(answer_key):
                        is_correct_answer = (answer_key[row_idx] == i)
                    
                    # Choose color based on answer correctness
                    if is_marked:
                        if answer_key and row_idx < len(answer_key):
                            if is_correct_answer:
                                color = (0, 255, 0)  # Green for correct answer
                            else:
                                color = (0, 0, 255)  # Red for incorrect answer
                        else:
                            color = (0, 255, 255)  # Yellow for marked answer (no key provided)
                        
                        # Fill the bubble if it's marked
                        cv2.circle(result_image, (x, y), r, color, 2)
                        cv2.circle(result_image, (x, y), r-3, color, -1)
                    else:
                        # For unmarked bubbles, only highlight if it's the correct answer
                        if answer_key and row_idx < len(answer_key) and is_correct_answer:
                            color = (255, 0, 0)  # Blue for correct answer that wasn't selected
                            cv2.circle(result_image, (x, y), r, color, 2)
                        else:
                            # Regular unmarked bubble
                            cv2.circle(result_image, (x, y), r, (100, 100, 100), 1)
                
                # Add option label
                option_letter = chr(65 + i)  # A, B, C, D...
                cv2.putText(result_image, option_letter, (x-5, y-r-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Add question number
            question_num = row_idx + 1
            cv2.putText(result_image, f"{question_num}", (row[0][0]-30, row[0][1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Calculate score if answer key is provided
        score = None
        if answer_key:
            correct_count = 0
            answer_count = 0
            
            for i, ans in enumerate(marked_answers):
                if i < len(answer_key) and ans is not None:
                    answer_count += 1
                    if ans == answer_key[i]:
                        correct_count += 1
            
            if answer_count > 0:
                score = (correct_count / len(answer_key)) * 100
                
                # Add score to the image
                cv2.putText(result_image, f"Score: {score:.1f}%", (30, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Add detailed score
                cv2.putText(result_image, f"Correct: {correct_count}/{len(answer_key)}", (30, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Display user's answers in text format
        answer_text = ""
        for i, ans in enumerate(marked_answers):
            if ans is not None:
                answer_text += f"Q{i+1}: {chr(65 + ans)}, "
        
        if answer_text:
            print(f"Marked answers: {answer_text[:-2]}")  # Remove trailing comma and space
        else:
            print("No answers detected")
        
        # Save the result
        cv2.imwrite(os.path.join(output_dir, "result_graded.jpg"), result_image)
        
        # Create a text result file
        with open(os.path.join(output_dir, "results.txt"), "w") as f:
            f.write("OMR SHEET RESULTS\n")
            f.write("=================\n\n")
            
            # Write all detected answers
            f.write("ANSWERS MARKED:\n")
            for i, ans in enumerate(marked_answers):
                if ans is not None:
                    f.write(f"Question {i+1}: {chr(65 + ans)}\n")
                else:
                    f.write(f"Question {i+1}: No answer\n")
            
            # Write score if available
            if score is not None:
                f.write("\nSCORE:\n")
                f.write(f"Correct answers: {correct_count}/{len(answer_key)}\n")
                f.write(f"Score: {score:.1f}%\n")
        
        return {
            "marked_answers": marked_answers,
            "score": score,
            "processed_image": result_image,
            "result_path": os.path.join(output_dir, "result_graded.jpg"),
            "results_text_path": os.path.join(output_dir, "results.txt"),
            "answer_confidences": answer_confidences
        }
    else:
        print("No circles detected!")
        return None

def generate_omr_template(output_path="omr_template.jpg", num_questions=10, num_options=4, include_name_field=True, include_roll_field=True):
    """Generate a template OMR sheet."""
    # Create a blank white image (simulating a paper)
    height, width = 900, 650
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Add title and header
    cv2.putText(image, "OMR Answer Sheet", (int(width/2)-120, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "Fill in your answers completely", (int(width/2)-120, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Draw horizontal line below the title
    cv2.line(image, (30, 80), (width-30, 80), (0, 0, 0), 1)

    # Create text fields for name, roll number, and date
    y_offset = 95
    
    # Name field
    if include_name_field:
        cv2.rectangle(image, (35, y_offset), (210, y_offset + 20), (0, 0, 0), 1)
        cv2.putText(image, "Name:", (40, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 30
    
    # Roll field
    if include_roll_field:
        cv2.rectangle(image, (35, y_offset), (210, y_offset + 20), (0, 0, 0), 1)
        cv2.putText(image, "Roll No:", (40, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 30
    
    # Date field
    cv2.rectangle(image, (35, y_offset), (210, y_offset + 20), (0, 0, 0), 1)
    cv2.putText(image, "Date:", (40, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Starting y-position for questions
    start_y = 160
    
    # Create questions with options
    for q in range(1, num_questions + 1):
        y_pos = start_y + ((q-1) * 30)
        
        # Question number
        cv2.putText(image, f"{q}.", (50, y_pos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Options
        for o in range(num_options):
            option_x = 100 + (o * 80)
            option_letter = chr(65 + o)  # A, B, C, D...
            
            # Draw circle (bubble)
            cv2.circle(image, (option_x, y_pos), 10, (0, 0, 0), 1)
            
            # Put option letter inside the circle
            cv2.putText(image, option_letter, (option_x-5, y_pos+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add instructions at the bottom
    instructions_y = start_y + (num_questions * 30) + 20
    cv2.putText(image, "Instructions:", (40, instructions_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(image, "1. Fill the circles completely with dark pen/pencil", (40, instructions_y+25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(image, "2. Erase any stray marks completely", (40, instructions_y+50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Save the template
    cv2.imwrite(output_path, image)
    print(f"Generated OMR template at {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="OMR Sheet Processing System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Template generation command
    template_parser = subparsers.add_parser("template", help="Generate an OMR template")
    template_parser.add_argument("-o", "--output", default="omr_template.jpg", help="Output path for template")
    template_parser.add_argument("-q", "--questions", type=int, default=10, help="Number of questions")
    template_parser.add_argument("-op", "--options", type=int, default=4, help="Number of options per question")
    template_parser.add_argument("-n", "--include-name", action="store_true", help="Include name field")
    template_parser.add_argument("-r", "--include-roll", action="store_true", help="Include roll number field")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process an OMR sheet")
    process_parser.add_argument("image", help="Path to the OMR sheet image")
    process_parser.add_argument("-k", "--key", help="Answer key (e.g. 'ABCDABCD')")
    process_parser.add_argument("-o", "--output", default="./results", help="Output directory for results")
    
    args = parser.parse_args()
    
    if args.command == "template":
        generate_omr_template(args.output, args.questions, args.options, args.include_name, args.include_roll)
    
    elif args.command == "process":
        # Convert answer key from letters to indices if provided
        answer_key = None
        if args.key:
            answer_key = []
            for letter in args.key.upper():
                if 'A' <= letter <= 'D':
                    answer_key.append(ord(letter) - ord('A'))
            print(f"Using answer key: {args.key.upper()}")
        
        # Process the OMR sheet
        result = process_omr_sheet(args.image, answer_key, args.output)
        
        if result:
            print(f"OMR processing complete. Results saved to: {args.output}")
            if result["score"] is not None:
                print(f"Score: {result['score']:.1f}%")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 