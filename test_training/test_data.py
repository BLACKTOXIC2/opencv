import os
import random
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

def create_mcq_template(file_name, num_questions=10):
    """Create an MCQ template matching the specified format"""
    # Create a blank white image (A4 proportions)
    width, height = 800, 1100
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to load fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        normal_font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        title_font = ImageFont.load_default()
        normal_font = ImageFont.load_default()
    
    # Draw title
    draw.text((width//2, 20), "MCQ Test Sheet", fill="black", font=title_font, anchor="mt")
    draw.text((width//2, 50), "Class 10 - Section A", fill="black", font=normal_font, anchor="mt")
    
    # Draw horizontal line
    draw.line([(50, 80), (width-50, 80)], fill="black", width=1)
    
    # Draw input fields
    field_y = 100
    field_height = 30
    
    # Name field
    draw.text((60, field_y), "Name:", fill="black", font=normal_font)
    draw.line([(120, field_y+field_height), (300, field_y+field_height)], fill="black", width=1)
    
    # Roll field
    draw.text((320, field_y), "Roll No.:", fill="black", font=normal_font)
    draw.line([(390, field_y+field_height), (500, field_y+field_height)], fill="black", width=1)
    
    # Date field
    draw.text((520, field_y), "Date:", fill="black", font=normal_font)
    draw.line([(570, field_y+field_height), (700, field_y+field_height)], fill="black", width=1)
    
    # Draw MCQ bubbles
    start_y = 160
    question_spacing = 25
    bubble_radius = 8
    
    for q in range(num_questions):
        y = start_y + (q * question_spacing)
        options = ['A', 'B', 'C', 'D']
        
        # Question number
        draw.text((40, y), f"{q+1}.", fill="black", font=normal_font)
        
        # Randomly choose which bubble to fill
        filled_option = random.choice(options)
        
        # Draw options horizontally
        x_start = 60
        for i, option in enumerate(options):
            # Calculate bubble position
            bubble_x = x_start + (i * 30)
            bubble_y = y
            
            # Draw bubble
            if option == filled_option:
                # Draw filled bubble (outline in black, fill in white)
                draw.ellipse([(bubble_x, bubble_y), (bubble_x+bubble_radius, bubble_y+bubble_radius)], 
                           outline="black", fill="white")
            else:
                # Draw empty bubble
                draw.ellipse([(bubble_x, bubble_y), (bubble_x+bubble_radius, bubble_y+bubble_radius)], 
                           outline="black")
            
            # Draw option letter
            draw.text((bubble_x + bubble_radius + 2, bubble_y), option, fill="black", font=normal_font)
    
    # Save the image
    image.save(file_name)
    print(f"Created: {file_name}")

def main():
    print("\nMCQ Template Generator")
    print("-" * 20)
    
    while True:
        try:
            # Get number of templates from user
            num_templates = input("\nHow many MCQ templates do you want to generate? ")
            if not num_templates.strip():
                print("Please enter a number.")
                continue
                
            num_templates = int(num_templates)
            if num_templates <= 0:
                print("Please enter a positive number.")
                continue
            
            # Create output directory
            output_dir = os.path.join(os.path.dirname(__file__), "mcq_templates")
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"\nGenerating {num_templates} templates...")
            
            # Generate templates
            for i in range(1, num_templates + 1):
                template_path = os.path.join(output_dir, f"mcq_template_{i}.png")
                create_mcq_template(template_path)
                print(f"Progress: {i}/{num_templates} completed")
            
            print(f"\nSuccess! Created {num_templates} templates in:")
            print(f"'{output_dir}'")
            break
            
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()