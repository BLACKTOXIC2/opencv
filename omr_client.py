import requests
import json
import os
from typing import List, Dict, Optional, Union, BinaryIO


class OMRScannerClient:
    """
    Python client for the OMR Scanner API
    Allows external applications to easily integrate with the OMR scanner
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the OMR Scanner client
        
        Args:
            base_url: The base URL of the OMR Scanner API
        """
        self.base_url = base_url.rstrip('/')
    
    def process_single_sheet(self, 
                             file_path: str, 
                             answer_key: Optional[str] = None,
                             extract_student_info: bool = True) -> dict:
        """
        Process a single OMR sheet
        
        Args:
            file_path: Path to the OMR sheet image file
            answer_key: Optional answer key (e.g., "ABCDABCD")
            extract_student_info: Whether to extract student information
            
        Returns:
            Dictionary containing the processing results
        """
        endpoint = f"{self.base_url}/api/process-omr"
        
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'image/jpeg')}
            data = {}
            
            if answer_key:
                data['answer_key'] = answer_key
                
            response = requests.post(endpoint, files=files, data=data)
            
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    
    def extract_student_info(self, file_path: str) -> dict:
        """
        Extract only student information from an OMR sheet
        
        Args:
            file_path: Path to the OMR sheet image file
            
        Returns:
            Dictionary containing the student information
        """
        endpoint = f"{self.base_url}/api/extract-student-info"
        
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'image/jpeg')}
            response = requests.post(endpoint, files=files)
            
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    
    def batch_process(self, 
                      file_paths: List[str], 
                      answer_key: Optional[str] = None,
                      extract_student_info: bool = True) -> dict:
        """
        Process multiple OMR sheets in a single request
        
        Args:
            file_paths: List of paths to OMR sheet image files
            answer_key: Optional answer key (e.g., "ABCDABCD")
            extract_student_info: Whether to extract student information
            
        Returns:
            Dictionary containing the processing results for all sheets
        """
        endpoint = f"{self.base_url}/api/batch-process"
        
        files = []
        for file_path in file_paths:
            files.append(('files', (os.path.basename(file_path), 
                                   open(file_path, 'rb'), 
                                   'image/jpeg')))
        
        data = {'extract_student_info': 'true' if extract_student_info else 'false'}
        if answer_key:
            data['answer_key'] = answer_key
            
        response = requests.post(endpoint, files=files, data=data)
        
        # Close all open file handles
        for _, (_, file_obj, _) in enumerate(files):
            file_obj.close()
            
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    
    def generate_template(self, 
                         num_questions: int = 20, 
                         num_options: int = 4, 
                         include_name: bool = True,
                         include_roll: bool = True,
                         download_path: Optional[str] = None) -> Union[str, bytes]:
        """
        Generate an OMR template
        
        Args:
            num_questions: Number of questions
            num_options: Number of options per question
            include_name: Whether to include a name field
            include_roll: Whether to include a roll number field
            download_path: Path to save the template (if None, returns bytes)
            
        Returns:
            Path where template was saved (if download_path provided) or bytes
        """
        endpoint = (f"{self.base_url}/api/generate-template"
                  f"?num_questions={num_questions}"
                  f"&num_options={num_options}"
                  f"&include_name={include_name}"
                  f"&include_roll={include_roll}")
        
        response = requests.get(endpoint)
        
        if response.status_code == 200:
            if download_path:
                with open(download_path, 'wb') as f:
                    f.write(response.content)
                return download_path
            else:
                return response.content
        else:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")


# Usage example
if __name__ == "__main__":
    # Initialize the client
    client = OMRScannerClient("http://localhost:8000")
    
    # Example 1: Process a single sheet
    result = client.process_single_sheet(
        "path/to/omr_sheet.jpg", 
        answer_key="ABCDABCD"
    )
    print(json.dumps(result, indent=2))
    
    # Example 2: Extract only student information
    student_info = client.extract_student_info("path/to/omr_sheet.jpg")
    print(json.dumps(student_info, indent=2))
    
    # Example 3: Batch process multiple sheets
    batch_result = client.batch_process(
        ["path/to/sheet1.jpg", "path/to/sheet2.jpg"],
        answer_key="ABCDABCD"
    )
    print(json.dumps(batch_result, indent=2))
    
    # Example 4: Generate and save a template
    template_path = client.generate_template(
        num_questions=30,
        download_path="omr_template.jpg"
    )
    print(f"Template saved to: {template_path}") 