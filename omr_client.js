const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

/**
 * OMR Scanner API Client for Node.js
 * Allows easy integration with the OMR scanner from JavaScript applications
 */
class OMRScannerClient {
  /**
   * Create a new OMR Scanner client
   * @param {string} baseUrl - The base URL of the OMR Scanner API
   */
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
  }

  /**
   * Process a single OMR sheet
   * @param {string} filePath - Path to the OMR sheet image
   * @param {string} answerKey - Optional answer key (e.g., "ABCDABCD")
   * @returns {Promise<object>} - The processing results
   */
  async processSingleSheet(filePath, answerKey = null) {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(filePath));
    
    if (answerKey) {
      formData.append('answer_key', answerKey);
    }
    
    try {
      const response = await axios.post(
        `${this.baseUrl}/api/process-omr`,
        formData,
        {
          headers: {
            ...formData.getHeaders(),
          },
        }
      );
      
      return response.data;
    } catch (error) {
      throw new Error(`API request failed: ${error.message}`);
    }
  }

  /**
   * Extract only student information from an OMR sheet
   * @param {string} filePath - Path to the OMR sheet image
   * @returns {Promise<object>} - The student information
   */
  async extractStudentInfo(filePath) {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(filePath));
    
    try {
      const response = await axios.post(
        `${this.baseUrl}/api/extract-student-info`,
        formData,
        {
          headers: {
            ...formData.getHeaders(),
          },
        }
      );
      
      return response.data;
    } catch (error) {
      throw new Error(`API request failed: ${error.message}`);
    }
  }

  /**
   * Process multiple OMR sheets in a single request
   * @param {string[]} filePaths - Array of paths to OMR sheet images
   * @param {string} answerKey - Optional answer key (e.g., "ABCDABCD")
   * @param {boolean} extractStudentInfo - Whether to extract student information
   * @returns {Promise<object>} - The batch processing results
   */
  async batchProcess(filePaths, answerKey = null, extractStudentInfo = true) {
    const formData = new FormData();
    
    // Add all files to the form data
    filePaths.forEach(filePath => {
      formData.append('files', fs.createReadStream(filePath));
    });
    
    if (answerKey) {
      formData.append('answer_key', answerKey);
    }
    
    formData.append('extract_student_info', extractStudentInfo);
    
    try {
      const response = await axios.post(
        `${this.baseUrl}/api/batch-process`,
        formData,
        {
          headers: {
            ...formData.getHeaders(),
          },
        }
      );
      
      return response.data;
    } catch (error) {
      throw new Error(`API request failed: ${error.message}`);
    }
  }

  /**
   * Generate an OMR template
   * @param {object} options - Template options
   * @param {number} options.numQuestions - Number of questions
   * @param {number} options.numOptions - Number of options per question
   * @param {boolean} options.includeName - Whether to include a name field
   * @param {boolean} options.includeRoll - Whether to include a roll number field
   * @param {string} outputPath - Path to save the template
   * @returns {Promise<string>} - The path where the template was saved
   */
  async generateTemplate(options = {}, outputPath) {
    const {
      numQuestions = 20,
      numOptions = 4,
      includeName = true,
      includeRoll = true,
    } = options;
    
    try {
      const response = await axios.get(
        `${this.baseUrl}/api/generate-template?num_questions=${numQuestions}&num_options=${numOptions}&include_name=${includeName}&include_roll=${includeRoll}`,
        { responseType: 'stream' }
      );
      
      // Save the template to the output path
      const writer = fs.createWriteStream(outputPath);
      response.data.pipe(writer);
      
      return new Promise((resolve, reject) => {
        writer.on('finish', () => resolve(outputPath));
        writer.on('error', reject);
      });
    } catch (error) {
      throw new Error(`API request failed: ${error.message}`);
    }
  }
}

// Example usage
async function main() {
  const client = new OMRScannerClient('http://localhost:8000');
  
  try {
    // Example 1: Process a single sheet
    const result = await client.processSingleSheet(
      'path/to/omr_sheet.jpg',
      'ABCDABCD'
    );
    console.log('Single sheet result:', result);
    
    // Example 2: Extract only student information
    const studentInfo = await client.extractStudentInfo('path/to/omr_sheet.jpg');
    console.log('Student info:', studentInfo);
    
    // Example 3: Batch process multiple sheets
    const batchResult = await client.batchProcess(
      ['path/to/sheet1.jpg', 'path/to/sheet2.jpg'],
      'ABCDABCD'
    );
    console.log('Batch result:', batchResult);
    
    // Example 4: Generate and save a template
    const templatePath = await client.generateTemplate(
      { numQuestions: 30 },
      'omr_template.jpg'
    );
    console.log('Template saved to:', templatePath);
  } catch (error) {
    console.error('Error:', error.message);
  }
}

// Comment out to use as a module
// main();

module.exports = OMRScannerClient; 