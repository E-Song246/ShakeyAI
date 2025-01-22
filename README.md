# Shakey AI: A Stylistic Imitation Model

Welcome to the **Shakey AI** project! This repository contains an advanced language model designed to generate stylistic imitations of William Shakespeare's works. By leveraging cutting-edge machine learning techniques and the Bard's timeless texts, this project aims to recreate the distinctive style of Shakespeare's comedies, tragedies, histories, and poems.

---

## **Features**
- **Category-Specific Text Generation**: Generate outputs tailored to one of four Shakespearean genres:
  - **Comedies**
  - **Tragedies**
  - **Histories**
  - **Poetry**
- **Custom Dataset**: Trained on meticulously organized and tokenized text files from Shakespeare's complete works.
- **Torch Implementation**: Built using PyTorch for flexibility, scalability, and integration with other ML tools.
- **Scalable Subfolder Structure**: Separate subfolders for each genre and preprocessed `.pt` files for efficient model training and inference.

---

## **Getting Started**

### **1. Prerequisites**
Ensure you have the following installed:
- Python 3.9+
- PyTorch 2.0+
- Required dependencies (install via `requirements.txt`)

```bash
pip install -r requirements.txt
```

### **2. Data Preparation**
1. Run the scraping script (shakescraper.py) to create the shakespeare_works folder
2. Run the tokenizing script (shaketoken.py) to tokenize and save the data in `.pt` format:
3. Alternatively, simply extract the shakespeare_works.zip folder in lieu of the previous steps

### **3. Training the Model**
Use the training script (shaketrain.py) to fine-tune your model

### **4. Generating Text**
After training, use the generation script (shakeyai.py) to create Shakespearean-style outputs

---

## **Contributing**
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with detailed explanations of your changes.

---

## **Acknowledgments**
- **William Shakespeare**: The eternal inspiration for this project.
- **PyTorch Community**: For their robust and versatile ML framework.
- **MIT**: Scraping was only possible because of their dedicated Shakespeare collection
- **ChatGPT**: A brilliant assistant for debugging and generating code outlines

---

## **Contact**
For questions, suggestions, or issues, please reach out to Evan at esong246@umd.edu.

