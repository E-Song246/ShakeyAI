# Shakespeare LLM: A Stylistic Imitation Model

Welcome to the **Shakespeare LLM** project! This repository contains an advanced language model designed to generate stylistic imitations of William Shakespeare's works. By leveraging cutting-edge machine learning techniques and the Bard's timeless texts, this project aims to recreate the distinctive style of Shakespeare's comedies, tragedies, histories, and poems.

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
1. Place raw Shakespeare texts in the appropriate subfolders under `data/`.
2. Run the preprocessing script to tokenize and save the data in `.pt` format:
   ```bash
   python scripts/preprocess_data.py
   ```

### **3. Training the Model**
Use the training script to fine-tune your model:
```bash
python scripts/train_model.py --genre comedies
```

Options:
- `--genre`: Specify the genre (e.g., `comedies`, `tragedies`, `histories`, `poetry`).
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size for training.

### **4. Generating Text**
After training, use the generation script to create Shakespearean-style outputs:
```bash
python scripts/generate_text.py --genre tragedies --length 200
```

---

## **Examples**

### **Comedy**
> "What light through yonder curtain speaks? A jest, most merry! A fool, unworthy of sorrow."

### **Tragedy**
> "O wretched fate, to know the sting of treachery and wear the crown of despair."

### **History**
> "Upon this field, where blood and valor meet, the tale of kings shall never fade."

### **Poetry**
> "Thy heart is a vessel, tempest-tossed, yet steadfast in the gale of eternity."

---

## **Contributing**
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with detailed explanations of your changes.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Acknowledgments**
- **William Shakespeare**: The eternal inspiration for this project.
- **PyTorch Community**: For their robust and versatile ML framework.
- **You**: For your interest and contributions to the Shakespeare LLM project.

---

## **Contact**
For questions, suggestions, or issues, please reach out to Evan at [your email here].

