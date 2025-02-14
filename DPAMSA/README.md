## 📌 **DPAMSA: Deep Reinforcement Learning for Multiple Sequence Alignment**
**DPAMSA** is a **deep reinforcement learning (DRL)-based framework** for **Multiple Sequence Alignment (MSA)**. It leverages **self-attention mechanisms** and **positional encoding** to enhance alignment accuracy while using **Deep Q-Network (DQN)** for reinforcement learning.

### **🔬 Research Paper**
This project is based on the research described in:
> **Yuhang Liu et al., "Multiple sequence alignment based on deep reinforcement learning with self-attention and positional encoding," Bioinformatics, 2023.**  
> 📄 **DOI:** [10.1093/bioinformatics/btad636](https://doi.org/10.1093/bioinformatics/btad636)  
> 📂 **Source Code Repository:** [GitHub: DPAMSA](https://github.com/ZhangLab312/DPAMSA)  

---
## 🏗 **Technical Components**
### **1️⃣ Transformer-Based Encoder (`models.py`)**
- Implements **self-attention** for feature extraction.
- Uses **positional encoding** to retain sequence order.
- **Multi-Layer Perceptron (MLP)** for final decision-making.

### **2️⃣ Deep Q-Network (DQN) (`dqn.py`)**
- Uses **ε-greedy policy** for action selection.
- Implements **experience replay** and **Q-learning updates**.
- Stores past experiences and trains a **Q-value network**.

### **3️⃣ Replay Memory (`replay_memory.py`)**
- Implements an **experience buffer** for reinforcement learning.
- Stores past `(state, action, reward, next_state, done)` tuples.
- Uses **random sampling** for stable learning.

### **4️⃣ MSA Reinforcement Learning Environment (`env.py`)**
- Converts DNA sequences into **numerical format**.
- Implements **progressive column alignment**.
- Computes **alignment scores** dynamically.
- Provides **reward signals** for reinforcement learning.
---
## 📦 **Installation**
### **📥 Clone the Repository**
```bash
git clone https://github.com/FLaTNNBio/GA-DPAMSA
cd DPAMSA
```

### **🔧 Requirements**
Ensure you have Python **3.8+** and install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## 🎯 **Running the DPAMSA Framework**

### **1️⃣ Load the dataset**
- In `main.py` change the imported dataset as you like
```bash
import datasets.[training or inference subfolder].[dataset name] as dataset
```
### **2️⃣ Adjust the Hyperparameters (Training) or load a saved model (Inference)**
- Modify `config.py` to adjust hyperparameters:
```python
MAX_EPISODE = 5000       # Total training episodes
BATCH_SIZE = 128         # Batch size for training
REPLAY_MEMORY_SIZE = 10000  # Memory buffer size
ALPHA = 0.001            # Learning rate
EPSILON = 0.9            # Initial epsilon (ε-greedy strategy)
GAMMA = 0.99             # Discount factor for Q-learning
DEVICE_NAME = "cuda"     # Use "cuda" for GPU, "cpu" for CPU
```
- Set the model to load in `main.py`: 
```bash
INFERENCE_MODEL = 'saved_model'  # File name without extension
```

### **3️⃣ Run `main.py`**
- Running `main.py` opens up a selection menu in the terminal where you can choose between training and inference

#### **Training**:
- Loads the dataset and trains a **Deep Q-Network (DQN)**.
- Uses **reinforcement learning** to optimize sequence alignments.
- Saves model weights for inference.

#### **Inference**:
- Loads a **pre-trained model** to generate alignments.
- Evaluates the model using **SP Score** and **Column Score**.

---

## 📝 **References**
If you use DPAMSA in your research, please cite:
```
@article{liu2023dpamsa,
  title={Multiple sequence alignment based on deep reinforcement learning with self-attention and positional encoding},
  author={Yuhang Liu, Hao Yuan, Qiang Zhang, Zixuan Wang, Shuwen Xiong, Naifeng Wen, Yongqing Zhang},
  journal={Bioinformatics},
  year={2023},
  doi={10.1093/bioinformatics/btad636}
}
```

---

## 📜 **License**
This project is licensed under the **MIT License**.  
See the **LICENSE** file for details.