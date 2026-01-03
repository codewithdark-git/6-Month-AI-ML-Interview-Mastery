# 🎤 Master Interview Prep: Tensors & Linear Algebra

This document is structured as a comprehensive Q&A bank to prepare you for technical interviews. Every concept has been converted into an interview-style question.

---



## 🏗️ Section 1: Tensors & Linear Algebra 

### 📚 Notes: [Tensor Notes](./notes/tensors.md)
### Code Examples: [Tensor Code Examples](./code/tensors.ipynb)

### ❓ Q1: What is a Tensor in plain terms?
**Answer:**
A **tensor** is a structured container for numerical data. It generalizes the concepts of scalars, vectors, and matrices to any number of dimensions. Tensors are defined by their **structure** (arrangement of axes) rather than just the numbers they hold.

### ❓ Q2: Why do we use Tensors in Machine Learning instead of just lists or tables?
**Answer:**
Real-world data is multi-dimensional (e.g., images are $H \times W \times C$, videos are $Batch \times Frames \times Channels \times H \times W$). Tensors allow us to organize, process, and perform complex mathematical operations on this high-dimensional data efficiently, especially on hardware like GPUs.

---

### ❓ Q3: What is the "Rank" of a tensor, and how does it differ from "Dimension"?
**Answer:**
- **Rank (Order)**: The number of axes or dimensions a tensor has. (e.g., a matrix has rank 2).
- **Dimension**: Typically refers to the size along a specific axis.
- **Example**: A tensor with shape `(32, 100)` has a **Rank of 2** and a **Dimension of 100** along the second axis.

### ❓ Q4: Can you describe the Rank hierarchy from 0 to 5?
**Answer:**
- **Rank-0 (Scalar)**: A single number (magnitude only).
- **Rank-1 (Vector)**: A list of numbers (direction + magnitude).
- **Rank-2 (Matrix)**: A table of numbers (used for linear transformations).
- **Rank-3 (Tensor)**: A stack of matrices (e.g., batched grayscale images).
- **Rank-4 (Tensor)**: Very common for color images `(Batch, Channels, Height, Width)`.
- **Rank-5 (Tensor)**: Typically used for video data `(Batch, Frames, Channels, Height, Width)`.

### ❓ Q5: Why is a 3D Tensor not the same as a 3D Vector?
**Answer:**
A 3D vector is a **Rank-1** tensor with 3 components (representing a single arrow in space). A 3D tensor is a **Rank-3** data structure (like a cube of numbers) used to store multi-dimensional data like pixel values. One represents a geometric direction; the other represents a structured data container.

---


### ❓ Q6: What is the relationship between a Vector and a Coordinate System?
**Answer:**
A vector is a geometric object (an arrow with direction and magnitude). The numbers we see (e.g., `[1, 2, 3]`) are simply the **coordinates** of that vector relative to a specific coordinate system. If you change the coordinate system, the numbers change, but the physical arrow stays the same.

### ❓ Q7: Is there a functional difference between Row and Column Vectors?
**Answer:**
Mathematically, they represent the same abstract vector. However, in ML and Linear Algebra, **Column Vectors** are the standard convention because they align naturally with matrix multiplication rules ($Ax = b$), where $A$ is a weight matrix and $x$ is the input vector.

---

### ❓ Q8: How should we conceptually understand tensors with Rank > 3?
**Answer:**
We don't try to visualize them geometrically. Instead, we understand them algebraically through **Slicing, Indexing, and Reshaping**. They are data structures where each axis represents a specific feature of the data (Batch, Time, Channel, etc.).

### ❓ Q9: What is "Shape Compatibility" (Broadcasting)?
**Answer:**
When performing operations on tensors of different shapes, PyTorch/NumPy uses broadcasting. Two dimensions are compatible if:
1. They are equal.
2. One of them is `1`.
This allows a smaller tensor to be logically "stretched" to match a larger one for element-wise calculation.

---

### ❓ Q10: What is the most common mistake candidates make regarding tensors?
**Answer:**
Confusing the **geometric dimension** of space (like 3D space) with the **Rank** of a tensor. For example, assuming a Rank-4 tensor represents a direction in "4D space" when it actually just stores a batch of images.

### ❓ Q11: In a CNN, what does a tensor of shape `(32, 3, 224, 224)` represent?
**Answer:**
It represents a **Batch of 32 color images**. Each image has **3 RGB channels**, with a resolution of **224x224 pixels**.

### ❓ Q12: Why are Tensors the core of Deep Learning frameworks?
**Answer:**
They provide:
1. **Efficient Storage** for structured data.
2. **Parallel Computation** support for GPUs.
3. **Automatic Differentiation** (Autograd) for calculating gradients during training.

---

## 📝 One-Line Memory Hook
> **"Rank defines the number of axes; Shape defines the size of those axes; Data defines the values within."**

---

