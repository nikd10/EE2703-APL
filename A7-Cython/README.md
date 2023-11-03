# Matrix Multiplication Optimization with Cython

This Python notebook contains the code and documentation for the assignment on optimizing matrix multiplication using Cython. The assignment involved several steps to measure and enhance the performance of matrix multiplication in Python.

## Assignment Overview

In this assignment, we explored techniques to optimize the multiplication of two matrices. We started with a basic Python implementation and progressively applied Cython optimizations to achieve better performance. The goal was to compare the execution times of various approaches and understand the impact of different optimizations.

The assignment consists of the following steps:

1. **Step 1: Baseline Measurement**
   - Measure the time required for matrix multiplication in Python.
   - Use `%timeit` to compare the performance with NumPy's `matmul`.
   - Estimate the FLOPS achieved by the system.

2. **Step 2: CPU Frequency Analysis**
   - Use the `lscpu` command to find the maximum operating frequency of the CPU.
   - Estimate the maximum achievable FLOPS for a single processor core.
   - Compare these estimates with the Step 1 results.

3. **Step 3: Matrix Size Scaling**
   - Repeatedly measure execution times for matrix multiplication while doubling the matrix sizes.
   - Plot measured times for Python code and NumPy's `matmul` and analyze the results.
   - Observe how large matrix sizes can be handled with NumPy.
   - Compare estimated GFLOPS with theoretical estimates.

4. **Step 4: Cython Integration and Optimization**
   - Integrate Cython into the Python code to optimize matrix multiplication.
   - Apply various enhancements, including decorators and variable type declarations.
   - Measure execution times and compare them with the original Python code and NumPy's `matmul`.

5. **Step 5: Advanced Transformations and Performance Analysis**
   - Experiment with advanced transformations to further optimize the Cython code.
   - Measure the running time for each transformation and analyze the observations.
   - Determine which transformation resulted in the best improvement and provide insights.
