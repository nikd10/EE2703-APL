# Traveling Salesman Problem Solver

This assignment implements a solution to the Traveling Salesman Problem (TSP) using the simulated annealing algorithm. It allows you to find an approximate solution to the TSP by optimizing the order in which a salesman visits a set of cities.

## Problem Statement

The TSP gives you a set of city locations represented as (x, y) coordinates. The goal is to find a route that visits all cities exactly once and returns to the origin while minimizing the total distance traveled.

## Approach

I have implemented a solution using the simulated annealing algorithm, which is a heuristic approach for optimization problems. Simulated annealing starts with a random initial solution and iteratively explores neighboring solutions. It probabilistically accepts worse solutions in the early stages and gradually refines the solution by decreasing the probability of accepting worse solutions.
