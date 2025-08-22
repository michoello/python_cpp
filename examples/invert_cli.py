#!/usr/bin/env python3
import sys
from listinvert import invert, Matrix

def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/invert_cli.py <int1> <int2> ...")
        sys.exit(1)

    try:
        numbers = [int(x) for x in sys.argv[1:]]
    except ValueError:
        print("Error: all arguments must be integers")
        sys.exit(1)

    inverted = invert(numbers)
    print("Original:", numbers)
    print("Inverted:", inverted)


def matrix_demo():
    A = Matrix(rows = 2, cols = 3,
      values=[
        [1, 2, 4],
        [4, 5, 6]
      ])
    B = Matrix([
        [7, 8],
        [9, 10],
        [11, 12]
    ])

    # Multiply A * B (handled in C++)
    C = A.multiply(B)

    # Get the result as Python list-of-lists
    result = C.value()

    # Print nicely
    print("Matrix A:")
    for row in A.value():
        print(row)

    print("\nMatrix B:")
    for row in B.value():
        print(row)

    print("\nResult (A * B):")
    for row in result:
        print(row)





if __name__ == "__main__":
    main()
    matrix_demo()
