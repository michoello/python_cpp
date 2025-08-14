#!/usr/bin/env python3
import sys
from listinvert import invert

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

if __name__ == "__main__":
    main()
