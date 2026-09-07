"""Compatibility entry point for the framework smoke demonstration."""

from demo import demo_basic_usage

if __name__ == "__main__":
    demo_basic_usage()
    print("\nScientific interpretation:")
    print("- Random-batch training loss is not a forgetting metric.")
    print("- Compare every learned task after each training stage.")
    print("- Report retention and new-task plasticity separately when evaluating EWC.")
