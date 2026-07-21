#!/usr/bin/env python3
try:
    from .build_two_stage_data import main
except ImportError:
    from build_two_stage_data import main


if __name__ == "__main__":
    main()
