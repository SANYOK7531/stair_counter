"""
Run the zone calibrator.
  python calibrate.py videos/stairs.mp4
  python calibrate.py 0   # camera
"""
from app.zone_calibrator import main

if __name__ == "__main__":
    main()
