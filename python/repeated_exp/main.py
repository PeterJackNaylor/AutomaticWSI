
import sys
from model_selection import ncv, ncv_single, repeat


if __name__ == '__main__':
    if sys.argv[1] == "repeat":
        repeat(sys.argv[2])
    elif sys.argv[1] == "plot":
        plot(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "ncv":
        ncv()
    elif sys.argv[1] == "ncv_single":
        
        ncv_single(int(sys.argv[2]), int(sys.argv[3]))
    else:
        print(f"Unknown command {sys.argv[1]}")
