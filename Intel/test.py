import os
import multiprocessing

def check_user():
    print(f"Process UID: {os.getuid()}, GID: {os.getgid()}")
    print(f"Effective UID: {os.geteuid()}, GID: {os.getegid()}")

# Check main process
print("Main process:")
check_user()

# Check spawned process
if __name__ == '__main__':
    p = multiprocessing.Process(target=check_user)
    p.start()
    p.join()