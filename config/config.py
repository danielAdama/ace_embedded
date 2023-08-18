import os

VIDEO_PATH = r'/home/chuby/Desktop/programming/computerVision/ace_embedded/video'
VIDEO = os.path.join(VIDEO_PATH, 'footage.mp4')
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
COLORS = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(10)]
ALLOWED_IDS = [0, 2]