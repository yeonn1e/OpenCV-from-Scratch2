import numpy as np
import cv2

smile = cv2.imread('smile.png' , cv2.IMREAD_GRAYSCALE)

def get_transformed_image(img, M):
    
    plane = np.full((801, 801), 255, dtype=np.float32)
    center_w, center_h = img.shape[1]//2, img.shape[0]//2

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            point = np.array([y-center_w, x-center_h, 1])
            new_point = M.dot(point)
            new_x, new_y, z = map(int, new_point/new_point[2]+400) 
            plane[new_y,new_x] = img[x,y]

    cv2.arrowedLine(plane, (0, 400), (801, 400), 0, thickness=2, tipLength=0.03) # x축
    cv2.arrowedLine(plane, (400, 801), (400, 0), 0, thickness=2, tipLength=0.03) # y축

    return plane

m = np.eye(3)

w = np.array([[1, 0, 0],
              [0, 1, -5],
              [0, 0, 1]])

s = np.array([[1, 0, 0],
              [0, 1, 5],
              [0, 0, 1]])

a = np.array([[1, 0, -5],
              [0, 1, 0],
              [0, 0, 1]])

d = np.array([[1, 0, 5],
              [0, 1, 0],
              [0, 0, 1]])

t = np.array([[np.cos(5*np.pi / 180), -np.sin(5*np.pi / 180), 0],
              [np.sin(5*np.pi / 180), np.cos(5*np.pi / 180), 0],
              [0, 0, 1]])

r = np.array([[np.cos(-5* np.pi / 180), -np.sin(-5*np.pi / 180), 0],
              [np.sin(-5* np.pi / 180), np.cos(-5*np.pi / 180), 0],
              [0, 0, 1]])

g = np.array([[1, 0, 0],
              [0, -1, 0],
              [0, 0, 1]])

f = np.array([[-1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

y = np.array([[1, 0, 0],
              [0, 0.95, 0],
              [0, 0, 1]])

u = np.array([[1, 0, 0],
              [0, 1.05, 0],
              [0, 0, 1]])

x = np.array([[0.95, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

c = np.array([[1.05, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

M = np.eye(3)
plane = get_transformed_image(smile, m)

while True:
    cv2.imshow('2d transformations', plane)
    key = cv2.waitKey()

    if key == ord('a'):
        M = a.dot(M)
    elif key == ord('d'):
        M = d.dot(M)
    elif key == ord('w'):
        M = w.dot(M)
    elif key == ord('s'):
        M = s.dot(M)
    elif key == ord('r'):
        M = r.dot(M)
    elif key == ord('t'):
        M = t.dot(M)
    elif key == ord('f'):
        M = f.dot(M)
    elif key == ord('g'):
        M = g.dot(M)
    elif key == ord('x'):
        M = x.dot(M)
    elif key == ord('c'):
        M = c.dot(M)
    elif key == ord('y'):
        M = y.dot(M)
    elif key == ord('u'):
        M = u.dot(M)
    elif key == ord('h'):
        M = np.eye(3)
    elif key == ord('q'):
        cv2.destroyAllWindows()
        break
    plane = get_transformed_image(smile, M)