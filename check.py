import cv2

img = cv2.imread("photos/pm.jpg")

if img is not None:
    height, width, _ = img.shape
    if height > 0 and width > 0:
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image size is invalid.")
else:
    print("Image not found.")
